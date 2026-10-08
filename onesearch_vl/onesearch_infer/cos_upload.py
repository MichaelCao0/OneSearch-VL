"""Load an asset publisher implementing the ``upload_cos`` callable contract.

The bundled ``upload.py`` publishes files through ASSET_DIR and ASSET_BASE_URL.
Custom publishers can be supplied through COS_UPLOAD_PATHS. If publication is
unavailable, image search reports an error and visual reasoning can use inline
image bytes.
"""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing as mp
import os
import sys
import tempfile
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from . import config


logger = logging.getLogger(__name__)


_UploadFunc = Callable[..., Tuple[Optional[str], Optional[str]]]
_upload_cos: Optional[_UploadFunc] = None
_attempted_paths: List[str] = []


def _upload_worker(upload_fn, args, kwargs, result_pipe) -> None:
    """Run one uploader call in an expendable child process."""

    try:
        result_pipe.send((True, upload_fn(*args, **kwargs)))
    except BaseException as exc:
        result_pipe.send((False, f"{type(exc).__name__}: {exc}"))
    finally:
        result_pipe.close()


def _call_upload(*args, **kwargs) -> Tuple[Optional[str], Optional[str]]:
    """Call the uploader directly, or in a killable eval-only subprocess."""

    if _upload_cos is None:
        return None, None
    seconds = float(os.environ.get("COS_UPLOAD_TOTAL_TIMEOUT", "0") or 0)
    if seconds <= 0:
        return _upload_cos(*args, **kwargs)
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("COS upload hard timeout requires multiprocessing fork")

    ctx = mp.get_context("fork")
    result_pipe, child_pipe = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_upload_worker,
        args=(_upload_cos, args, kwargs, child_pipe),
        daemon=True,
    )
    process.start()
    child_pipe.close()
    try:
        if not result_pipe.poll(seconds):
            raise TimeoutError(f"COS upload exceeded total timeout of {seconds:g}s")
        ok, payload = result_pipe.recv()
        if not ok:
            raise RuntimeError(f"COS upload subprocess failed: {payload}")
        return payload
    finally:
        result_pipe.close()
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=5)


def _candidate_paths() -> List[str]:
    """Return the ordered list of directories to search for ``upload.py``."""

    candidates: List[str] = []
    here = os.path.dirname(os.path.abspath(__file__))
    # An explicitly configured publisher must take precedence over the bundle.
    if config.COS_UPLOAD_PATHS:
        for entry in config.COS_UPLOAD_PATHS.split(":"):
            entry = entry.strip()
            if entry:
                candidates.append(entry)
    candidates.append(os.path.dirname(here))  # parent of onesearch_infer/
    candidates.append(here)
    # Deduplicate while keeping order.
    seen = set()
    ordered: List[str] = []
    for path in candidates:
        if path and path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def _load_upload_module(path: str) -> Optional[_UploadFunc]:
    upload_file = os.path.join(path, "upload.py")
    if not os.path.exists(upload_file):
        return None
    try:
        spec = importlib.util.spec_from_file_location("upload", upload_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate = getattr(module, "upload_cos", None)
        if callable(candidate):
            return candidate
    except Exception as exc:
        logger.warning(
            "Skipping upload.py at %s due to import error: %s",
            upload_file,
            exc,
        )
    # Fallback: add to sys.path and ``import upload`` directly. Some
    # legacy uploaders rely on relative imports inside their package.
    try:
        if path not in sys.path:
            sys.path.insert(0, path)
        import upload  # type: ignore
        candidate = getattr(upload, "upload_cos", None)
        if callable(candidate):
            return candidate
    except Exception as exc:
        logger.warning(
            "Direct import of upload.py at %s failed: %s", upload_file, exc
        )
    return None


def _ensure_loaded() -> None:
    """Discover and bind the external ``upload_cos`` once."""

    global _upload_cos, _attempted_paths

    if _upload_cos is not None or _attempted_paths:
        return

    paths = _candidate_paths()
    for path in paths:
        loaded = _load_upload_module(path)
        if loaded is not None:
            _upload_cos = loaded
            logger.info("Loaded COS uploader from %s", path)
            break
    _attempted_paths = paths

    if _upload_cos is None:
        logger.info(
            "COS uploader not available. Image upload will be skipped. "
            "Searched: %s",
            ", ".join(paths),
        )


def upload_available() -> bool:
    """Return ``True`` when an external ``upload_cos`` was discovered."""

    _ensure_loaded()
    return _upload_cos is not None


def upload_pil_image(
    image_pil,
    filename_prefix: str,
    case_idx: int,
    turn_num: int,
    tool_name: str,
    userid: Optional[str] = None,
) -> Optional[str]:
    """Upload a PIL image to COS and return a public URL.

    Returns ``None`` when the uploader is unavailable or the upload fails.
    Errors are swallowed and logged so the pipeline can fall back to inline
    base64 transmission for visual reasoning.
    """

    _ensure_loaded()
    if _upload_cos is None:
        return None

    user = userid or config.COS_UPLOAD_USERID
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_pil.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = (
            f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_{tool_name}.png"
        )
        mode = f"{filename_prefix}_{date_str}_{filename_prefix}"

        cos_key, cos_url = _call_upload(
            tmp_path,
            filename,
            date_str,
            mode,
            user,
            use_direct_url=True,
        )

        if cos_url:
            return cos_url
        if cos_key and config.COS_BUCKET_HOST_TEMPLATE:
            return config.COS_BUCKET_HOST_TEMPLATE.rstrip("/") + cos_key
        return None
    except Exception as exc:
        logger.warning("Upload to COS failed: %s", exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def upload_local_mp4(
    local_path: str,
    filename_prefix: str,
    case_idx: int,
    turn_num: int,
    tag: str,
    userid: Optional[str] = None,
) -> Optional[str]:
    """Upload a local mp4 file to COS and return a public URL.

    Used by ``select_timespan`` to publish the freshly cut sub-clip so the
    runner can attach it as a ``video_url`` part. Returns ``None`` when the
    uploader is unavailable or the upload fails — caller is expected to
    fall back to the parent video URL.
    """

    _ensure_loaded()
    if _upload_cos is None:
        return None
    if not os.path.exists(local_path):
        logger.warning("upload_local_mp4: file not found: %s", local_path)
        return None

    user = userid or config.COS_UPLOAD_USERID
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = (
        f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_{tag}.mp4"
    )
    mode = f"{filename_prefix}_{date_str}_{filename_prefix}"

    try:
        cos_key, cos_url = _call_upload(
            local_path,
            filename,
            date_str,
            mode,
            user,
            use_direct_url=True,
        )
    except Exception as exc:
        logger.warning("upload_local_mp4 raised: %s", exc)
        return None

    if cos_url:
        return cos_url
    if cos_key and config.COS_BUCKET_HOST_TEMPLATE:
        return config.COS_BUCKET_HOST_TEMPLATE.rstrip("/") + cos_key
    return None
