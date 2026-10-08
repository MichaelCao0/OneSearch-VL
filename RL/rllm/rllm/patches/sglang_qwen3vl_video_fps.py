"""Supply video metadata and preserve sampled frames in SGLang's Qwen3-VL processor."""
import os

_patched = False


def _frame_count(v):
    """Best-effort frame count of a single video item (tensor / ndarray / list)."""
    try:
        # torch.Tensor / np.ndarray (T, C, H, W) or (T, H, W, C)
        if hasattr(v, "shape") and len(v.shape) >= 1:
            return int(v.shape[0])
        if isinstance(v, (list, tuple)):
            return len(v)
    except Exception:
        pass
    return None


def apply() -> bool:
    """Idempotently wrap BaseMultimodalProcessor.process_mm_data. Returns True if applied."""
    global _patched
    if _patched:
        return True
    try:
        from sglang.srt.multimodal.processors import base_processor as _bp
    except Exception:
        return False

    Base = getattr(_bp, "BaseMultimodalProcessor", None)
    if Base is None or not hasattr(Base, "process_mm_data"):
        return False

    _orig = Base.process_mm_data
    if getattr(_orig, "_qwen3vl_video_fps_patched", False):
        _patched = True
        return True

    # fps used for TIMESTAMP computation, NOT frame sampling. The grid.mp4 is
    # composed at 1 fps (ffmpeg -framerate 1 in preprocess_grids), so each frame
    # is one real second and the correct <t seconds> tags come from fps=1.0.
    # This is DECOUPLED from the sampling fps (VIDEO_FPS=2.0): do_sample_frames=
    # False already keeps all frames, so this fps only sets the timestamp scale.
    # Using the sampling fps (2.0) here would halve every timestamp (tags 0.2,
    # 1.2, ... instead of the SFT/inference 0.5, 2.5, ...), corrupting the
    # model's notion of video time. Env-overridable for non-1fps grids.
    _vfps = float(os.environ.get("VIDEO_GRID_FPS", "1.0"))

    def process_mm_data(self, input_text, images=None, videos=None, audios=None, **kwargs):
        # Only touch the qwen video path, and only when the caller hasn't already
        # supplied metadata. Guard on the processor being a Qwen3-VL processor so
        # we never perturb other models.
        if videos and "video_metadata" not in kwargs:
            try:
                proc_cls = type(getattr(self, "_processor", None)).__name__
            except Exception:
                proc_cls = ""
            if proc_cls in ("Qwen3VLProcessor", "Qwen2_5_VLProcessor", "Qwen3VLMoeProcessor"):
                vids = videos if isinstance(videos, (list, tuple)) else [videos]
                meta = []
                ok = True
                for v in vids:
                    n = _frame_count(v)
                    if n is None:
                        ok = False
                        break
                    # frames are ALREADY sampled by sglang's preprocess_video
                    # (Layer 1, fps=2). Provide full metadata so the HF video
                    # processor (Layer 2) does not re-sample them: fps + a
                    # frames_indices covering all T frames (needed for the
                    # timestamp / video-token path in processing_qwen3_vl.py).
                    meta.append({
                        "total_num_frames": int(n),
                        "fps": _vfps,
                        "duration": float(n) / _vfps,
                        "frames_indices": list(range(int(n))),
                    })
                if ok and meta:
                    kwargs["video_metadata"] = meta
                    kwargs["do_sample_frames"] = False
                    import sys as _sys
                    print(
                        f"[fps-patch] injected video_metadata (n_frames={[m['total_num_frames'] for m in meta]}, "
                        f"fps={_vfps}) + do_sample_frames=False",
                        file=_sys.stderr, flush=True,
                    )
        return _orig(self, input_text, images=images, videos=videos, audios=audios, **kwargs)

    process_mm_data._qwen3vl_video_fps_patched = True
    Base.process_mm_data = process_mm_data
    _patched = True
    import sys as _sys
    print(
        f"[fps-patch] applied to {Base.__module__}.{Base.__name__}.process_mm_data "
        f"(id={id(Base)})",
        file=_sys.stderr, flush=True,
    )
    return True
