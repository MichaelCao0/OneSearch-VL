#!/usr/bin/env python3
"""Bounded pinned Qwen2.5-Omni acquisition and CPU import probes.

Use the isolated runtime for import probes. Acquisition may run on the
existing locked CPU runtime while GPU dependencies install, without loading
model weights or changing that runtime. Public HF access uses token=False.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import urllib.request

REPOSITORY_ID = "Qwen/Qwen2.5-Omni-7B"
REVISION = "ae9e1690543ffd5c0221dc27f79834d0294cba00"
MAX_MODEL_BYTES = 30_000_000_000
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "research/runs/av-native-smoke-20261010-v1/runtime-setup"
CACHE_DIR = Path("/home/jovyan/.cache/huggingface/hub")
MANIFEST_PATH = EVIDENCE / "model-manifest.json"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def metadata() -> dict:
    url = f"https://huggingface.co/api/models/{REPOSITORY_ID}/revision/{REVISION}?blobs=true"
    request = urllib.request.Request(url, headers={"User-Agent": "OneSearch-VL-runtime-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        info = json.load(response)
    if info["sha"] != REVISION:
        raise RuntimeError("HF metadata revision mismatch")
    siblings = info["siblings"]
    if any(not isinstance(item.get("size"), int) or item["size"] < 0 for item in siblings):
        raise RuntimeError("Unknown model artifact size: refusing acquisition")
    total = sum(item["size"] for item in siblings)
    if total > MAX_MODEL_BYTES:
        raise RuntimeError(f"Snapshot exceeds acquisition cap: {total} > {MAX_MODEL_BYTES}")
    for item in siblings:
        lfs = item.get("lfs")
        if lfs and (lfs.get("size") != item["size"] or len(lfs.get("sha256", "")) != 64):
            raise RuntimeError("Invalid LFS size/hash metadata")
    return info

def acquire(verify_only: bool = False) -> dict:
    info = metadata()
    write_json(EVIDENCE / "hf-model-info-pinned.json", info)
    expected_total = sum(item["size"] for item in info["siblings"])
    print(f"Acquisition preflight: {len(info['siblings'])} files, {expected_total} bytes, cap {MAX_MODEL_BYTES}", flush=True)
    # Set before importing hub constants to avoid an implicit Xet cache.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(
        repo_id=REPOSITORY_ID, revision=REVISION, cache_dir=str(CACHE_DIR),
        allow_patterns=[item["rfilename"] for item in info["siblings"]],
        max_workers=2, local_files_only=verify_only, token=False,
    ))
    files = []
    for item in info["siblings"]:
        file = snapshot / item["rfilename"]
        if not file.is_file() or file.stat().st_size != item["size"]:
            raise RuntimeError(f"Missing or wrong size: {item['rfilename']}")
        actual_sha = sha256(file)
        lfs = item.get("lfs")
        # Non-LFS files are Git blobs; authenticate config and processor assets
        # with the pinned Git blob hash too, while recording a SHA-256 for loading.
        if lfs:
            expected_sha = lfs["sha256"]
            verified = actual_sha == expected_sha
            verification_kind = "hf_lfs_sha256"
        else:
            content = file.read_bytes()
            git_blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            verified = git_blob == item["blobId"]
            expected_sha = actual_sha
            verification_kind = "pinned_git_blob_sha1_and_local_sha256"
        if not verified:
            raise RuntimeError(f"Integrity mismatch: {item['rfilename']}")
        row = dict(path=item["rfilename"], local_path=str(file), size_bytes=item["size"],
                   expected_sha256=expected_sha, actual_sha256=actual_sha,
                   verified=True, is_lfs=bool(lfs), verification_kind=verification_kind,
                   hf_blob_id=item["blobId"])
        files.append(row)
        print(f"Verified {item['rfilename']} {item['size']} {actual_sha}", flush=True)
    data = dict(
        schema_version="av-omni-model-manifest-v1", status="verified",
        repository_id=REPOSITORY_ID, revision=REVISION, snapshot_path=str(snapshot),
        total_size_bytes=expected_total, downloaded_model_bytes=expected_total,
        max_model_bytes=MAX_MODEL_BYTES, all_lfs_verified=True,
        lfs_file_count=sum(row["is_lfs"] for row in files),
        verified_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        files=files, model_loaded=False, gpu_inference_verified=False,
        acquisition_environment={"python": sys.executable, "huggingface-hub": importlib.metadata.version("huggingface-hub")},
        lock_path=str(ROOT / "runtime/av-observer/pixi.lock"),
        lock_sha256=sha256(ROOT / "runtime/av-observer/pixi.lock"),
    )
    write_json(MANIFEST_PATH, data)
    print(f"Manifest ready: {MANIFEST_PATH}", flush=True)
    return data

def probe() -> dict:
    import torch
    import torchvision
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    result = dict(
        status="imports_verified", python=sys.version, executable=sys.executable,
        versions={name: importlib.metadata.version(name) for name in
                  ["torch", "torchvision", "transformers", "qwen-omni-utils", "accelerate",
                   "huggingface-hub", "librosa", "av", "decord"]},
        torch_cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        model_class=Qwen2_5OmniForConditionalGeneration.__name__,
        processor_class=Qwen2_5OmniProcessor.__name__,
        process_mm_info_signature=str(inspect.signature(process_mm_info)),
        generate_signature=str(inspect.signature(Qwen2_5OmniForConditionalGeneration.generate)),
        model_loaded=False, gpu_inference_verified=False,
        pixi_lock_sha256=sha256(ROOT / "runtime/av-observer/pixi.lock"),
    )
    write_json(EVIDENCE / "import-probe.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "acquire", "verify", "probe"])
    args = parser.parse_args()
    if args.command == "preflight":
        info = metadata()
        write_json(EVIDENCE / "hf-model-info-pinned.json", info)
        print(json.dumps({"revision": info["sha"], "files": len(info["siblings"]),
                          "total_bytes": sum(item["size"] for item in info["siblings"]),
                          "max_model_bytes": MAX_MODEL_BYTES}, indent=2))
    elif args.command == "probe":
        probe()
    else:
        acquire(verify_only=args.command == "verify")

if __name__ == "__main__":
    main()
