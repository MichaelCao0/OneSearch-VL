#!/usr/bin/env python3
"""CPU-only official raw AV processor probe; no model weights are loaded."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import av_omni_setup as setup

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import hf_hub_download
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

info = setup.metadata()
selected = [item for item in info["siblings"] if not item["rfilename"].startswith("model-")]
for item in selected:
    path = Path(hf_hub_download(setup.REPOSITORY_ID, item["rfilename"],
                revision=setup.REVISION, cache_dir=str(setup.CACHE_DIR), token=False))
    if path.stat().st_size != item["size"]:
        raise RuntimeError("Processor asset size mismatch")
    content_sha = setup.sha256(path)
    if item.get("lfs"):
        verified = content_sha == item["lfs"]["sha256"]
    else:
        content = path.read_bytes()
        git_hash = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        verified = git_hash == item["blobId"]
    if not verified:
        raise RuntimeError("Processor asset integrity mismatch")
snapshot = setup.CACHE_DIR / "models--Qwen--Qwen2.5-Omni-7B/snapshots" / setup.REVISION
out = setup.EVIDENCE
clip = out / "synthetic-processor-probe.mp4"
if not clip.exists():
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=160x160:r=4:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(clip)], check=True)
messages = [{"role": "user", "content": [
    {"type": "video", "video": str(clip), "fps": 2, "min_frames": 4,
     "max_frames": 4, "min_pixels": 12544, "max_pixels": 12544},
    {"type": "text", "text": "Describe the visible scene and the sound."},
]}]
processor = Qwen2_5OmniProcessor.from_pretrained(str(snapshot), local_files_only=True)
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
if not audios or not videos or not np.any(audios[0]):
    raise RuntimeError("Raw joint AV decoding missing a modality")
inputs = processor(text=text, audio=audios, images=images, videos=videos,
                   return_tensors="pt", padding=True, use_audio_in_video=True)
required = {"input_features", "feature_attention_mask", "pixel_values_videos", "video_grid_thw"}
if not required.issubset(inputs):
    raise RuntimeError(f"Official processor missing modality tensors: {required - set(inputs)}")
result = {
    "status": "raw_av_processor_verified_cpu", "synthetic_probe": True,
    "model_loaded": False, "generation_run": False, "gpu_inference_verified": False,
    "clip_path": str(clip), "clip_sha256": setup.sha256(clip),
    "model_revision": setup.REVISION, "use_audio_in_video": True,
    "audio_waveforms": [{"shape": list(x.shape), "sample_rate": 16000,
                         "rms": float(np.sqrt(np.mean(np.square(x)))),
                         "sha256": hashlib.sha256(x.tobytes()).hexdigest()} for x in audios],
    "video_shapes": [list(x.shape) for x in videos],
    "processor_tensors": {name: {"shape": list(t.shape), "dtype": str(t.dtype)}
                          for name, t in inputs.items() if hasattr(t, "shape")},
    "installed_sources": {
        cls.__name__: {"path": inspect.getfile(cls),
                      "sha256": setup.sha256(Path(inspect.getfile(cls)))}
        for cls in [Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor]},
}
setup.write_json(out / "processor-probe.json", result)
print(json.dumps(result, indent=2))
