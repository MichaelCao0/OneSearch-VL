"""Align SGLang Qwen3-VL image/video preprocessing with the SFT media settings.

The patch configures the media decode/resize layer for RL rollout servers; model
weights, tool implementations, and the HF processor remain untouched.
"""

from __future__ import annotations

import os
from typing import Any


IMAGE_PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2
IMAGE_FACTOR = IMAGE_PATCH_SIZE * SPATIAL_MERGE_SIZE
IMAGE_MIN_PIXELS = 1024
IMAGE_MAX_PIXELS = 262144
VIDEO_MIN_PIXELS = 256
VIDEO_MAX_PIXELS = 100352
VIDEO_MAX_FRAMES = 128
VIDEO_FPS = 2.0
_PATCH_MARKER = "_rllm_qwen3vl_sft_preprocess"
_patched = False


def resize_image_like_sft(image: Any):
    """Use the exact qwen_vl_utils image path and SFT pixel bounds."""

    from qwen_vl_utils import vision_process as hf_vision

    return hf_vision.fetch_image(
        {
            "image": image,
            "min_pixels": IMAGE_MIN_PIXELS,
            "max_pixels": IMAGE_MAX_PIXELS,
        },
        image_patch_size=IMAGE_PATCH_SIZE,
    )


def preprocess_video_like_sft(video_reader: Any):
    """Decode/sample/resize a SGLang VideoReader exactly like HF/SFT."""

    import numpy as np
    import torch
    import torchvision
    from qwen_vl_utils import vision_process as hf_vision
    from torchvision.transforms import InterpolationMode

    total_frames = len(video_reader)
    video_fps = float(video_reader.get_avg_fps())
    element = {
        "fps": VIDEO_FPS,
        "min_frames": 4,
        "max_frames": VIDEO_MAX_FRAMES,
        "min_pixels": VIDEO_MIN_PIXELS,
        "max_pixels": VIDEO_MAX_PIXELS,
    }
    nframes = hf_vision.smart_nframes(
        element,
        total_frames=total_frames,
        video_fps=video_fps,
    )
    indices = torch.linspace(0, total_frames - 1, nframes).round().long()
    video_np = video_reader.get_batch(indices.tolist()).asnumpy()
    video = torch.from_numpy(np.asarray(video_np)).permute(0, 3, 1, 2)
    _, _, height, width = video.shape
    resized_height, resized_width = hf_vision.smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=VIDEO_MIN_PIXELS,
        max_pixels=VIDEO_MAX_PIXELS,
    )
    video = torchvision.transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    metadata = {
        "fps": video_fps,
        "duration": total_frames / video_fps,
        "total_num_frames": total_frames,
        "frames_indices": indices.tolist(),
        "video_backend": "decord",
    }
    return video, metadata


def apply() -> bool:
    """Install the patch in the current SGLang server process."""

    global _patched
    if _patched:
        return True
    try:
        from sglang.srt.multimodal.processors import qwen_vl as sgl_qwen
    except Exception as exc:
        raise RuntimeError(
            "cannot import SGLang Qwen-VL processor; the SFT preprocessing "
            "contract was not installed"
        ) from exc

    current_video = getattr(sgl_qwen, "preprocess_video", None)
    if getattr(current_video, _PATCH_MARKER, False):
        _patched = True
        return True

    async def aligned_resize_image(image, *args, **kwargs):
        del args, kwargs
        return resize_image_like_sft(image)

    async def aligned_preprocess_video(video_reader, image_factor=IMAGE_FACTOR):
        del image_factor
        return preprocess_video_like_sft(video_reader)

    setattr(aligned_resize_image, _PATCH_MARKER, True)
    setattr(aligned_preprocess_video, _PATCH_MARKER, True)
    sgl_qwen.resize_image_async = aligned_resize_image
    sgl_qwen.preprocess_video = aligned_preprocess_video

    original_init = sgl_qwen.QwenVLImageProcessor.__init__
    if not getattr(original_init, _PATCH_MARKER, False):

        def aligned_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            processor = getattr(self, "_processor", None)
            image_processor = getattr(processor, "image_processor", None)
            if processor is not None:
                setattr(processor, "image_min_pixels", IMAGE_MIN_PIXELS)
                setattr(processor, "image_max_pixels", IMAGE_MAX_PIXELS)
                setattr(processor, "video_min_pixels", VIDEO_MIN_PIXELS)
                setattr(processor, "video_max_pixels", VIDEO_MAX_PIXELS)
                setattr(processor, "video_fps", VIDEO_FPS)
            if image_processor is not None:
                setattr(image_processor, "min_pixels", IMAGE_MIN_PIXELS)
                setattr(image_processor, "max_pixels", IMAGE_MAX_PIXELS)

        setattr(aligned_init, _PATCH_MARKER, True)
        sgl_qwen.QwenVLImageProcessor.__init__ = aligned_init

    # The preprocessing-cache key includes these module constants, so expose
    # the actual contract before that cache wrapper is installed.
    for name, value in {
        "IMAGE_FACTOR": IMAGE_FACTOR,
        "MIN_PIXELS": IMAGE_MIN_PIXELS,
        "MAX_PIXELS": IMAGE_MAX_PIXELS,
        "VIDEO_MIN_PIXELS": VIDEO_MIN_PIXELS,
        "VIDEO_MAX_PIXELS": VIDEO_MAX_PIXELS,
        "FPS": VIDEO_FPS,
        "FPS_MAX_FRAMES": VIDEO_MAX_FRAMES,
    }.items():
        setattr(sgl_qwen, name, value)

    _patched = True
    print(
        "[sft-preprocess-patch] applied "
        f"image_factor={IMAGE_FACTOR} image_pixels={IMAGE_MIN_PIXELS}:{IMAGE_MAX_PIXELS} "
        f"video_fps={VIDEO_FPS} video_frames={VIDEO_MAX_FRAMES} "
        f"video_pixels={VIDEO_MIN_PIXELS}:{VIDEO_MAX_PIXELS}",
        flush=True,
    )
    return True


def is_applied() -> bool:
    return _patched


if os.environ.get("RLLM_APPLY_SFT_PREPROCESS_PATCH_ON_IMPORT", "0") == "1":
    apply()
