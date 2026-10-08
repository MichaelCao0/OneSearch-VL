"""Environment-driven configuration and the model registry.

Every value is read from an environment variable so the codebase does
not embed any internal hostname, credential or storage path. Callers
that want non-default behaviour set the variable before running, or
override the value via the CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Generic runtime knobs
# ---------------------------------------------------------------------------

# Maximum number of agent turns per case before forcing termination.
MAX_TURNS: int = int(os.environ.get("AGENT_MAX_TURNS", "12"))

# Local directory that holds raw benchmark images keyed by ``case_id``.
# When set, the pipeline can recover from broken URLs by loading the
# original picture from disk. Leave empty to disable the fallback.
FVQA_IMAGE_DIR: str = os.environ.get("FVQA_IMAGE_DIR", "")


# ---------------------------------------------------------------------------
# Video deep research inputs
# ---------------------------------------------------------------------------

# Cap on the longer edge of a bootstrap-sampled frame (pixels). Keeps
# inline base64 payloads under upstream gateway request-size limits.
# Set <=0 to disable.
BOOTSTRAP_FRAME_MAX_SIDE: int = int(os.environ.get("BOOTSTRAP_FRAME_MAX_SIDE", "768"))

# Seed gateway rejects images whose width or height is below this threshold
# (HTTP 400: "Minimum allowed dimension: 14 pixels"). Crop outputs are
# upscaled proportionally when needed before upload / the next model turn.
SEED_MIN_IMAGE_SIDE: int = int(os.environ.get("SEED_MIN_IMAGE_SIDE", "14"))

# Qwen3-VL video sampling knobs — must match SFT joint yaml
# (video_max_pixels=100352, video_maxlen=128, video_fps=2). These are
# passed through to qwen_vl_utils.process_vision_info via message-level
# fields; processor.video_max_pixels alone does NOT affect infer.
VIDEO_FPS: float = float(os.environ.get("VIDEO_FPS", "2.0"))
VIDEO_MAX_PIXELS: int = int(os.environ.get("VIDEO_MAX_PIXELS", "100352"))
VIDEO_MAX_FRAMES: int = int(os.environ.get("VIDEO_MAX_FRAMES", "128"))
# Match the LLaMA-Factory ProcessorArguments default used by SFT (16*16).
VIDEO_MIN_PIXELS: int = int(os.environ.get("VIDEO_MIN_PIXELS", "256"))

# ---------------------------------------------------------------------------
# Public Anthropic Messages API. Configure a model ID and your own key.
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL: str = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "")
ANTHROPIC_VERSION: str = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")


# ---------------------------------------------------------------------------
# COS uploader bootstrap. We keep this isolated so the package can run
# with a user-supplied asset uploader.
# ---------------------------------------------------------------------------

COS_UPLOAD_PATHS: str = os.environ.get("COS_UPLOAD_PATHS", "")
COS_UPLOAD_USERID: str = os.environ.get("COS_UPLOAD_USERID", "onesearch-vl")
COS_BUCKET_HOST_TEMPLATE: str = os.environ.get(
    "COS_BUCKET_HOST_TEMPLATE", ""
)
"""Optional template used to reconstruct a public URL when the uploader only
returns an object key. Use ``{bucket}`` / ``{region}`` placeholders, e.g.
``http://{bucket}.cos.{region}.myqcloud.com``. Leave empty when the
uploader already returns a fully-qualified URL.
"""


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata describing one supported model variant."""

    name: str
    family: str  # "qwen3_vl_dense" | "qwen3_vl_moe" | "claude" | "seed" | "vllm" | "vllm_inproc"
    display_name: str
    default_checkpoint_env: Optional[str] = None
    default_checkpoint: str = ""
    supports_multi_gpu: bool = True
    needs_moe_patch: bool = False
    # ``True`` when the runner can take ``video_url`` / ``video`` parts
    # directly. Qwen3-VL native video reader handles mp4 base64 / file
    # paths; the Claude gateway used here cannot.
    supports_video: bool = False
    extra: Dict[str, str] = field(default_factory=dict)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "8b": ModelSpec(
        name="8b",
        family="qwen3_vl_dense",
        display_name="Qwen3-VL-8B-Instruct",
        default_checkpoint_env="QWEN3VL_8B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-8B-Instruct",
        supports_video=True,
    ),
    "32b": ModelSpec(
        name="32b",
        family="qwen3_vl_dense",
        display_name="Qwen3-VL-32B-Instruct",
        default_checkpoint_env="QWEN3VL_32B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-32B-Instruct",
        supports_video=True,
    ),
    "30b-a3b": ModelSpec(
        name="30b-a3b",
        family="qwen3_vl_moe",
        display_name="Qwen3-VL-30B-A3B-Instruct",
        default_checkpoint_env="QWEN3VL_30B_A3B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-30B-A3B-Instruct",
        needs_moe_patch=True,
        supports_video=True,
    ),
    "claude": ModelSpec(
        name="claude",
        family="claude",
        display_name="Claude",
        default_checkpoint="",
        supports_multi_gpu=False,
    ),
    "claude-api": ModelSpec(
        name="claude-api",
        family="claude_anthropic",
        display_name="Claude (Anthropic API)",
        # checkpoint here is the model id sent to /v1/messages; env overrides.
        default_checkpoint_env="ANTHROPIC_MODEL",
        default_checkpoint="",
        supports_multi_gpu=False,
        supports_video=False,  # no native video; agent uses select_frame
    ),
    # Send video frames as image_url parts for checkpoints trained on frame inputs.
    "8b-frames": ModelSpec(
        name="8b-frames",
        family="qwen3_vl_dense",
        display_name="Qwen3-VL-8B-Instruct (frames mode)",
        default_checkpoint_env="QWEN3VL_8B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-8B-Instruct",
        supports_video=False,
    ),
    "seed-2.0": ModelSpec(
        name="seed-2.0",
        family="seed",
        display_name="Doubao-Seed-2.0-pro",
        default_checkpoint_env="SEED_MODEL",
        # default_checkpoint here is the upstream model marker, NOT a
        # filesystem path. SeedRunner reads it via env override too.
        default_checkpoint="Doubao-Seed-2.0-pro",
        supports_multi_gpu=False,
        supports_video=True,
    ),
    # Local vLLM OpenAI-compatible server hosting a Qwen3-VL checkpoint.
    # Same HTTP contract as Seed (POST /chat/completions) but pointed at
    # ``VLLM_API_BASE`` (default http://localhost:8001/v1). Drives all three
    # modalities via the shared agent loop. ``supports_video=True`` sends the
    # manifest's presigned https mp4 as a single ``video_url`` — vLLM fetches
    # and samples it natively (fps/max_pixels are set on the server, not the
    # request), matching the native-video SFT strategy. No additional upload needed
    # since the video URL is already remote https.
    "vllm": ModelSpec(
        name="vllm",
        family="vllm",
        display_name="Qwen3-VL (local vLLM)",
        # This is the served-model-name the vLLM server advertises, NOT a
        # filesystem path. VLLMRunner reads it via VLLM_MODEL env override.
        default_checkpoint_env="VLLM_MODEL",
        default_checkpoint="Qwen/Qwen3-VL-8B-Instruct",
        supports_multi_gpu=False,
        supports_video=True,
    ),
    # Use frame image_url parts with the local vLLM backend.
    # Requires a checkpoint trained on frame inputs.
    "vllm-frames": ModelSpec(
        name="vllm-frames",
        family="vllm",
        display_name="Qwen3-VL (local vLLM, frames mode)",
        default_checkpoint_env="VLLM_MODEL",
        default_checkpoint="Qwen/Qwen3-VL-8B-Instruct",
        supports_multi_gpu=False,
        supports_video=False,
    ),
    # In-process vLLM: instantiate ``vllm.LLM`` directly inside this process
    # instead of talking to a separate OpenAI server over HTTP. Same agent
    # loop, same OpenAI message format (reuses to_oai_messages) — only the
    # transport differs. Best for single-process offline batch inference /
    # alignment / debugging where standing up a server is overkill, and it
    # avoids the URL/data-URI image serialization hop entirely. ``checkpoint``
    # here is a real filesystem path / HF id (loaded by the engine), unlike
    # the HTTP "vllm" spec whose default_checkpoint is a served-model-name.
    "vllm-local": ModelSpec(
        name="vllm-local",
        family="vllm_inproc",
        display_name="Qwen3-VL (in-process vLLM)",
        default_checkpoint_env="QWEN3VL_8B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-8B-Instruct",
        supports_multi_gpu=True,
        supports_video=True,
    ),
    # Frames-as-images variant of the in-process backend (supports_video=False):
    # the video pipeline sends frame PNGs as image_url parts instead of one
    # video_url. Use with frames-strategy SFT checkpoints.
    "vllm-local-frames": ModelSpec(
        name="vllm-local-frames",
        family="vllm_inproc",
        display_name="Qwen3-VL (in-process vLLM, frames mode)",
        default_checkpoint_env="QWEN3VL_8B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-8B-Instruct",
        supports_multi_gpu=True,
        supports_video=False,
    ),
}


def resolve_checkpoint(spec: ModelSpec, override: Optional[str] = None) -> str:
    """Resolve a checkpoint path / HF id with CLI > env > default priority."""

    if override:
        return override
    if spec.default_checkpoint_env:
        env_value = os.environ.get(spec.default_checkpoint_env, "").strip()
        if env_value:
            return env_value
    return spec.default_checkpoint


def list_model_names() -> str:
    return ", ".join(sorted(MODEL_REGISTRY))
