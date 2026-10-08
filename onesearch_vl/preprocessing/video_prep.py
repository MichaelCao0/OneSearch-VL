"""Optional Seed client video resampling helper."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import tempfile
from typing import Optional

DEFAULT_TARGET_FPS = 2.0
DEFAULT_MAX_FRAMES = 128
DEFAULT_LONG_EDGE = 512


@dataclasses.dataclass
class PreppedVideo:
    """Result of ``prep_video_for_seed``."""

    output_path: str
    target_frames: int
    effective_input_fps: float
    repacked_duration: float
    long_edge: int
    expected_prompt_tokens: int  # rough estimate, see _estimate_tokens


def probe_duration(input_path: str, *, ffprobe_bin: str = "ffprobe") -> float:
    """Return the video's duration in seconds via ffprobe."""

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=nw=1:nk=1",
        input_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def _estimate_tokens(target_frames: int, long_edge: int) -> int:
    """Estimate the token budget for a repacked video."""

    tokens_per_frame = max(50, round(long_edge * long_edge / 2870))
    return target_frames * tokens_per_frame + 50  # +50 for text prompt overhead


def prep_video_for_seed(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    target_fps: float = DEFAULT_TARGET_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    long_edge: int = DEFAULT_LONG_EDGE,
    crf: int = 23,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    overwrite: bool = True,
) -> PreppedVideo:
    """Sample + downscale + repack a video for Seed-2.0-pro consumption.

    Parameters mirror the Qwen3-VL CLI flags so existing pipeline
    configurations port over directly.
    """

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    duration = probe_duration(input_path, ffprobe_bin=ffprobe_bin)
    if duration <= 0:
        raise ValueError(f"non-positive duration probed for {input_path}: {duration}")

    # Qwen3-VL-style sampling: cap total frames at max_frames.
    ideal_frames = max(1, int(round(target_fps * duration)))
    target_frames = min(ideal_frames, max_frames)
    effective_input_fps = target_frames / duration

    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        fd, output_path = tempfile.mkstemp(prefix=f"seed_{base}_", suffix=".mp4")
        os.close(fd)

    # The two filters: (1) extract `target_frames` frames evenly across
    # the source, (2) clamp the long edge. We then -r 1 the output so
    # the repacked clip is target_frames seconds long at 1 fps, matching
    # Seed's default sampling rate.
    vf = (
        f"fps={target_frames}/{duration:.6f},"
        f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
    )
    cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel", "error",
        "-i", input_path,
        "-vf", vf,
        "-r", "1",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)

    return PreppedVideo(
        output_path=output_path,
        target_frames=target_frames,
        effective_input_fps=effective_input_fps,
        repacked_duration=float(target_frames),
        long_edge=long_edge,
        expected_prompt_tokens=_estimate_tokens(target_frames, long_edge),
    )


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="source video path")
    ap.add_argument("--output", default=None, help="output mp4 path (default: temp file)")
    ap.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--long-edge", type=int, default=DEFAULT_LONG_EDGE)
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    prepped = prep_video_for_seed(
        args.input,
        args.output,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        long_edge=args.long_edge,
        crf=args.crf,
    )
    print(json.dumps(dataclasses.asdict(prepped), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
