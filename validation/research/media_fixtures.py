"""Bounded real-media fixtures and independent decoded-content measurements."""
import json
import subprocess
from pathlib import Path

import numpy as np


def run(args):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def probe(path, frames=False):
    args = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if frames:
        args += ["-show_frames"]
    return json.loads(run(args + [str(path)]).stdout)


def make_offset_vfr(path):
    """4 s video, audio begins at .4 s, simultaneous pulse at 1.6 s, origin 5 s."""
    run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
        "color=c=black:s=64x64:r=25:d=4", "-f", "lavfi", "-i",
        "aevalsrc=if(between(t\\,1.2\\,1.28)\\,0.8*sin(2*PI*1000*t)\\,0):s=16000:d=3.6",
        "-filter_complex",
        "[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(t,1.6,1.68)',select='not(mod(n,3))+eq(mod(n,3),1)'[v];[1:a]asetpts=PTS+0.4/TB[a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "ffv1", "-c:a", "pcm_s16le",
        "-fps_mode", "passthrough", "-output_ts_offset", "5", "-y", str(path),
    ])


def make_audio_only(path):
    run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "sine=frequency=400:sample_rate=16000:duration=1", "-c:a", "pcm_s16le", "-y", str(path)])


def decoded_markers(path):
    """Read pixels/samples, then associate them with independently probed decode PTS."""
    data = probe(path, frames=True)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next(s for s in data["streams"] if s["codec_type"] == "audio")
    vframes = [f for f in data["frames"] if f["media_type"] == "video"]
    aframes = [f for f in data["frames"] if f["media_type"] == "audio"]
    pixels = np.frombuffer(run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
          "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"]).stdout, dtype=np.uint8)
    pixels = pixels.reshape((-1, video["height"], video["width"]))
    lit = np.flatnonzero(pixels.mean(axis=(1, 2)) > 200)
    if not len(lit):
        raise AssertionError("video marker missing from decoded content")
    visual_time = float(vframes[int(lit[0])]["pts_time"])
    samples = np.frombuffer(run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
          "-map", "0:a:0", "-ac", "1", "-f", "f32le", "pipe:1"]).stdout, dtype="<f4")
    loud = np.flatnonzero(np.abs(samples) > .3)
    if not len(loud):
        raise AssertionError("audio marker missing from decoded content")
    # Samples are consecutive within audio decode frames; never anchor audio to video zero.
    sample_index = int(loud[0])
    consumed = 0
    audio_time = None
    for frame in aframes:
        count = int(frame["nb_samples"])
        if sample_index < consumed + count:
            audio_time = float(frame["pts_time"]) + (sample_index - consumed) / int(audio["sample_rate"])
            break
        consumed += count
    if audio_time is None:
        raise AssertionError("decoded audio sample does not map to an original frame")
    return visual_time, audio_time, data
