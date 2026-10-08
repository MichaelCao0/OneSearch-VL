#!/usr/bin/env python3
"""Prepare silent sampled video grids using the retained preprocessing code."""
import argparse
import json
from pathlib import Path
import tempfile

from preprocessing.preprocess_grids import materialize_frame_grid, compose_mp4_from_grid
from upload import upload_cos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos-dir', type=Path, required=True, help='MP4 filenames must match integer CSV index values.')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=128)
    p.add_argument('--long-edge', type=int, default=1024)
    p.add_argument('--fps', type=float, default=2.0)
    args = p.parse_args()
    if args.max_frames < 1 or args.long_edge < 16 or args.fps <= 0:
        p.error('Frame count, long edge and FPS must be positive.')
    if args.manifest.exists():
        p.error('Use a new manifest path to avoid mixing preprocessing protocols.')
    videos = sorted(args.videos_dir.glob('*.mp4'))
    if not videos:
        p.error('No MP4 videos found.')
    if any(not f.stem.isdigit() for f in videos):
        p.error('MP4 basenames must be integer case IDs matching the CSV index.')
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open('x') as out:
        for src in videos:
            with tempfile.TemporaryDirectory(prefix='onesearch-grid-') as tmp:
                grid = materialize_frame_grid(str(src), tmp, max_frames=args.max_frames,
                                              long_edge=args.long_edge, target_fps=args.fps)
                if not grid:
                    raise RuntimeError(f'Frame extraction failed: {src.name}')
                clip = Path(tmp) / 'grid.mp4'
                if not compose_mp4_from_grid(grid, str(clip)):
                    raise RuntimeError(f'Grid encoding failed: {src.name}')
                frames = [upload_cos(Path(tmp) / f'frame_{i:03d}.png') for i in range(grid['n_frames'])]
                key, url = upload_cos(clip)
                record = {k: v for k, v in grid.items() if k != 'workdir'}
                record.update(ok=True, case_id=str(int(src.stem)), video_url=url, video_object_key=key,
                              object_keys=[x[0] for x in frames], object_urls=[x[1] for x in frames],
                              target_fps=args.fps, grid_playback_fps=1,
                              representation='sampled_frames_repacked_at_1fps')
                out.write(json.dumps(record) + '\n')
                out.flush()
                print(f'{src.name}: {grid["n_frames"]} frames')


if __name__ == '__main__':
    main()
