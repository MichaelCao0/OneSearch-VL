#!/usr/bin/env python3
"""Register train JSONL or parquet, with an optional explicit validation split.

Preserves unique task IDs and order. Rejects repeated IDs before registration.
"""
import argparse
from pathlib import Path
import sys


def validate_rows(rows, split):
    seen = set()
    for i, row in enumerate(rows):
        for key in ('id', 'question', 'answer', 'modality'):
            if row.get(key) is None:
                raise ValueError(f'{split} row {i}: missing {key}')
        key = str(row['id'])
        if key in seen:
            raise ValueError(f'{split} row {i}: duplicate task ID {key}; prepare one row per task first')
        seen.add(key)
        images = row.get('images')
        count = len(images) if images is not None else 0
        actual = 'video' if row.get('video_grid') else 'multi_image' if count > 1 else 'single_image' if count == 1 else None
        if actual != row['modality']:
            raise ValueError(f'{split} row {i}: modality does not match initial media')
    return rows


def load_rows(path, media_root=None, source_map=None):
    import json
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif path.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        raise ValueError('Expected .jsonl or .parquet input')
    sources = None
    if source_map:
        with Path(source_map).open() as f:
            sources = {r['path']: r['source'] for r in map(json.loads, f)}
    # Resolve the shared root once, not once per frame in the media inventory.
    root = Path(media_root).resolve() if media_root else path.resolve().parent
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def resolve(value):
        if sources is not None:
            return sources[value]
        if value.startswith(('https://', 'http://')) or Path(value).is_absolute():
            return value
        relative = Path(value)
        if '..' in relative.parts:
            raise ValueError('Parent traversal is not allowed in media references')
        resolved = root / relative
        if not resolved.is_file():
            raise FileNotFoundError(f'{relative}: provide --media-root or --source-map')
        return str(resolved)
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'rllm'))
    from vision_deepresearch_async_workflow.reward_data import media_identity
    for row in rows:
        # Preserve the content-addressed identity before local path resolution.
        # GT annotations remain in a judge-only sidecar, not the actor prompt.
        row['evgr_media_identity'] = list(media_identity(row))
        if row.get('images') is not None:
            row['images'] = [resolve(v) for v in row['images']]
        if row.get('video_grid'):
            grid = row['video_grid']
            grid['frame_urls'] = [resolve(v) for v in grid['frame_urls']]
            grid['video_url'] = resolve(grid['video_url'])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', default='OneSearch-VL-RL-10K')
    p.add_argument('--train', type=Path, required=True)
    p.add_argument('--test', type=Path, help='Optional validation split; the release uses train only')
    p.add_argument('--evgr-root', type=Path, help='EVGR paired data/ and evgr_gt/ directory; defaults to train directory')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--media-root', type=Path, help='Directory containing relative media references')
    group.add_argument('--source-map', type=Path, help='Local JSONL path-to-source map')
    args = p.parse_args()
    prepared = {split: validate_rows(load_rows(path, args.media_root, args.source_map), split)
                for split, path in [('train', args.train), ('test', args.test)] if path is not None}
    if not all(prepared.values()):
        raise ValueError('Every supplied split must be nonempty.')
    overlap = {str(r['id']) for r in prepared['train']} & {str(r['id']) for r in prepared.get('test', [])}
    if overlap:
        raise ValueError(f'Train/test IDs overlap ({len(overlap)}); use globally unique task IDs.')
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'rllm'))
    from vision_deepresearch_async_workflow.reward_preflight import validate_dataset
    validate_dataset(args.evgr_root or args.train.parent, prepared)
    from rllm.data.dataset import DatasetRegistry
    for split, rows in prepared.items():
        DatasetRegistry.register_dataset(args.name, rows, split)
        print(f'{split}: {len(rows)} rows, {len({str(r["id"]) for r in rows})} unique task IDs')


if __name__ == '__main__':
    main()
