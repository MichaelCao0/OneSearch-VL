"""Offline validation for the prepared EVGR recipe; no judge requests or CUDA."""
import argparse
import json
import os
from pathlib import Path
from vision_deepresearch_async_workflow.reward_config import RewardWeights

from vision_deepresearch_async_workflow.reward_data import GroundTruthIndex, frame_indices, ground_frames, task_identity, validate_gt_fields
from vision_deepresearch_async_workflow.reward_client import JudgeConfig
from vision_deepresearch_async_workflow.evidence import _ground_descriptions


def validate_dataset(root, splits=None):
    root = Path(root)
    index = GroundTruthIndex(root)
    if splits is None:
        splits = {}
        for split in ('train', 'test'):
            if split == 'test' and not (root / 'test.jsonl').is_file():
                continue
            with (root / f'{split}.jsonl').open() as f:
                splits[split] = [json.loads(line) for line in f if line.strip()]
    counts = {}
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f'EVGR {split} is empty')
        for task in rows:
            gt = index.lookup(task)
            validate_gt_fields(gt, task['modality'])
            if not gt.get('claim_facts'):
                raise ValueError(f'EVGR rubric is incomplete for task {task["id"]}')
            modality = task['modality']
            if modality in ('multi_image', 'video'):
                _ground_descriptions(gt)
                grid = task.get('video_grid') or {}
                values = task.get('image_frame_indices') if modality == 'multi_image' else grid.get('frame_indices')
                count = len(task.get('images') or ()) if modality == 'multi_image' else len(grid.get('frame_urls') or ())
                if modality == 'video' and values is None:
                    values = list(range(count))
                indices = frame_indices(values, count)
                if modality == 'video':
                    indices = indices[:int(grid.get('n_frames') or count)]
                if not ground_frames(gt) or not set(ground_frames(gt)).issubset(indices):
                    raise ValueError(f'EVGR GT frames are absent from task {task["id"]}')
            elif not gt.get('gt_bboxes') and gt.get('grounding_contract') != 'source_only':
                raise ValueError(f'EVGR image rubric has no grounding contract for task {task["id"]}')
        counts[split] = len(rows)
    return {'gt_counts': index.counts, 'matched': counts}


def validate_registered(root):
    """Reject stale registered tasks before rollout, including old frame metadata."""
    import pyarrow.parquet as pq
    registry_path = Path(os.getenv('RLLM_DATASET_REGISTRY_FILE') or (
        Path(__file__).resolve().parents[1] / 'rllm/registry/dataset_registry.json'))
    if not registry_path.is_file():
        raise ValueError('RL registry is missing; run register_rl_dataset.sh first')
    registry = json.loads(registry_path.read_text())
    name = os.getenv('RL_DATASET_NAME', 'OneSearch-VL-RL-10K')
    if name not in registry:
        raise ValueError(f'{name} is not registered; run register_rl_dataset.sh first')
    splits = {}
    for split in ('train', 'test'):
        if split == 'test' and not (Path(root) / 'test.jsonl').is_file():
            continue
        path = registry[name].get(split)
        if not path:
            raise ValueError(f'{name}/{split} is not registered')
        # Validate the exact extra_info read by Verl, not only the source parquet.
        verl_path = str(Path(path).with_name(Path(path).stem + '_verl.parquet'))
        rows = [row['extra_info'] for row in pq.read_table(verl_path, columns=['extra_info']).to_pylist()]
        with (Path(root) / f'{split}.jsonl').open() as f:
            expected = [json.loads(line) for line in f if line.strip()]
        def identity(row):
            grid = row.get('video_grid') or {}
            return (str(row['id']), task_identity(row), row.get('image_frame_indices'),
                    grid.get('frame_indices', list(range(len(grid.get('frame_urls', []))))) if grid else None)
        if list(map(identity, rows)) != list(map(identity, expected)):
            raise ValueError(f'{name}/{split} is stale or reordered; re-run register_rl_dataset.sh')
        splits[split] = rows
    return validate_dataset(root, splits)


def validate_environment():
    """Fail before starting workers if the release recipe is misconfigured."""
    weights = RewardWeights.from_env()
    weights.validate_enabled()
    if os.getenv('FORMAL_EVGR_MODE', 'both') != 'both':
        raise ValueError('The current reward requires FORMAL_EVGR_MODE=both')
    if os.getenv('EVGR_ENABLED', '0').lower() in ('1', 'true', 'yes'):
        raise ValueError('Legacy EVGR_ENABLED was removed; use FORMAL_EVGR_ENABLED')
    if os.getenv('FORMAL_EVGR_JUDGE_BACKEND', 'seed') != 'seed':
        raise ValueError('The current reward requires the seed judge backend')
    config = JudgeConfig.from_env()
    if config.model != os.getenv('FORMAL_EVGR_EXPECTED_JUDGE_MODEL', 'Doubao-Seed-2.0-pro'):
        raise ValueError('Reward judge model differs from expected model')
    if int(os.getenv('FORMAL_EVGR_MAX_CONCURRENCY', '32')) < 1:
        raise ValueError('Formal EVGR concurrency must be positive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=os.getenv('FORMAL_EVGR_DATA_ROOT'))
    parser.add_argument('--check-env', action='store_true', help='Check judge configuration without making API calls')
    parser.add_argument('--check-registered', action='store_true', help='Check the registered Verl data matches this release')
    args = parser.parse_args()
    if not args.data_root:
        parser.error('--data-root or FORMAL_EVGR_DATA_ROOT is required')
    if args.check_env:
        validate_environment()
    result = validate_dataset(args.data_root)
    if args.check_registered:
        result['registered'] = validate_registered(args.data_root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
