#!/usr/bin/env python3
"""Project an existing RL bundle onto fields consumed by the training workflow.

Keeps one row per training task in first-occurrence order, with unchanged media
and reward targets. Conflicting rows with the same ID fail. No model calls.
The output uses a relative media symlink; copy its target for standalone export.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'rllm'))
from vision_deepresearch_async_workflow.evidence import normalize_evidence
from vision_deepresearch_async_workflow.reward_data import task_identity, GT_COMMON, GROUND_ANNOTATION_FIELDS, validate_gt_fields, ground_frames
from vision_deepresearch_async_workflow.reward_preflight import validate_dataset

VERSION = 'OneSearch-VL-RL-10K'
TASK_COMMON = ('id', 'question', 'answer', 'modality')
GRID_FIELDS = ('video_url', 'frame_urls')
ANNOTATION_FIELDS = GROUND_ANNOTATION_FIELDS


def select(row, fields):
    return {key: row[key] for key in fields if key in row}


def project_task(row):
    result = select(row, TASK_COMMON)
    if row['modality'] == 'video':
        grid = row['video_grid']
        count = len(grid['frame_urls'])
        if (grid.get('n_frames', count) != count
                or grid.get('frame_indices', list(range(count))) != list(range(count))
                or grid.get('grid_fps', 1.) != 1.):
            raise ValueError('Minimal video data requires an ordered, zero-based 1 fps grid')
        result['video_grid'] = select(row['video_grid'], GRID_FIELDS)
    elif row['modality'] in ('single_image', 'multi_image'):
        result['images'] = row['images']
        if row['modality'] == 'multi_image':
            result['image_frame_indices'] = row['image_frame_indices']
    else:
        raise ValueError('Unsupported task modality')
    return result


def project_gt(row, modality=None):
    result = select(row, GT_COMMON)
    modality = modality or row.get('modality')
    if modality == 'single_image':
        if row.get('gt_bboxes'):
            result['gt_bboxes'] = row['gt_bboxes']
        elif row.get('grounding_contract') == 'source_only':
            result['grounding_contract'] = 'source_only'
        else:
            raise ValueError('Single-image target requires boxes or source_only')
    else:
        descriptions = {a['frame_index']: a.get('object_description') for a in row.get('grounding_annotations', [])}
        result['grounding_annotations'] = [dict(frame_index=f, **(
                                               {'object_description': descriptions[f]} if descriptions.get(f) is not None else {}))
                                           for f in ground_frames(row)]
    validate_gt_fields(result, modality)
    return result


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def unique_training_rows(records):
    """Remove exact repeated tasks, never choose between conflicting payloads."""
    unique = {}
    for row in records:
        key = str(row['id'])
        if key in unique and unique[key] != row:
            raise ValueError(f'Conflicting training rows for task ID {key}')
        unique.setdefault(key, row)
    return list(unique.values())


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError('Output already exists; choose a new version directory')
    training_rows = unique_training_rows(read_rows(source / 'train.jsonl'))
    training_ids = {(row['modality'], str(row['id'])) for row in training_rows}
    # Validate every pair before writing any projected data.
    paired = {}
    for modality in ('single_image', 'multi_image', 'video'):
        tasks = read_rows(source / 'data' / f'{modality}.jsonl')
        gts = read_rows(source / 'evgr_gt' / f'{modality}.jsonl')
        if len(tasks) != len(gts):
            raise ValueError('Task/GT length mismatch')
        selected_tasks, selected_gts = [], []
        for task, gt in zip(tasks, gts, strict=True):
            if (modality, str(task['id'])) not in training_ids:
                continue
            if (str(task['id']), task['answer'], task['modality']) != (str(gt['id']), gt.get('answer', task['answer']), gt.get('modality', modality)):
                raise ValueError('Task/GT pairing mismatch')
            projected, target = project_task(task), project_gt(gt, modality)
            if task_identity(projected) != task_identity(task):
                raise ValueError('Projection changed runtime task identity')
            if normalize_evidence(task, [], gt) != normalize_evidence(projected, [], target):
                raise ValueError('Projection changed evidence/targets')
            selected_tasks.append(task)
            selected_gts.append(gt)
        paired[modality] = (selected_tasks, selected_gts)
    output.mkdir(parents=True)
    media = (source / 'media').resolve(strict=True)
    (output / 'media').symlink_to(os.path.relpath(media, output), target_is_directory=True)
    counts = {}
    for name in ('train',):
        records = read_rows(source / f'{name}.jsonl')
        source_rows = len(records)
        if name == 'train':
            records = unique_training_rows(records)
        write_rows(output / f'{name}.jsonl', map(project_task, records))
        unique = {(r['modality'], str(r['id'])): r for r in records}
        counts[name] = dict(rows=len(records), unique=len(unique), source_rows=source_rows,
                            removed_repeated_rows=source_rows-len(records),
                            unique_by_modality=dict(Counter(r['modality'] for r in unique.values())))
    for modality, (tasks, gts) in paired.items():
        write_rows(output / 'data' / f'{modality}.jsonl', map(project_task, tasks))
        write_rows(output / 'evgr_gt' / f'{modality}.jsonl', (project_gt(gt, modality) for gt in gts))
    checks = validate_dataset(output)
    summary = dict(version=VERSION, source_annotation_version=source.name, counts=counts,
                   training_layout='one_row_per_task_first_occurrence',
                   paired_tasks=sum(len(tasks) for tasks, _ in paired.values()),
                   task_identity_preserved=True, fact_and_appearance_targets_preserved=True,
                   multi_image_video_boxes_removed=True, annotation_metadata_removed=True,
                   preflight=checks, media_storage='relative symlink; dereference for standalone export')
    (output / 'release_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Existing paired annotation bundle with media')
    parser.add_argument('--output', type=Path, required=True, help='New minimal training bundle')
    args = parser.parse_args()
    summary = prepare(args.source, args.output)
    print(json.dumps(summary, indent=2))
