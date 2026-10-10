# AV benchmark and SFT data pipeline v0 Implementation Plan

> For agentic workers: execute inline with superpowers:executing-plans. Checkboxes track deliverables; no new agent dispatch is needed.

**Goal:** Produce a real 16-case development candidate package with media review, executable scoring, and a separate replay-verified SFT export path.

**Architecture:** Benchmark actor tasks and private references are separate records. Training consumes train-only executed episodes, never benchmark references; existing explored assets are reserved for development. Share the frozen corpus and media receipts, not task labels.

**Tech Stack:** Python 3.11, unittest, existing SQLite FTS5 corpus, Pixi, FFmpeg/Pillow for review thumbnails, existing LLaMA-Factory ShareGPT format.

**Spec:** docs/research/AV-DeepResearch-Overall-Research-Proposal-v2-Train-2026-10-10.md

## Global Constraints

- Develop in /home/jovyan/OneSearch-VL on coder ssh 3, branch research/av-deepresearch.
- Every run is create-only and versioned; private material stays in ignored research/runs.
- Existing explored media cannot become a held-out test or independent training corpus.
- No fabricated observations, answers, human reviews or training-ready trajectories.
- Compare Prompted and SFT with the same controller start, fixed observer, corpus and inference budget.
- Train only after real semantic references and retained trajectories exist. GPU remainder remains 1373 job seconds; this CPU batch consumes none.

## Task 1: Closed benchmark contracts and scoring

Files: research/av_bench/contracts.py, research/av_bench/evaluate.py; validation/research/test_research_pipeline.py.
Interfaces: validate_task(task, scope); validate_reference(reference, task, sources); score_prediction(task, reference, prediction, sources, k=5, iou_threshold=.5).
- [ ] Write tests for equivalent complete evidence packages, partial-package rejection, wrong event with matching span, source ranking deduplication, unreviewed reference rejection and timeout denominator.
- [ ] Run `pixi run --locked python -m unittest discover -s validation/research -p test_research_pipeline.py -v`; missing modules must fail.
- [ ] Implement contracts with separate accepted-values reference, per-fact AND evidence and package OR alternatives; scoring returns package recall, moment coverage, necessary fact accuracy and EBS.
- [ ] Run tests and existing research validation. Commit with the complete batch after review.

## Task 2: Executed episode and train-only exporter

Files: research/av_train/trajectory.py, research/av_train/export.py; same validation file.
Interfaces: make_step(action, observation); validate_trace(trace, source_splits); export_trajectory(trace, source_splits, replay, verdict).
- [ ] Test train/dev isolation, tampered observations, fixture rejection, replay mismatch and ShareGPT alternating roles.
- [ ] Implement content-addressed executed step records; exporter calls independent replay and compares deterministic observations, then requires semantic verification and successful evaluation.
- [ ] Export only assistant actions/final as targets, observation as context; set train_on_prompt=false, mask_history=false in accompanying recipe. Token label audit remains required before training.
- [ ] Run tests. An empty retained file is allowed only with explicit rejection counts and reasons.

## Task 3: Development candidate construction

Files: research/av_data/build_research_batch.py; docs/research/AV-Data-Pipeline-v0.md.
Interfaces: build_batch(registry_paths, out_dir); CLI --sources repeated, --out-dir.
- [ ] Test create-only output, 4 tasks per family, absence of private fields in public records, all explored assets assigned dev and reference status pending.
- [ ] Build 16 concrete Chinese questions from registered natural sources, private candidate aids and blank evidence packages; generate one shared development corpus.
- [ ] Run actual search and read_text smoke against built corpus and record receipts; do not score candidate tasks as gold.

## Task 4: Human review visualization

Files: research/av_data/visualize_batch.py.
Interfaces: render_review(batch_dir, out_dir); CLI --batch-dir --out-dir.
- [ ] Test HTML escaping, family counts and separate reviewer decisions export.
- [ ] Render frames at actual original PTS into contact sheets. Link original AV playback with audio and text snapshots; embed only publisher media URLs, preserving source attribution.
- [ ] Show question, required evidence, metric, suitability risk, candidate sources and editable decisions. Browser export creates user review notes, not automatic gold.
- [ ] Copy review deliverables to local outputs, inspect rendering and open the HTML for user review.

## Task 5: Verification and publication

- [ ] Run relevant tests once after all changes; inspect diff and manifest hashes.
- [ ] Record number of candidates, reviewed references, executed metadata receipts, real teacher episodes and retained SFT rows separately.
- [ ] Commit/push only code, validation and docs to the authorized fork. Exclude raw media, private gold, traces and generated training files.

## Execution checkpoint 2026-10-10

Completed: 16 candidate build, shared corpus/search receipts, 10 original-PTS contact sheets, package evaluator, executed episode API and train-only replay/export rejection smoke. 35 new contract/integration tests pass; final research regression: 165/165 passed. New independent-source authoring acquisition produced 4 AV and 4 pages; split remains unfrozen pending family review. Pending: semantic teacher adapter, independent train family freeze, adjudicated reference, actual retained model trajectories and token-label/sequence audits. These pending items prevent training; they do not invalidate the author-review artifact.

Final verification: Pixi research tests 165/165; Python compileall passed. Actual script episode executed two tools; exporter rejected it as dev/fixture and retained zero rows. No GPU or SFT run in this batch.
