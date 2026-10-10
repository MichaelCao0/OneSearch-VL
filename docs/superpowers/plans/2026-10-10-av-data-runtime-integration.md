# AV data runtime integration Implementation Plan

**Goal:** Verify real trainer labels and connect native observer execution to research episodes, while keeping unreviewed candidates outside retained training data.
**Architecture:** An isolated CPU data-preprocessing Pixi environment imports the repository's actual LLaMA-Factory converter/template/processor. A separate native-observer adapter prepares original-clock AV and returns minimized stable semantic receipts. Existing observer inference and clock checks are reused.
**Spec:** docs/research/AV-DeepResearch-Overall-Research-Proposal-v2-Train-2026-10-10.md and docs/research/AV-Data-Pipeline-v0.md.

- [x] Create pinned runtime/sft-data/pixi.toml and lock; verify native trainer import without disabling version checks.
- [x] Add native adapter tests before implementation: stable exported receipt, unavailable/contract fixture rejection, interval lineage and backend costs.
- [x] Implement research/av_train/native_observer.py; runner supplies real SemanticObserver and saves private input/output plus minimized observation.
- [x] Add label-audit tests before implementation: observation masked; all action/final targets present; train_on_prompt=true/mask_history=true/truncation rejected.
- [x] Implement research/av_train/label_audit.py using actual converter, qwen3_vl_nothink template and SupervisedDatasetProcessor, actual pinned local tokenizer. Do not emulate labels or replace native trainer modules.
- [x] Run CPU trainer preprocessing on explicit fixture and report fixture status, never count it as real teacher data.
- [x] Prepare semantically reviewed annotation import contract; preserve empty candidate gold until actual reviewer content exists.
- [x] Run relevant regression and prepare scoped publication of code, docs, runtime lock and validation. Preserve private labels/tokenization/observer records under ignored run directory.

No GPU submissions in this CPU stage. Existing GPU budget remains unchanged. Candidate suitability decisions and semantic gold are separate; real model trajectories are generated only after sources/tasks are ready.

Verification: research regression 188/188; native trainer preprocessing 4/4; native adapter/runner 11/11. CPU real-media preparation episode executed 2 tools with an explicit contract backend, never teacher inference. Native API injection is connected; default teacher CLI still has no semantic backend configuration. No retained SFT rows or backward occurred. Scoped commit/push remains the final publication step.
