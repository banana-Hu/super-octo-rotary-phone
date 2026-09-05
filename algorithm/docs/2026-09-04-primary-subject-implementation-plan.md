# Primary Subject Cutout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve individual person cutouts and add grouped primary-subject cutouts with optional connected-foreground retention.

**Architecture:** Person masks remain the anchors and stable fallback. A focused grouping module builds geometric person groups and optionally expands them with foreground pixels reachable from those anchors; exporting and contracts expose the new artifacts without changing existing field semantics.

**Tech Stack:** Python 3.11–3.14, NumPy, Pillow, Torchvision Mask R-CNN, optional `transparent-background`/InSPyReNet, pytest, Ruff

**Spec:** `algorithm/docs/2026-09-04-primary-subject-design.md`

## Global Constraints

- Modify only `algorithm/`; do not implement frontend, HTTP backend, video processing, or model training.
- Keep `people/person_*.png` and existing JSON fields backward compatible.
- Do not commit model weights or test photographs.
- Foreground inference failure must fall back to people-only subjects.

---

### Task 1: Subject grouping and mask composition

**Files:**
- Create: `algorithm/src/momentmaker_cv/subject_processing.py`
- Test: `algorithm/tests/test_subject_processing.py`

**Interfaces:**
- Consumes: `list[ProcessedMask]`, image size, optional foreground probability mask.
- Produces: `build_subject_masks(...) -> list[ProcessedSubjectMask]`.

- [ ] Write tests for near/far grouping, transitive grouping, connected foreground retention, remote foreground rejection, and invalid foreground shape.
- [ ] Run `python -m pytest tests/test_subject_processing.py -q` and confirm the tests fail because the module is absent.
- [ ] Implement box grouping, bounded low-resolution flood fill, alpha union, crop calculation, and deterministic ordering.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Optional foreground adapter

**Files:**
- Create: `algorithm/src/momentmaker_cv/foreground.py`
- Test: `algorithm/tests/test_foreground.py`
- Modify: `algorithm/pyproject.toml`
- Modify: `algorithm/constraints-demo.txt`

**Interfaces:**
- Produces: `ForegroundSegmenter.predict(image) -> np.ndarray` and lazy `InSPyReNetForegroundSegmenter`.

- [ ] Write tests using a fake remover for grayscale normalization, shape validation, lazy loading, and unavailable dependency errors.
- [ ] Run the focused tests and confirm failure.
- [ ] Implement the protocol and lazy adapter without importing optional packages at module import time.
- [ ] Add the exact optional demo dependency and run the focused tests.

### Task 3: Contracts, exports, and pipeline fallback

**Files:**
- Modify: `algorithm/src/momentmaker_cv/contracts.py`
- Modify: `algorithm/src/momentmaker_cv/exporter.py`
- Modify: `algorithm/src/momentmaker_cv/pipeline.py`
- Modify: `algorithm/src/momentmaker_cv/__init__.py`
- Test: `algorithm/tests/test_contracts.py`
- Test: `algorithm/tests/test_exporter.py`
- Test: `algorithm/tests/test_pipeline.py`

**Interfaces:**
- Produces: `SubjectCutout`, `ProcessingResult.subjects`, subject PNG export, and `process_image(..., foreground_segmenter=None)`.

- [ ] Add failing contract, artifact cleanup, pipeline success, and foreground-fallback tests.
- [ ] Implement compatible result serialization and atomic `subjects/subject_*.png` export.
- [ ] Connect grouping after person post-processing; run foreground inference only for `subject_mode="foreground"`.
- [ ] Run the three focused test modules and confirm they pass.

### Task 4: CLI, documentation, and verification

**Files:**
- Modify: `algorithm/src/momentmaker_cv/cli.py`
- Modify: `algorithm/tests/test_cli.py`
- Modify: `algorithm/README.md`
- Modify: `algorithm/INTEGRATION.md`
- Modify: `algorithm/THIRD_PARTY_NOTICES.md`

**Interfaces:**
- Produces: `--subject-mode none|people|foreground` and documented JSON/artifact contract.

- [ ] Add a failing parser test for all subject modes and the `people` default.
- [ ] Implement the CLI option and instantiate the optional foreground adapter only in enhanced mode.
- [ ] Document setup, output semantics, performance, fallback behavior, licensing, and limitations.
- [ ] Run Ruff check/format, all pytest tests, compileall, pip check, and three licensed local-image smoke tests.
- [ ] Review the diff for scope, secrets, generated artifacts, and contract compatibility; commit and push only the feature branch to the fork.
