# Project and testing checklist

Status snapshot: November 17, 2026. The [problem statement and plan](problem-statement-and-plan.md)
describes the targets; this checklist records what actually exists and what has evidence.

A checked item means the work or verification was completed. An unchecked item still needs
evidence, even when related code exists. Record dates and artifact or test references when
checking items off; keep manual checks distinct from automated tests.

## Current working flow

Photos to a stored object history, not yet accurate. A YOLO26s detector fine-tuned on Kaggle
close-ups and 449 labeled counter photos from three kitchens finds 453 of 470 objects and 189
of 203 pieces of cutlery (93%) on 68 held-out photos of a random split, which is optimistic; on
an unseen kitchen an earlier model fell to 50% cutlery ([detector results](detector-results.md)).
The camera is handheld and can be at any angle. `objhist_detection.detect` writes detector
output per session as `SessionDetections` files; `objhist_tracking.import_detections` loads them
into the SQLite history ([ADR 0006](adr/0006-history-store-schema-and-detection-import.md));
`objhist_tracking.embed_detections` caches DINOv2 embeddings; and
`objhist_tracking.track_session` stores objects, observations, and added/removed events with
placement intervals. On the static first-kitchen sessions the best tracker still reports about
four times too many objects, mostly because DINOv2 does not recognize cutlery reliably (see
Milestone 3). The assistant, the web page, and formal evaluation are not implemented.

## Completed: repository setup (2026-10-08, branch `chore/project-setup`)

- [x] uv workspace with `core`, `detection`, `tracking`, `assistant`, and `evaluation` packages.
- [x] Selectable PyTorch builds: `--extra cu130` (NVIDIA) or `--extra cpu` (CI, CPU-only);
  see [ADR 0001](adr/0001-workspace-layout-and-pytorch-builds.md).
- [x] Shared contracts `BoundingBox`, `Detection`, `FrameDetections` with validation of box
  geometry, finite values, confidence range, image bounds, and timezone-aware capture times.
- [x] GPU environment check `python -m objhist_detection.env_check --require-cuda`.
- [x] Ruff lint/format and pytest configuration, with a `gpu` marker for CUDA-only tests.
- [x] GitHub Actions `python` job, PR template, CODEOWNERS, and contributor rules
  (`AGENTS.md`, `CONTRIBUTING.md`).
- [x] Repo-local Claude Code skills and agents copied from AgentWatch and adapted.
- [x] Starter Next.js web app in `apps/web` with web rules, `OBJHIST_API_URL` validation and tests,
  and a `javascript` CI job (2026-10-09, [ADR 0004](adr/0004-nextjs-front-end.md)).

### Setup verification (2026-10-08, Windows 11, RTX 5070 Laptop GPU 8 GB, driver 617.42)

- [x] `uv sync --all-packages --extra cu130`: torch 2.14.1+cu130, ultralytics 8.4.174,
  Python 3.12.12; `env_check --require-cuda` reported CUDA available.
- [x] Pretrained `yolo26n.pt` predicted on the Ultralytics sample image on `cuda:0`
  (about 15 ms inference after warm-up). Manual check, not an automated test.
- [x] `uv run pytest`: 17 passed (16 unit tests and 1 GPU test) with the CUDA build.
- [x] CI install path reproduced locally in a separate environment:
  `uv sync --all-packages --extra cpu --frozen`, Ruff checks passed, 16 passed and 1 GPU test
  skipped.
- [x] Observe a passing remote CI run on GitHub (every PR since #1; latest #27, 2026-10-28).
- [ ] Setup reproduced on Justin's machine.
  Partly (2026-11-17, macOS on Apple silicon, CPU build): Ruff and `uv run pytest` ran (286
  passed, 1 skipped, 1 failed: `test_detect.py::test_unreadable_photo_stops_the_run`, which
  passes alone and fails only after other detector tests, with or without the tracking
  package), and `objhist_detection.detect` ran on all 19 sessions on the CPU. No CUDA here.

## Milestone 1: dataset (provided by Benoy)

- [ ] Choose the class list with the scenarios in mind (including items such as milk and dish
  soap) and record it in the [data card](data-card.md).
- [x] Provide the snapshot sequences in the format in [data/README.md](../data/README.md),
  with a timezone-aware capture time for every image (2026-10-24): 450 photos in 19 sessions
  from three kitchens; see [Provided sequences](data-card.md#provided-sequences).
- [ ] Download the Roboflow dataset; record version, license, and hash.
- [x] Decide whether the Kaggle data is usable under its rules and worth re-annotating
  (2026-10-09: Benoy confirmed training is permitted; recorded in the data card).
- [x] Kaggle auto-labeling pipeline (2026-10-09, branch `feature/kaggle-auto-labels`):
  frozen class-stratified split, YOLOE box proposals with review flags, HTML review page, and
  YOLO-format export that requires a verdict on every validation and test image. Unit tests in
  `packages/detection/tests/test_kaggle_*.py`; proposals run on all 5,559 labeled images.
- [x] Review of the Kaggle proposals (all validation/test, flagged training, 300-image training
  spot-check), done by Claude visual review rather than a person (2026-10-09); error rates and
  audit in the data card. A human re-check of the test split is still recommended.
- [x] Labeling tool for the provided images (2026-10-12, branch `feature/counter-auto-labels`):
  YOLOE proposes every box at 1280 px, a local review page keeps, relabels, rejects, or draws
  boxes, and an export writes session-split YOLO data. Checked on the two test photos; tests in
  `packages/detection/tests/test_counter_*.py`.
- [x] Label boxes on the provided images (2026-10-12): 100 photos in 6 sessions, all reviewed
  by Benoy and checked against the known object list; dataset `counter-2026-10-12` in the
  [data card](data-card.md#dataset-counter-2026-10-12).
- [x] Freeze splits grouped by capture session (2026-10-12): train 75, val 9, test 16 photos
  (`session_splits.csv`).
- [x] Label the second and third kitchens (2026-10-21 and 2026-10-27): all 450 photos reviewed
  by Benoy, the first kitchen twice (re-review 2026-10-28).
- [x] Random split by photo, chosen by Benoy over all three kitchens (2026-10-27, seed
  20261027): train 314, val 67, test 68. An exception to the session rule that makes scores
  optimistic; see the [data card](data-card.md#random-splits-by-photo-2026-10-21-and-2026-10-27).

**Exit test:** a versioned dataset with a documented class list, split lists, and hashes that
trains with `make train`.

## Milestone 2: detector (lead: Benoy)

- [ ] Zero-shot COCO-pretrained YOLO26 baseline on the held-out split.
- [ ] Fine-tune YOLO26; record the command, seed, hardware, and metrics for each run.
  First run on `kaggle-kitchenware-v1` (2026-10-09): test mAP50 0.984, mAP50-95 0.961 on
  single-object close-ups; details in [detector results](detector-results.md).
  Second run `yolo26n-counter-kitchen-1280-v2` (2026-10-12): Kaggle + counter data at 1280 px.
  On the 16 counter test photos it scores mAP50 0.885 and mAP50-95 0.782 (old model: 0.334 and
  0.295), finding 97 of 112 objects (old model: 22).
- [x] Compare model sizes (n/s) for accuracy versus speed on the 8 GB GPU (2026-10-13):
  YOLO26s at 1280 px (batch 4, 1.94 h) found no more cutlery than the nano on 29 validation
  photos (69 vs 68 of 87) and added glass-as-cup errors; the nano was kept. See [detector results](detector-results.md).
- [x] Final detector `yolo26n-counter-final-1280-e15` (2026-10-13): trained on 129 counter
  photos including both night sessions; on held-out session 1643 it scores 98 of 112 objects
  (cutlery 34 of 48), mAP50-95 0.796, against 95, 31, and 0.782 for the previous model.
- [x] Tiled inference for wide counter photos and a `preview` command (2026-10-12, branch
  `feature/tiled-inference`): on the two test photos the Kaggle model went from 2 of 10 objects
  (640 px) to 9 of 10 correct, with 1 wrong class and 2 false boxes. Settings still need
  checking on held-out sessions. Tests in `packages/detection/tests/test_tiled_inference.py`.
- [x] Tiling works at lower resolutions (2026-10-12, branch
  `feature/resolution-independent-tiling`): tiles scale with the photo, giving 8–10 of 10 objects
  on the two test photos downscaled to 640–2560 px (fixed tiles: 5–9).
- [x] Convert predictions to `FrameDetections` with tests for coordinate conversion
  (2026-10-12, branch `feature/detector-tracker-bridge`): `python -m objhist_detection.detect`
  writes one `SessionDetections` file per session with EXIF or manifest capture times
  ([ADR 0005](adr/0005-detector-output-file-for-the-tracker.md), proposed). Ran on the test
  session: 2 frames, 12 detections, the same as `preview`. Tests in
  `packages/detection/tests/test_detect.py` and `test_capture_times.py`.
- [x] Justin agrees to the detector output file format (ADR 0005) (2026-11-17); the tracker
  reads it unchanged, and the ADR is marked accepted.
- [x] Rerunnable detector scoring (2026-10-12, branch `feature/detector-scoring`):
  `python -m objhist_evaluation.score_detections` counts correct, wrong-class, missed, and false
  boxes per class from `detect` output; it reproduced the recorded test comparison exactly.
  Tests in `packages/evaluation/tests/`.
- [x] YOLO26s on the full data (2026-10-13): on held-out session 1643 it finds 40 of 48
  cutlery pieces (nano 34) and 104 of 112 objects, mAP50-95 0.830. Chosen on that same session,
  so it needs confirming on a new test session.
- [x] Three kitchens, random split (2026-10-27/28): the best model,
  `yolo26s-counter-random-relabeled-flipud-2026-10-28-1280-e15`, finds 189 of 203 pieces of
  cutlery (93%), mAP50-95 0.934, on 68 test photos; see
  [detector results](detector-results.md#random-split-runs-2026-10-21-to-2026-10-28).
- [ ] Score the best model on a new, unseen session (ideally a new place or night, with piles
  and partly hidden cutlery) for an honest result.
- [ ] Choose and record the confidence threshold on the validation split.

**Exit test:** mAP50 and mAP50-95 on the frozen test split, compared with the zero-shot baseline.

## Milestone 3: tracking and history (lead: Justin)

Work so far is on branch `feature/tracking` (2026-11-17, not yet merged), in
`packages/tracking`.

- [ ] Association by position and DINOv2/CLIP appearance embeddings across snapshots.
  Position and DINOv2 are implemented (`tracker.py`), with optional camera-movement
  compensation from the boxes (`camera_alignment.py`) and two-stage matching that follows moved
  objects by appearance. CLIP is not implemented, and accuracy on real photos is not yet
  acceptable (results below).
- [ ] Compare DINOv2 and CLIP embeddings on the validation scenarios; record the choice.
  No tracking validation split exists yet.
- [ ] Added, removed, and moved events with placement intervals between snapshots.
  Added and removed are written with intervals between two photos (an object present in the
  first photo gets an open start). Moved is not written until its threshold is agreed in the
  [evaluation protocol](evaluation-protocol.md); observations keep every position.
- [x] Rule that keeps missed detections and occlusion distinct from removal (2026-11-17):
  removal needs 2 processed photos in a row without the object (`REMOVAL_MISSING_FRAMES`,
  provisional), dated between the last photo it was seen in and the first it was missing from;
  missing only at a session's end gives no event. Tests in `test_tracker.py`.
- [x] SQLite history store with a documented schema and ordered migrations (2026-11-17):
  migrations `0001`–`0003` with recorded hashes, documented in
  [ADR 0006](adr/0006-history-store-schema-and-detection-import.md) (proposed; needs Benoy's
  agreement). Tests in `test_history_store.py`, `test_import_detections.py`,
  `test_tracker_tables.py`, and `test_embed_detections.py`.
- [x] Unit tests for duplicates, occlusion, swaps, and empty frames (2026-11-17): 96 tracking
  tests on synthetic detections and stub embeddings, including identical cups, cups that trade
  places, camera moves, a moved object, one missed photo, and empty photos.

**Real-data runs (2026-11-17, informal):** the current best detector
(`yolo26s-counter-random-relabeled-flipud-2026-10-28-1280-e15`, single pass at 1280 px) on all
19 sessions gave 3,067 detections in 450 photos; these photos are in its training data, so the
detections are better than on new photos. DINOv2 (`facebook/dinov2-small`, revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`) embedded every detection locally. The first
kitchen's 8 sessions (151 photos) hold the same seven objects in every photo, with one or two
moved between shots (data card, change logs, and Benoy, 2026-11-17), so the correct history is
56 objects with no added or removed events after the first photo:

| Tracker version | Objects (truth 56) | False added | False removed |
| --- | --- | --- | --- |
| `appearance-aligned-v2` (DINOv2, alignment, two-stage) | 222 | 167 | 156 |
| `appearance-gated-v1` (DINOv2) | 244 | 189 | 177 |
| `position-centre-v1` (baseline) | 313 | 258 | 243 |
| `appearance-aligned-v1` | 349 | 294 | 277 |
| `position-aligned-v1` | 362 | 307 | 290 |

All thresholds are provisional and were chosen while looking at these sessions, so these are
not evaluation results. Cutlery causes most errors: the one knife in each first-kitchen session
scores below the 0.5 similarity threshold against itself in 55% of consecutive photos (fork
44%, spoon 29%; cups 4%). Next: split these sessions into tuning and held-out sets in the
evaluation protocol before tuning.

**Exit test:** synthetic `FrameDetections` sequences produce the expected event log, including
the edge cases above.

## Milestone 4: generative AI assistant (joint)

- [ ] Allowlisted, read-only history query tools with strict input schemas over SQLite.
- [ ] Claude tool-use loop with `claude-opus-5-5`; refusal and API errors shown clearly.
- [ ] Answers cite history records and say "unknown" when evidence is missing.
- [ ] FastAPI history service exposing the assistant and read-only history endpoints.
- [ ] Next.js question page with loading, error, unreachable-service, and missing-key states.
- [ ] Latest snapshot with labeled boxes and a per-object timeline, with text alternatives.
- [ ] Tests with stubbed model responses, including rejected tool calls.

**Exit test:** a fixed question set over a recorded history answered correctly, with no
invented times or locations.

## Milestone 5: evaluation (joint)

- [ ] Freeze the [evaluation protocol](evaluation-protocol.md) thresholds before evaluating.
- [ ] Provide scripted scenario sequences for every listed condition and annotate their
  ground-truth changes.
- [ ] Implement event matching and metrics with unit tests.
- [ ] Run the IoU-only baseline, DINOv2, and CLIP association on the same scenarios.

**Exit test:** a results table of event precision/recall per condition, identity switches,
placement-interval coverage, and assistant accuracy, reproducible from recorded commands.

## Milestone 6: final report and demo

- [ ] End-to-end demo from new snapshots to an assistant answer.
- [ ] Final report with methods, results, failures, and limitations.

## Decisions to settle

- [x] LLM provider: Claude API, default model `claude-opus-5-5` (2026-10-08,
  [ADR 0002](adr/0002-assistant-claude-sqlite-web.md)).
- [x] History store: SQLite, opened read-only by the assistant (2026-10-08, ADR 0002).
- [x] Interface: a web page served by a Python backend (2026-10-08, ADR 0002).
- [x] Appearance features: deep embeddings, DINOv2 by default with CLIP compared on validation
  scenarios (2026-10-09, [ADR 0003](adr/0003-appearance-embeddings-and-provided-dataset.md)).
- [x] Capture hardware: out of scope; Benoy provides the image dataset (2026-10-09, ADR 0003).
  The snapshot interval is whatever the provided sequences use, recorded in the data card.
- [x] Web page stack: Next.js front end and FastAPI history service (2026-10-09,
  [ADR 0004](adr/0004-nextjs-front-end.md)).
- [x] `main` branch protection enabled (2026-10-08): PR with 1 approval, stale approvals
  dismissed, `python` and `javascript` checks required on an up-to-date branch, conversations resolved, no force
  pushes or deletion. Admin bypass stays available until Justin is a collaborator.
- [x] Coding agents may view Benoy's counter photos (downscaled, metadata removed, no people)
  for label review and error analysis (2026-10-12, Benoy's decision for his own photos; see
  AGENTS.md). Justin to be told; his images still need both contributors' agreement.
- [ ] Add Justin as a repository collaborator and to `.github/CODEOWNERS`.
