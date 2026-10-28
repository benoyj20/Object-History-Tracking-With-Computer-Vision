# Physical Version Control for Object History Tracking

A phone camera takes snapshots of a shared kitchen counter over time, handheld or loosely placed
and from any angle. This project turns those snapshots into an object-level history: what is on
the counter, where each item is, when it was added, moved, or removed, and how long it has
stayed in one place. A natural-language assistant answers questions such as "how long has the
plate been out?" or "where is the knife?" from that history.

CS5330 Computer Vision final project, Northeastern University, by Justin Gubbens and Benoy Joseph.

## Pipeline

1. **Detect**: a fine-tuned Ultralytics YOLO26 model finds kitchen objects in each snapshot.
2. **Track**: each detection is matched to an existing object by appearance and position, so the
   same object keeps the same identity across frames.
3. **Record history**: added, moved, and removed events are logged with the time interval in which
   each change happened. Snapshots are sparse, so placement times are intervals, not exact times.
4. **Ask**: a Next.js web page where a Claude-powered assistant, behind a FastAPI service,
   answers questions using read-only queries over the SQLite history.
5. **Evaluate**: scripted scenarios with known changes measure event precision and recall.

## Current status

| Part | Lead | Status |
| --- | --- | --- |
| Detector | Benoy | **Working** |
| Tracking and history | Justin | Not started |
| Assistant and web page | Joint | Not started (starter web app only) |
| Evaluation | Joint | Detector scoring done; scenario evaluation not started |

**Detector.** A YOLO26s model fine-tuned on Kaggle close-ups and 449 labeled counter photos from
three kitchens finds 453 of 470 objects and 189 of 203 pieces of cutlery (93%) on 68 held-out
photos. Classes: cup, glass, plate, fork, knife, spoon. The split is random by photo, so this
score is optimistic; an earlier model fell to 50% of cutlery in a kitchen it had never seen, and
a new, unseen session is still needed for an honest score. Every run is in
[detector results](docs/detector-results.md). `objhist_detection.detect` writes one JSON file
per session for the tracker (format proposed in
[ADR 0005](docs/adr/0005-detector-output-file-for-the-tracker.md)).

**The camera is not fixed.** It is a handheld phone that can view the counter from any angle and
moves between snapshots, so an object's pixel position changes even when nothing on the counter
moved. Tracking has to rely on appearance and on positions relative to the counter and to other
objects.

**Data.** The 450 counter photos (19 sessions with capture times), the labels, the train/val/test
split, and the best model are not in GitHub, because the photos are private and the files are
large. Ask Benoy for the private data package; [data/README.md](data/README.md) describes the
layout.

## Who does what next

### Justin: tracking and history (`packages/tracking`)

1. **Read the detector output.** Run the detector on the sessions, or use its JSON files: each
   photo's boxes (class, confidence, position) and capture time. Accept the format in
   [ADR 0005](docs/adr/0005-detector-output-file-for-the-tracker.md) or propose changes.
2. **Keep object identity across photos.** Match each detection to an object from the previous
   photo using appearance (DINOv2 embeddings, compared with CLIP) and position relative to the
   counter and other objects. Two identical cups must stay two objects.
3. **Turn matches into events:** added, removed, and moved, each with a time interval. A missed
   detection or an object hidden behind another must not be logged as removed.
4. **Store the history in SQLite** with a documented schema: objects, events, positions, and
   time intervals.
5. **Tests:** synthetic detection sequences must produce the expected event log, including
   duplicates, occlusion, two objects swapping places, and empty photos.

6. Score the best model on a new, unseen session.
7. Shoot scripted scenario sessions and log the true changes (what was added, moved, or removed,
   and when) as ground truth for evaluation.

### Together

- **Evaluation:** agree the open values in the
  [evaluation protocol](docs/evaluation-protocol.md) before testing, including the "moved"
  threshold (it cannot be in pixels with a moving camera). Score event precision and recall on
  the scripted scenarios against an IoU-only baseline.
- **Assistant and web page:** Claude answers questions from the SQLite history through a FastAPI
  service and the Next.js page.
- **Final demo and report.**

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/), Node.js 22+, pnpm 10, and Git. GNU Make is optional; every target
in the `Makefile` is a single `uv` command you can run directly. For GPU training, an NVIDIA
driver 580 or newer (check with `nvidia-smi`).

```sh
# NVIDIA GPU (CUDA 13.0 PyTorch build)
uv sync --all-packages --extra cu130      # or: make bootstrap
uv run python -m objhist_detection.env_check --require-cuda   # or: make env-check

# macOS (Apple Silicon) and machines without NVIDIA GPUs
uv sync --all-packages --extra cpu        # or: make bootstrap TORCH=cpu
```

On a Mac, the `cpu` build also includes Apple GPU support: pass `--device mps` to detector
commands (or `--device cpu`). Training is only practical on an NVIDIA GPU.

Install the web app's dependencies with `pnpm install`, then start it with `make dev-web`
(<http://localhost:3000>).

uv installs Python 3.12 automatically if needed. Always pass one of the two extras; a plain
`uv sync` falls back to the default PyPI PyTorch build, which is CPU-only on Windows.

Run the checks used by CI (or everything at once with `make check`):

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

Run the best detector on every session (after unpacking the data package). Leave out
`--device` on an NVIDIA GPU; on a Mac use `--device mps`, or `--device cpu` if that fails:

```sh
uv run python -m objhist_detection.detect --no-tiled --device mps \
    --weights runs/detect/yolo26s-counter-random-relabeled-flipud-2026-10-28-1280-e15/weights/best.pt
```

Once a dataset is in place (see [data/README.md](data/README.md)), fine-tune YOLO26:

```sh
make train DATA=data/processed/kitchen/data.yaml EPOCHS=100
```

## Repository layout

| Path | Contents | Lead |
| --- | --- | --- |
| `packages/core` | Shared data contracts (`BoundingBox`, `Detection`, `FrameDetections`) | Joint |
| `packages/detection` | Dataset preparation, YOLO26 training and inference | Benoy |
| `packages/tracking` | Identity association and the object history log | Justin |
| `packages/assistant` | Natural-language questions over the history | Joint |
| `packages/evaluation` | Scripted scenarios, event precision and recall | Joint |
| `apps/web` | Next.js web page for asking questions and viewing history | Joint |
| `data/` | Local datasets, captures, and outputs (not committed) | |
| `docs/` | Plan, checklist, data card, detector results, evaluation protocol, decisions | |

## Documentation

- [Problem statement and plan](docs/problem-statement-and-plan.md)
- [Project checklist](docs/project-checklist.md)
- [Data card](docs/data-card.md)
- [Detector results](docs/detector-results.md)
- [Evaluation protocol](docs/evaluation-protocol.md)
- [Architecture decisions](docs/adr/README.md)

## Contributing

See [the contributor guide](CONTRIBUTING.md) for setup, branch naming, verification, and PR
review. Coding agents must also follow [AGENTS.md](AGENTS.md); Claude Code skills and agents for
this repository are described in [.claude/skills/README.md](.claude/skills/README.md).
