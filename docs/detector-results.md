# Detector results

Every detector training run and scored comparison, oldest first. The datasets, labeling, and
splits they use are described in the [data card](data-card.md). Unless stated otherwise,
"correct" counts use `objhist_evaluation.score_detections` (rule `detector-counts-v1`: right
class at IoU ≥ 0.5, confidence ≥ 0.25) and mAP comes from `yolo detect val`. Hardware for all
runs: RTX 5070 Laptop GPU 8 GB, torch 2.14.1+cu130, Ultralytics 8.4.174.

Runs from before 2026-10-27 are in `runs/detect/old/`; the three latest are in `runs/detect/`.
All of these runs, with per-epoch curves, parameters, plots, and the held-out scores in
`data/interim/scores/`, are also in the local MLflow store (`make mlflow-ui`; see the README).
Runs before 2026-10-29 were imported from their run folders after training.
The Models tab holds one model per run (`best.pt`; `final.pt` for
`yolo26n-counter-final-1280-e15` and `last.pt` for `yolo26s-counter-final-1280-e15`, the
checkpoints scored below), with the current best tagged `objhist.current_best`.
The 2026-10-27/28 test counts (overall and per kitchen) were recomputed on 2026-10-29 with
`score_detections --json-out` from single-pass 1280 px detections in
`data/interim/detections-2026-10-29/`; they match the tables below. Per-kitchen scores use label
copies of the test split in `data/interim/score-subsets/` (first kitchen day and night, second
kitchen, third kitchen); the score files are `data/interim/scores/scores-test-random-2026-10-27-relabeled*.json`.

## Current best model

`yolo26s-counter-random-relabeled-flipud-2026-10-28-1280-e15` (`best.pt` SHA-256
`57f3572d…`): on the 68 random-split test photos it finds 453 of 470 objects and 189 of 203
pieces of cutlery (93%), mAP50-95 0.934. The random split makes this optimistic; a new, unseen
session is still needed for an honest score. Use it single pass at 1280 px with duplicates
merged (`--no-tiled`).

**How the best model is chosen:** Benoy chose the flip run (confirmed 2026-10-29) over the
relabeled run, whose validation mAP50-95 is slightly higher (0.953 against 0.951). From now on,
`make train` (`log-models --auto-best` in MLflow) makes a new run the best model only if its
validation mAP50-95 beats the current best's on the same data config; ties keep the current
best, and the test split is not used. A run on a new dataset has a different validation set and
is compared by hand.

## Run index

| Date | Run | Trained on | Headline result |
| --- | --- | --- | --- |
| 2026-10-09 | `yolo26n-kitchen` | Kaggle close-ups | Test mAP50-95 0.961 on close-ups, but 22 of 112 objects on counter photos |
| 2026-10-12 | `yolo26n-counter-kitchen-1280-v2` | + 75 counter photos, 1280 px | 97 of 112 objects on session 1643 |
| 2026-10-13 | `yolo26s-counter-kitchen-1280-b4` | same, YOLO26s | No cutlery gain on validation; nano kept |
| 2026-10-13 | `yolo26n-counter-kitchen-1600-e15` | same, 1600 px | Stopped at epoch 8, below 1280 px |
| 2026-10-13 | `yolo26n-counter-final-1280-e15` | 129 counter photos | 98 of 112, cutlery 34 of 48 on 1643 |
| 2026-10-13 | `yolo26s-counter-final-1280-e15` | 129 counter photos | 104 of 112, cutlery 40 of 48 on 1643 |
| 2026-10-21 | `yolo26s-counter-kitchen-2026-10-21-1280-e15` | kitchens 1–2, split by session | Cutlery 85 of 123 (69%) |
| 2026-10-21 | `yolo26s-counter-random-2026-10-21-1280-e15` | kitchens 1–2, random split | Cutlery 120 of 135 (89%); 50% on the unseen third kitchen |
| 2026-10-27 | `yolo26s-counter-random-2026-10-27-1280-e15` | kitchens 1–3, random split | Cutlery 188 of 203 |
| 2026-10-28 | `yolo26s-counter-random-relabeled-2026-10-28-1280-e15` | + re-reviewed labels | Cutlery 187 of 203 |
| 2026-10-28 | `yolo26s-counter-random-relabeled-flipud-2026-10-28-1280-e15` | + vertical flip | **Cutlery 189 of 203 (93%)**, current best |

## Training run `yolo26n-kitchen` (2026-10-09)

- **Command:** `make train DATA=data/processed/kitchen/data.yaml` (`yolo26n.pt`, 100 epochs, imgsz 640, batch 16, seed 0, `deterministic=True`, other Ultralytics 8.4.174 defaults).
- **Hardware:** RTX 5070 Laptop GPU 8 GB, torch 2.14.1+cu130. Training took 0.93 h.
- **Best checkpoint:** epoch 84; `best.pt` SHA-256 `4f404eb4be11a25e155d8b8849704d253a6c20d366cc5c559ff1ff13ab6f6392`.
  Output was written to `runs/detect/runs/detect/yolo26n-kitchen/` (the Makefile has since been fixed to write `runs/detect/<name>/`) and is now in `runs/detect/old/yolo26n-kitchen/`.
- **Validation (721 images):** mAP50 0.985, mAP50-95 0.962.
- **Test (706 images), evaluated once after training:**
  `yolo detect val model=.../best.pt data=data/processed/kitchen/data.yaml split=test`

| Class | Images | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| all | 706 | 0.964 | 0.967 | 0.984 | 0.961 |
| cup | 154 | 0.936 | 0.957 | 0.981 | 0.969 |
| fork | 77 | 0.958 | 0.974 | 0.977 | 0.938 |
| glass | 98 | 0.928 | 0.922 | 0.964 | 0.952 |
| knife | 125 | 0.986 | 0.976 | 0.994 | 0.953 |
| plate | 128 | 0.988 | 0.992 | 0.995 | 0.987 |
| spoon | 124 | 0.988 | 0.984 | 0.995 | 0.969 |

Inference took 7.6 ms per image at 640 px.

**What this number means:** the test boxes come from the same YOLOE proposals that a reviewer accepted, and hard images were removed. The score therefore measures how well YOLO26n reproduces reviewed YOLOE boxes on single-object close-ups. It is not object-history accuracy and not counter-view performance. It is a single run with a single seed. A zero-shot COCO baseline has not been run yet; COCO has no plate class.


## Tiled inference with the Kaggle model (2026-10-12, same two photos)

The Kaggle model recognized every object when run on a crop around it (spoon 0.88, glass 0.98,
fork 0.96, plate 0.97, cup 0.95, knife 0.92). The misses on the full photos came from object
size: a knife is 21% x 79% of a Kaggle photo but 3% x 28% of a counter photo, about 19 px wide
at a 640 px input. `objhist_detection.tiled_inference` runs the model on overlapping tiles plus
the full photo and merges the results.

Scored against the reviewed labels above, at confidence 0.25 or above:

| Mode | Photo 1 (6 objects) | Photo 2 (4 objects) |
| --- | --- | --- |
| Single pass, 640 px | 1 correct, 1 false | 1 correct |
| Single pass, 1280 px | 4 correct | 2 correct |
| Tiled (default settings) | 5 correct, 1 wrong class (mug as glass), 1 false | 4 correct, 1 false |

- **False boxes:** both are a "knife" on the wall light switch (0.66 and 0.26).
- **Settings were adjusted after seeing these photos:** even tile spacing and the cross-class
  merge threshold (0.6). Treat these results as a demonstration, not an evaluation; the
  settings must be checked on held-out sessions.
- **Reference labels:** they come from YOLOE boxes accepted in that review, which
  favors box placements close to YOLOE's.

**Lower resolutions (2026-10-12).** Tiles were first a fixed 1280 px, which only suited 4000 px
photos: at 1920 px and below, tiling found no more than a single pass. Tiles are now 0.32 of
the photo's long side (1280 px at 4000 px, so results there are unchanged). The same two photos,
downscaled with Lanczos and scored against the scaled reviewed labels, correct of 10 objects:

| Long side (px) | 4000 | 2560 | 1920 | 1280 | 960 | 640 |
| --- | --- | --- | --- | --- | --- | --- |
| Single pass at 1280 px | 6 | 6 | 6 | 5 | 7 | 5 |
| Tiled, fixed 1280 px tiles | 9 | 9 | 6 | 5 | 7 | 5 |
| Tiled, relative tiles | 9 | 10 | 9 | 9 | 9 | 8 |

- **False boxes with relative tiles:** 0–3 per resolution (most at 2560 and 1920 px).
- **Wrong class with relative tiles:** at most 2.
- **Labeling tool:** YOLOE proposals covered all 10 objects at every resolution down to 640 px.
- **Same caveat:** these are downscaled copies of the same two photos, not independent data.

## Training run `yolo26n-counter-kitchen-1280-v2` (2026-10-12)

- **Data:** `data/processed/counter-plus-kitchen-2026-10-12/data.yaml`. Training uses
  `kaggle-kitchenware-v1` train (3,534 images) plus `counter-2026-10-12` train, with each of
  its 75 photos listed 10 times, 4,284 entries in all. The repetition makes counter photos
  about 18% of each epoch instead of 2%. Validation and test use only `counter-2026-10-12`.
- **Command:** `make train DATA=data/processed/counter-plus-kitchen-2026-10-12/data.yaml
  MODEL=runs/detect/runs/detect/yolo26n-kitchen/weights/best.pt EPOCHS=30 IMGSZ=1280 BATCH=8
  RUN_NAME=yolo26n-counter-kitchen-1280-v2`. It starts from the `yolo26n-kitchen` weights and
  uses seed 0, `deterministic=True`, and the other Ultralytics 8.4.174 defaults.
- **Hardware:** RTX 5070 Laptop GPU 8 GB, torch 2.14.1+cu130; 30 epochs in 0.94 h.
- **Best checkpoint:** epoch 19, picked by Ultralytics fitness on the 9 validation photos.
  `best.pt` SHA-256 `cb28044be38eed795f63b156f9f2a414213b5ecd4de1073aaa3434c3bf30b99a`.
  Validation mAP50 0.923, mAP50-95 0.842. It reached about 0.95 / 0.83 by epoch 4 and then
  stayed between 0.86 and 0.94, which is about as stable as 9 photos allow.
- **Earlier attempt:** `yolo26n-counter-kitchen-1280` was stopped during epoch 1 because the
  repeated paths made Ultralytics skip 76 entries (see the JPEG note above). It is kept, not
  used.

**Test comparison (session `2026-10-12-1643`, 16 photos, 112 objects), scored once after
training.** "Correct" means the right class with IoU ≥ 0.5 at confidence ≥ 0.25. Predictions
are matched greedily, most confident first. A box over an object of another class counts as
wrong class. These counts come from `objhist_evaluation.score_detections` (matching rule
`detector-counts-v1`; commands in [data/README.md](../data/README.md#scoring-detectors-on-a-labeled-split)),
which reproduces the first, one-off script exactly. The YOLOE row comes from its proposals
file, which that command does not read. mAP comes from `yolo detect val ... split=test`.

| Detector | Correct | Wrong class | Missed | False boxes | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| `yolo26n-kitchen`, single pass 640 px | 22 | 3 | 87 | 1 | 0.334 | 0.295 |
| `yolo26n-kitchen`, single pass 1280 px | – | – | – | – | 0.365 | 0.337 |
| `yolo26n-kitchen`, tiled | 48 | 0 | 64 | 7 | – | – |
| YOLOE-26s zero-shot, 1280 px (pre-review proposals) | 67 | 27 | 18 | 7 | – | – |
| `yolo26n-counter-kitchen-1280-v2`, single pass 1280 px | 97 | 6 | 9 | 9 | 0.885 | 0.782 |
| `yolo26n-counter-kitchen-1280-v2`, tiled | 99 | 7 | 6 | 8 | – | – |
| `yolo26n-counter-kitchen-1280-v2`, single pass 1280 px, duplicates merged | 95 | 6 | 11 | 6 | – | – |

Per class, for the new model at 1280 px:

| | cup | fork | glass | knife | plate | spoon |
| --- | --- | --- | --- | --- | --- | --- |
| mAP50-95 | 0.918 | 0.619 | 0.995 | 0.613 | 0.916 | 0.632 |

- **Tiled settings were not checked on held-out data.** On these test photos the old model
  with tiling found 48 of 112 objects, below the 9 of 10 seen on the two photos the tiling
  settings were tuned on. With the new model, tiling adds 2 correct objects and removes 3
  misses, but it also adds 1 wrong class.
- **Cutlery is the weak point:** fork, knife, and spoon score about 0.62 mAP50-95. The other
  classes score at least 0.92. At 1280 px, every one of the new model's mistakes is cutlery:
  spoon read as fork 4 times, fork as spoon once, knife as spoon once, 9 cutlery objects
  missed (fork 3, knife 4, spoon 2), and 9 false cutlery boxes. All 32 cups, 16 glasses, and
  16 plates were found.
- **What this number means:** every session shows the same counter and the same seven objects
  in the same 33 minutes, so the test measures this setup, not other kitchens, lighting, or
  objects. The test labels started as YOLOE boxes, which may favor YOLOE's box placement. This
  is one run with one seed and 16 test photos.

**Validation error analysis (2026-10-12, session `2026-10-12-1624`, 9 photos).** The new model
at 1280 px, single pass, found 58 of 63 objects: 1 wrong class, 4 missed, 8 false boxes, all
cutlery. Claude viewed the 9 photos as downscaled 2000 px copies without metadata, with each box
drawn in a color for its outcome, under the AGENTS.md exception for Benoy's photos. Test photos
were not viewed. Causes:

- **Duplicate boxes on one object (5 of the 8 false boxes):** a second box of another class on
  the same spoon (fork 0.78, fork 0.31, 0.34), or a second fork box inside the fork's box.
  YOLO26 does not suppress overlapping boxes, and the single pass does not merge them; tiled
  inference already merges.
- **Boxes covering part of an object (3 pairs counted as both missed and false):** a knife
  whose black handle blends into the black granite was boxed only along its blade; a spoon
  lying across the fork was boxed together with it; a fork partly behind the glass was boxed
  only where it shows.
- **Partly hidden cutlery:** the fork's label runs through the clear glass, and the knife's
  label stops at the ceramic mug that hides its handle. That is one consistent rule, not two
  (see the [label audit](data-card.md#label-audit-for-partly-hidden-objects-2026-10-12)); the model boxed the fork only where it shows and drew the
  knife's box into the mug.
- **Backlit spoon in the glass:** read as knife (0.47), with only its bright handle visible.
- No false boxes on the wall switches or the stove grates.


## Merging duplicate single-pass boxes (2026-10-12)

Single-pass output now goes through the
tiled merge rule by default (`merge_duplicates`, thresholds 0.5 same-class IoU, 0.7
containment, 0.6 cross-class IoU, all unchanged). The default was chosen on the validation
session: false boxes fell from 8 to 3 there, with 58 objects correct either way, and merged
single pass matched tiled inference (4 false boxes) at a fraction of the cost. Scored once on
the test session afterwards, it gave 95 correct, 6 wrong class, 11 missed, and 6 false boxes,
against 97 / 6 / 9 / 9 without merging. That is 3 fewer false boxes but 2 lost objects, a knife
and a spoon, most likely cutlery lying close enough to another piece to be merged into it. The
default was not changed on the basis of the test result. Revisit it, and the cross-class
threshold, on a larger validation set. `--no-merge-duplicates` keeps the old behavior.


## Model comparison on the 29 validation photos (2026-10-13)

None of these models trained on the validation photos. Counts use `score_detections`
(`detector-counts-v1`, confidence ≥ 0.25). The cutlery columns count forks, knives, and spoons,
87 objects in all.

| Model | Mode | Correct (of 203) | Cutlery correct (of 87) | False cutlery boxes | Glass read as cup |
| --- | --- | --- | --- | --- | --- |
| `yolo26n-counter-kitchen-1280-v2` | single pass, merged | 184 | 68 | 4 | 0 |
| `yolo26n-counter-kitchen-1280-v2` | tiled | 189 | 73 | 10 | 0 |
| `yolo26s-counter-kitchen-1280-b4` | single pass, merged | 181 | 69 | 7 | 4 |
| `yolo26s-counter-kitchen-1280-b4` | tiled | 186 | 73 | 10 | 3 |

- **`yolo26s-counter-kitchen-1280-b4`:** official `yolo26s.pt` (SHA-256
  `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`) trained on
  `counter-plus-kitchen-2026-10-12` at 1280 px for 30 epochs, batch 4, seed 0; 1.94 h; `best.pt`
  `54746e5dd77c8f90a77aa833558d71d743d2fa87f971d1fd4aa3accd9ec36185`. A batch-8 attempt
  (`yolo26s-counter-kitchen-1280`) needed 10.7 GB, spilled into system memory at 18.7 s per
  step, and was stopped in epoch 1. **Result:** the larger model found no more cutlery than the
  nano and added glass-as-cup errors, so the nano was kept. It started from COCO weights rather
  than Kaggle-trained weights, so model size is not the only difference.
- **Confidence threshold (nano, single pass):** at thresholds from 0.05 to 0.50 the correct
  count ranged from 186 to 182, while false boxes went from 16 down to 4. The threshold stays at
  0.25.
- **`yolo26n-counter-kitchen-1600-e15`** (nano at 1600 px, 15 epochs, batch 4, data
  `counter-plus-kitchen-2026-10-13`): stopped after epoch 8 at Benoy's request. It had stayed
  at about 0.78–0.80 mAP50-95 for four epochs (best 0.798 at epoch 5; `best.pt`
  `09b66d9bb113aa707b3b7803310cda10d1cd5240e9bef962ee0f70f5f9c9f49b`), against 0.820 for the
  1280 px nano on the same 29 photos (`yolo detect val`). Higher resolution was not pursued.

## Final training run `yolo26n-counter-final-1280-e15` (2026-10-13)

- **Data:** `data/processed/counter-final-2026-10-13/data.yaml`. Kaggle train plus every
  counter photo except session `2026-10-12-1643`: train 75, val 29, and the night test session
  `2026-10-12-2322` (25), which was moved into training at Benoy's request. That is 129 photos,
  each listed 10 times, 4,824 entries in all. Because Ultralytics' val (session 1624) is also
  trained on, the epoch count was fixed at 15 in advance and the last epoch's weights are used.
- **Command:** `make train DATA=data/processed/counter-final-2026-10-13/data.yaml
  MODEL=runs/detect/runs/detect/yolo26n-kitchen/weights/best.pt EPOCHS=15 IMGSZ=1280 BATCH=8
  RUN_NAME=yolo26n-counter-final-1280-e15`. It used seed 0 and took 0.57 h on the RTX 5070
  Laptop GPU. `last.pt` (copied to `final.pt`) has SHA-256
  `3d2d29c4662893802f5b4752b10eb201b5f971f4c7cfdb6c4fd28aa7ccb19c89`.
- **Held-out test:** session `2026-10-12-1643`, 16 afternoon photos with 112 objects, 48 of
  them cutlery. No model has trained on it. Scored once.

| Model | Mode | Correct | Missed | False boxes | Cutlery correct (of 48) | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `yolo26n-counter-kitchen-1280-v2` | single pass, merged | 95 | 11 | 6 | 31 | 0.885 | 0.782 |
| `yolo26n-counter-kitchen-1280-v2` | tiled | 99 | 6 | 8 | 35 | – | – |
| `yolo26n-counter-final-1280-e15` | single pass, merged | **98** | **7** | **3** | **34** | **0.908** | **0.796** |
| `yolo26n-counter-final-1280-e15` | tiled | 99 | 7 | 8 | 35 | – | – |

Cutlery mAP50-95: fork 0.618 → 0.585, knife 0.613 → 0.645, spoon 0.632 → 0.705.

- **What changed:** with the 54 extra photos (45 of them night), the single pass found 3 more
  pieces of cutlery and drew half as many false boxes. It now matches the old model's tiled
  cutlery count without tiling's cost or its extra false boxes. Spoons improved most; forks got
  slightly worse (3 forks read as spoon).
- **Caveats:** these are 16 photos and 48 pieces of cutlery, so a 3-object difference is small.
  The test session is from the same afternoon, counter, and objects as the training data. Now
  that both night sessions are in training, no held-out night photos remain to measure night
  performance.

## Training run `yolo26s-counter-final-1280-e15` (2026-10-13)

- **Setup:** same data as the final nano (`counter-final-2026-10-13`, 129 counter photos
  including both night sessions). It starts from the official `yolo26s.pt` and trains at 1280
  px for 15 epochs, batch 4, seed 0, taking 1.10 h. `last.pt` SHA-256
  `f9e10aa9ec4d4e29a13f4e4a432f58cb03133d8f3ed100d4c2a0a1c61d3c7e51`.
- **Epoch count:** Benoy's rule was 15 epochs, more if the score was still rising. On the
  monitored validation photos (also trained on) mAP50-95 went 0.954, 0.953, 0.952 over epochs
  13–15, so it stopped at 15.
- **Held-out test (session 1643, 16 photos, 48 cutlery pieces):**

| Model | Mode | Correct (of 112) | Wrong class | Missed | False | Cutlery correct (of 48) | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `yolo26n-counter-final-1280-e15` | single pass, merged | 98 | 7 | 7 | 3 | 34 (71%) | 0.796 |
| `yolo26s-counter-final-1280-e15` | single pass, merged | **104** | **1** | 7 | 5 | **40 (83%)** | **0.830** |
| `yolo26s-counter-final-1280-e15` | tiled | 105 | 1 | 6 | 7 | 41 (85%) | – |

Cutlery mAP50-95 for YOLO26s: fork 0.658, knife 0.697, spoon 0.778 (nano: 0.585, 0.645,
0.705).

- **What changed:** the fork/spoon mix-ups nearly disappeared, from 7 wrong-class objects to 1.
  On the smaller training set (75 photos) YOLO26s was no better than the nano on validation;
  with 129 photos it is clearly better. The larger model needed the extra data.
- **Caveat:** no held-out validation photos remained, so this test session both measured the
  two models and chose between them. That makes 83% slightly optimistic. Confirm it on a new
  test session before relying on it, together with the tiling choice. 16 photos and one seed.
  Session `2026-10-18-1140` (48 photos from a second kitchen) is that new test session; see
  [Provided sequences](data-card.md#provided-sequences).

## Held-out run `yolo26s-counter-kitchen-2026-10-21-1280-e15` (2026-10-21)

- **Data:** `counter-2026-10-21`: train 159 photos (75 first-kitchen plus the 84 of
  `2026-10-18-1055`), val 29 (sessions 1624 and 2314), test 41 (sessions 1643 and 2322), with
  Kaggle train and each counter train photo listed 10 times (`counter-plus-kitchen-2026-10-21`).
- **Setup:** official `yolo26s.pt`, 1280 px, 15 epochs, batch 4, seed 0. `best.pt` (epoch 13,
  val mAP50-95 0.861) SHA-256
  `396ade23caedb3bfb9412315442bd06ea46b44378b1c84b6465bc15318e6b983`.
- **Test (41 photos, 287 objects, 123 cutlery):** mAP50-95 0.775, 248 correct, cutlery 85
  (69%). The control `yolo26s-counter-kitchen-1280-b4` (the same first-kitchen photos without
  the second kitchen) scored 242 and 81, so the second-kitchen photos gave a small gain.
- **Cutlery errors:** about a third are pieces the model found but boxed only in part
  (IoU < 0.5): a fork under the plate rim, a spoon in the glass, a dark knife behind the green
  mug at night, plus piles and fork/spoon mix-ups. The second-kitchen photos mostly show
  cutlery laid apart, so they do not teach these cases.

## Random-split runs (2026-10-21 to 2026-10-28)

Each model starts from the official `yolo26s.pt` and trains at 1280 px for 15 epochs, batch 4,
seed 0, with Kaggle train plus each counter train photo listed 10 times; epoch 15 was best in
every run. The splits are described in the
[data card](data-card.md#random-splits-by-photo-2026-10-21-and-2026-10-27).

**`yolo26s-counter-random-2026-10-21-1280-e15`** (val mAP50-95 0.916; `best.pt`
`de987b57c8e3831b5dc666c4b8d5d2e65f2a687f6382bf38f9db6fb4cef1a80d`). On its 45 test photos:
mAP50-95 0.897, 296 of 311 objects, cutlery 120 of 135 (89%). A 5-epoch continuation at a lower
learning rate did not improve validation and was stopped after 2 epochs.

**New-kitchen test (2026-10-24).** The same model on all 149 labeled third-kitchen photos
(`counter-2026-10-24-test`; it had seen none of that kitchen) found 670 of 1,042 objects (64%)
and 221 of 446 cutlery (50%), mAP50-95 0.636. It read the all-steel dinner knife as a spoon (56)
or fork (14), the appliance as a plate (50 false boxes), and 16 glasses as cups. The 89% from
the random split does not carry over to an unseen kitchen.


**Three runs on the 2026-10-27 split**, all scored on the same 68 test photos with the
[re-reviewed labels](data-card.md#re-review-of-the-first-kitchen-2026-10-28) (470 objects, 203 cutlery; `detector-counts-v1`, single pass, duplicates
merged, confidence ≥ 0.25):

| Run | Training labels | Augmentation | `best.pt` SHA-256 | Val mAP50-95 | Test mAP50-95 | Correct | Cutlery | Kitchen 1 night cutlery (of 27) | False |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `yolo26s-counter-random-2026-10-27-1280-e15` | before re-review | default | `4233cd0583aa2b3670431cc4b479a50f8f1d4c18da55e68ae67f3d926e0bac98` | 0.945 | 0.934 | 453 | 188 | 21 | 8 |
| `yolo26s-counter-random-relabeled-2026-10-28-1280-e15` | re-reviewed | default | `61af703cdb2dd8bd0679f5dfac4d16271f27befc4a487817aabe6f36eaf1c3b5` | 0.953 | 0.935 | 452 | 187 | 23 | 6 |
| `yolo26s-counter-random-relabeled-flipud-2026-10-28-1280-e15` | re-reviewed | + `flipud=0.5` | `57f3572d41b0ff925cb2e58f4659a0a9a5f66740b22c1fa17f2476bcfe24ebcb` | 0.951 | 0.934 | 453 | **189 (93%)** | **25** | 6 |

- **By kitchen, relabeled run (cutlery):** first kitchen day 43 of 51, night 23 of 27; second
  kitchen 64 of 66; third kitchen 57 of 59.
- **By kitchen, flip run (cutlery):** first kitchen day 44 of 51, night 25 of 27; second
  kitchen 63 of 66; third kitchen 57 of 59 (3 false boxes). Per-class mAP50-95: fork 0.886,
  knife 0.889, spoon 0.896. At IoU ≥ 0.1, 196 of 203 cutlery pieces are found.
- **Reading the table:** the three runs differ by 1–3 objects, which is noise on 68 photos.
  The only steady trend is night cutlery in the first kitchen (21 → 23 → 25). The flip run is
  the current best model.
- **Rotation was not used:** rotating a photo makes Ultralytics enlarge each box to cover the
  rotated corners, which loosens boxes on thin diagonal cutlery (+42% area for a diagonal knife
  at 10°). A vertical flip keeps boxes exact. Benoy chose the flip only.
- **Caveat:** with a random split, near-duplicate photos from one moment can sit in both train
  and test, so these numbers are optimistic. The new-kitchen test above shows how far an
  unseen setting can fall. A new, unseen session is needed for an honest score.
