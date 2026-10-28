# Kitchen object dataset — data card

Status: draft. Current training data: `kaggle-kitchenware-v1` plus
`counter-random-2026-10-27-relabeled` (449 counter photos from three kitchens). Datasets live
under `OBJHIST_DATA_DIR` (default `data/`) and are never committed; this card records where
they came from, how they were labeled, and how they are split so results can be reproduced.
Training runs and their scores are in [detector results](detector-results.md).

## Intended use

Fine-tune a YOLO26 detector for objects on a shared kitchen counter photographed from any
angle (handheld or loosely placed, from the side to straight down), and build scripted
scenarios for evaluating the object history.

## Candidate sources

| Source | Content | Usable for detector training | Open questions |
| --- | --- | --- | --- |
| [Roboflow Kitchen Items](https://universe.roboflow.com/new-bywkn/kitchen-items-u5c2t) | 1,839 images, boxes for 9 classes: cup, plate, spoon, fork, knife, dish, kettle, pan, rice cooker | Yes | License and version to record at download; viewpoint differs from an overhead counter camera |
| [Kaggle Kitchenware Classification](https://www.kaggle.com/competitions/kitchenware-classification/data) | Image-level labels for 6 categories: cups, glasses, plates, spoons, forks, knives | Only after box annotation; see [below](#kaggle-kitchenware-auto-labels) | Training confirmed allowed (2026-10-09); no redistribution |
| Provided snapshot sequences | Ordered images of the counter from any camera angle, supplied by Benoy in the format in [data/README.md](../data/README.md) | Yes, after box labeling | Required for the target classes and viewpoint; see gaps below |
| COCO-pretrained YOLO26 | Pretrained classes include cup, bottle, bowl, knife, spoon, fork, wine glass, banana, apple, orange | Zero-shot baseline only | Useful to measure what fine-tuning adds |

## Kaggle kitchenware auto-labels

Status: reviewed and exported as dataset version `kaggle-kitchenware-v1` (2026-10-09); one
YOLO26n model trained on it
([detector results](detector-results.md#training-run-yolo26n-kitchen-2026-10-09)). Commands are in
[data/README.md](../data/README.md#kaggle-kitchenware-auto-labels); the code is
`packages/detection/src/objhist_detection/kaggle_kitchenware/`.

**License: training permitted.** On 2026-10-09 Benoy checked the competition terms
(DataTalks.Club, December 2022 to February 2023, images collected with Toloka) and confirmed
that training on the images for this course project is allowed. The images and anything derived
from them (boxes, exports, weights) stay local and are not redistributed; add the exact license
name from the Rules tab here when it is next opened.

### Source facts (checked 2026-10-09)

- `train.csv`: 5,559 labeled images; `test.csv`: 3,808 unlabeled images, not used.
  SHA-256 of `train.csv`: `db6d55c5d0d6fc8d1e7ea70fd41672215d795578ceed26dab538c9dcb7957719`.
- Labels per class: plate 1,227, cup 1,135, spoon 989, knife 909, glass 742, fork 557.
- One centered object per photo, mostly close-up. All images are RGB JPEG; 52 distinct sizes,
  most often 750x1000 (3,968 images). Three images carry an EXIF rotation; they are proposed and
  exported upright.
- Three pairs of byte-identical images have conflicting labels (knife/fork, cup/glass,
  spoon/fork). They are kept in the same split and flagged; at most one of each pair can be
  right.

### Method

1. **Split before labeling.** `splits.csv` assigns 70/15/15 per class with seed 5330 from the
   labels and image hashes only, so detector output cannot influence the split. Identical
   images share a split. Result: train 3,891, val 836, test 832.
   SHA-256 of `splits.csv`: `84819968ece3f71f0e61337f24b908785a452ee215a0a8a2db907e5b25954363`.
2. **Propose boxes.** YOLOE-26s-seg (`yoloe-26s-seg.pt`, official Ultralytics release) is
   prompted with the six class names ("drinking glass" for glass). The highest-confidence box
   becomes the object box, and its class always comes from `train.csv`. YOLOE's class for the box
   disagreed with the label on 1,602 images (29%), almost all one-directional (glass read as cup
   680 times, spoon as knife 307, fork as knife 250, plate as cup 187). That is a systematic bias,
   not evidence of mislabeling, so it is recorded but is not a review reason.
3. **Flag for review**, with thresholds fixed before any review:
   no box at confidence 0.05 or above; top box below confidence 0.4; a second box at
   confidence 0.4 or above with less than half its area inside the top box (a second object that
   would be left unlabeled); byte-identical duplicate. Result on 2026-10-09: 1,312 of 5,559
   images flagged (875 low confidence, 355 second object, 76 no box, 6 duplicates; no image had
   more than one reason). No-box images are mostly plates (35) and spoons (23) that YOLOE missed.
   SHA-256 of `proposals.csv`: `27240ced66d697175adf5c9883c39980e195bba50bb7be335922a3dc37187313`; model SHA-256 and versions are in
   `proposals.meta.json`.
4. **Review** (designed for people on the generated `review.html`; this release was reviewed by
   spot-checked and approved by Benoy, see below): every validation and test image, every
   flagged training image, and a seeded random spot-check of 300 unflagged training images.
   Reviewers accept, reject (bad box, several objects, or not a listed class), or change the
   class. Queue: 2,872 images (1,668 validation/test, 904 flagged training, 300 spot-check).
5. **Export** to `data/processed/kitchen/` in YOLO format. Images without a box and rejected
   images are dropped; flagged training images are kept only if accepted. The export refuses to
   run while any validation or test image lacks a verdict. `summary.json` records the counts and
   review outcomes; copy them into the table below.

### Review outcomes (2026-10-09)

**Who reviewed:** spot-checked and approved by Benoy (2026-10-18). The per-image verdicts came from an automated pass that read 175 contact sheets of 16 images, each with its proposed box drawn on. The rubric below was calibrated on two sheets, and seven parallel review passes using it covered the rest. The reviewer column in `reviews.csv` names that pass. Images without a box were rejected automatically. Three duplicate pairs were decided by keeping the correctly labeled copy (0237 knife, 8532 glass, 7196 fork).

**Rubric:**
- Accept a box that encloses the whole object.
- Reject when the box covers only part of the object (a handle, a rim, or a cup handle more than about 15% outside), sits on background, another listed object is clearly visible, or the object is not a listed class (bowls, pans, ladles).
- Relabel only when the label is clearly wrong.

| Queue | Queued | Without a box | Rejected (box shown) | Relabeled | Error rate among images with a box |
| --- | --- | --- | --- | --- | --- |
| Validation and test | 1,668 | 26 | 215 | 10 | 13.7% (225/1,642) |
| Flagged training | 904 | 50 | 285 | 1 | 33.5% (286/854) |
| Training spot-check | 300 | 0 | 22 | 3 | 8.3% (25/300; 95% CI 5.7–12.0%) |

- **Unreviewed training images:** the spot-check estimates that about 8% of the 2,979 unreviewed training images carry a bad box or label. Most are partial boxes (a spoon handle without its bowl) or bowls labeled plate.
- **Audit:** a random 32 of the verdicts (16 accepts, 16 rejects) were re-checked, and 31 were confirmed. The miss was a deep bowl accepted as plate, so bowl-versus-plate leniency is the main known reviewer error.
- **Reviewer inconsistency:** reviewers differed on opaque handleless tumblers labeled glass. Some relabeled them cup (12 glass-to-cup relabels in total) and others kept glass. This leaves a little cup/glass label noise.
- **Before reporting test results externally**, a person should re-check the test split.

### Exported dataset `kaggle-kitchenware-v1`

`data/processed/kitchen/`; class order `cup, fork, glass, knife, plate, spoon`.

| Split | cup | fork | glass | knife | plate | spoon | Total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 737 | 377 | 490 | 604 | 718 | 608 | 3,534 |
| val | 153 | 82 | 99 | 127 | 134 | 126 | 721 |
| test | 154 | 77 | 98 | 125 | 128 | 124 | 706 |

- Excluded: 598 images (76 without a box, 522 rejected).
- `manifest.csv` SHA-256: `421bdcd1b41f7ba19ca661bd3efe3910ef09fe1e81a3bbce6fae0ce418d7d039`.
- `reviews.csv` SHA-256: `85cd382dd43c7ba31b94f36184fbd77ba7cad18e73c17ce3962c5a0195d8013b`.
- Input hashes are in `summary.json`.

### Limitations

- Close-up single-object photos differ from the multi-object counter views the project
  targets; this set can only supplement in-domain data, and all evaluation stays on the provided
  sequences.
- Dropping images without a usable box removes hard cases from validation and test, so metrics
  on this set are optimistic.
- Near-duplicate photos (same object, slightly different shot) are not detected; only
  byte-identical images are grouped. Validation metrics may be inflated by such pairs.
- No milk, dish soap, or other scenario classes.

## Counter-session auto-labels

Status: dataset `counter-2026-10-12` exported from 100 photos reviewed by Benoy (2026-10-12),
and one YOLO26n model fine-tuned on it; see [below](#dataset-counter-2026-10-12) and
[detector results](detector-results.md).
Code: `packages/detection/src/objhist_detection/counter_labels/`; commands in
[data/README.md](../data/README.md#counter-session-auto-labels).

**Why it exists.** On Benoy's first two counter photos (`2026-10-12-counter-test`: Galaxy S25
Ultra, 4000x2252 JPEG, EXIF times with UTC offset -04:00, handheld from standing height), the
Kaggle-trained YOLO26n at its default 640 px found only the plate (1 of 6 and 1 of 4 objects at
confidence 0.25 or above, plus a wrong "cup" box on the plate in the first photo). Text-prompted YOLOE at 1280 px boxed every object.

**Method.**
1. YOLOE-26s-seg (same weights and prompts as the Kaggle pipeline) runs at 1280 px with
   class-agnostic NMS. Every box with confidence 0.15 or above becomes a proposal, so one glass
   is not proposed as both cup and glass.
2. A reviewer keeps or rejects each proposal, sets its class, and draws boxes for missed
   objects. A photo is exported only once every proposal has a verdict and it is marked done.
   Since 2026-10-13 a loose or cut-off box can be adjusted (drag handles or type edges); the
   review records it as `adjusted`, and the export's `kept_adjusted` counts it. Earlier
   reviews could only keep a proposed box as is or reject and redraw it.
3. Whole sessions are assigned to splits; validation and test sessions must be fully reviewed.

**First check (2026-10-12, 2 photos, pipeline test only).**

| Photo | Objects (from Benoy's log) | Proposed | Kept, same class | Kept, relabeled | Rejected | Drawn |
| --- | --- | --- | --- | --- | --- | --- |
| 20261012_134754 | 6 | 7 | 5 | 1 (glass proposed as cup) | 1 (stove grate as spoon) | 0 |
| 20261012_134837 | 4 | 8 | 3 | 1 (glass proposed as cup) | 4 (stove grates as spoon) | 0 |

- **Proposals:** every object in Benoy's log got a correct box; no box needed drawing. YOLOE's
  known glass-as-cup bias appeared on both glasses, and its false positives were low-confidence
  "spoons" on the stove grates (0.19-0.33).
- **Who reviewed:** spot-checked and approved by Benoy (2026-10-18). The two photos were never used for training;
  the 100 photos below replaced them.
- **Export check:** the exported labels were drawn back onto the images and matched the log.
  Ultralytics loaded the export (2 images, 10 boxes) for a one-epoch load check.
- **Too small to measure:** two photos are a smoke test, not a measurement. Detector accuracy
  on counter photos needs held-out sessions.

### Dataset `counter-2026-10-12`

`data/processed/counter-2026-10-12/`; class order `cup, fork, glass, knife, plate, spoon`.

- **Photos:** 100 handheld photos of one counter, shot 16:18:50–16:51:45 on 2026-10-12
  (-04:00). The same seven objects are in every photo: white plate, silver fork, silver knife,
  silver spoon, transparent glass, green mug, white design mug. The two mugs are both `cup`.
- **Sessions:** the photos arrived as one folder. They were split into six sessions wherever
  the shooting paused, which was every gap of at least 47 s (the median gap is 15 s). Each
  session folder holds byte-identical copies of its photos (SHA-256 checked) and its own
  `changes.xlsx`. The original folder `2026-10-12-counter-test` was deleted on 2026-10-12, after
  every photo and change-log row was confirmed to have an identical copy in the sessions.
- **Splits** (`session_splits.csv`, frozen before review, SHA-256
  `21c114dabd1274e6e90fce1646bcda5747da6d840f0ec1fca83debbe88afccfd`):

| Split | Sessions | Photos | cup | fork | glass | knife | plate | spoon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 1618, 1628, 1635, 1647 | 75 | 150 | 75 | 75 | 75 | 75 | 75 |
| val | 1624 | 9 | 18 | 9 | 9 | 9 | 9 | 9 |
| test | 1643 | 16 | 32 | 16 | 16 | 16 | 16 | 16 |

- **Proposals:** YOLOE matched the known object list in none of the 100 photos. It called the
  glass "cup" in all 100, missed the fork in 48 and the knife in 29, and added 92 extra spoons.
- **Review:** Benoy reviewed all 100 photos on the review page: 475 boxes kept with the same
  class, 140 relabeled, 134 rejected, and 85 drawn. Each photo's labels were then compared with
  the known object list. That flagged 7 photos (a glass left as cup, an extra spoon, a missing
  mug, a fork labeled spoon), and Benoy corrected all 7. After the corrections all 100 photos
  match the list. The check counts labels only; it cannot catch a box that is in the wrong
  place.
- **Hashes:** `manifest.csv` `ec54ffb8f36917fb735a146939eaff00aeed96130ab7c6e2c363703e27dd4dbb`,
  `reviews.json` `fe5277059a01448265370e74272ac296f773b1637ec25df3ab30b85937feb6aa`. Input
  hashes are in `summary.json`.
- **Ultralytics rewrites these JPEGs:** the S25 adds data after the JPEG end marker, so
  Ultralytics treats each image as corrupt and saves a repaired copy the first time it scans
  the folder. It writes a new file, so the hard-linked raw photos are not changed (all 100 raw
  hashes were re-checked). A training list that names the same image more than once makes
  data-loader workers repair the same file at the same time; on Windows this fails and the
  image is skipped. Before training, scan each split once with unique paths.

### Label audit for partly hidden objects (2026-10-12)

Rule, now on the review page and in
[data/README.md](../data/README.md#counter-session-auto-labels): a box encloses the visible part
of its object; parts seen through clear or tinted glass count as visible, and parts hidden
behind or inside something opaque (a ceramic mug, under the plate) are left out. Every photo
where a labeled fork, knife, or spoon box covers more than 10% of a mug or glass box was
checked: 40 of the 100 photos (31 train, 5 val, 4 test). Claude viewed downscaled crops without
metadata showing the labels only, with no detector output, plus 3 full photos where the crop
was too tight. All 40 already follow the rule, so no label was changed and no retraining is
needed for it. Cutlery under the plate is not covered by the 10% filter; the 3 full photos
showed none that broke the rule.


### Dataset `counter-2026-10-13` (adds two night sessions)

`data/processed/counter-2026-10-13/`; same class order. It is `counter-2026-10-12` plus 45
photos shot at night (23:14–23:30, -04:00) with the same seven objects. They were split at a
45 s pause into `2026-10-12-2314` (20 photos, val) and `2026-10-12-2322` (25 photos, test), at
Benoy's choice. The earlier sessions keep their splits.

| Split | Photos | cup | fork | glass | knife | plate | spoon |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 75 | 150 | 75 | 75 | 75 | 75 | 75 |
| val | 29 | 58 | 29 | 29 | 29 | 29 | 29 |
| test | 41 | 82 | 41 | 41 | 41 | 41 | 41 |

- **Proposals:** YOLOE ran on the CPU while the GPU was training. It matched the object list in
  none of the 45 photos (glass as cup in all 45, 15 forks and 22 knives missing, 45 extra
  spoons).
- **Review:** Benoy reviewed all 45 (recorded once as `Benoy` and once as `benoy`, the same
  person). The label check flagged 9 photos, all of them slips: 5 glasses left as cup, 2 forks
  labeled spoon or knife, a knife standing in a mug left unboxed, and a glass left unboxed.
  Claude viewed the 9 as downscaled copies without metadata, with labels only, and Benoy
  corrected them. All 145 photos now match the list. Review totals over the 145 photos: 658
  kept, 203 relabeled, 230 rejected, 154 drawn.
- **Hashes:** `manifest.csv` `4ae66c5a60b6ab3ce60e1392ce8e5faa94b4be9d4cff04b3a3b748a047794dfd`,
  `reviews.json` `f92109162f7c509e103aae999b84c523aaf7c769428ba6d70175a04130b7e395`,
  `session_splits.csv` `25380b745b5020472c9eecc7265ace24fe6e0d5216cdfa119b3588358a12d33e`.

### Random splits by photo (2026-10-21 and 2026-10-27)

After all photos were reviewed, Benoy chose to split by photo at random instead of by session,
after being told that consecutive photos are near-duplicates and that this makes test scores
optimistic (see [Known gaps](#known-gaps)). This is a deliberate exception to the session-split
rule in AGENTS.md. Every kitchen appears in every split. The models trained on these splits are
in [detector results](detector-results.md#random-split-runs-2026-10-21-to-2026-10-28).

| Dataset | Seed | Train | Val | Test | `split.json` SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `counter-random-2026-10-21` (kitchens 1–2, 300 photos) | 20261021 | 210 | 45 | 45 | `ce0083d5ef1dd47a33e23039566a906281083e53617b8ccdac4d0385584a7185` |
| `counter-random-2026-10-27` (kitchens 1–3, 449 photos) | 20261027 | 314 | 67 | 68 | `c3f2485ce20de5d4dcff9e2ac781882cf47ef6515cb04178e7405bff77ca9392` |

`counter-random-2026-10-27-relabeled` keeps the 2026-10-27 split and takes its labels from
Benoy's re-review of his first-kitchen photos (below); `manifest.csv` SHA-256
`b5b12717f5975ad5e251317a45ea0c24f23bceeed3715802f9ba192796a3295d`.

### Re-review of the first kitchen (2026-10-28)

Benoy re-reviewed all 145 first-kitchen photos: 140 changed, mostly
tightened boxes (371 adjusted, 154 drawn). `reviews.json` now holds 450 photo reviews, SHA-256
`31be658fb3af45f8c56a3c803408eccd735f982b00951dbbe983014acc6259ee`.


### Label fixes and photo viewings by Claude (2026-10-21 to 2026-10-28)

All photos were Benoy's (see AGENTS.md); each viewing was a downscaled copy (1400–2000 px)
without EXIF, and no photo showed a person.

- **Label fixes:** two glasses labeled cup were relabeled glass (`2026-10-18-1140/IMG_6664`,
  `-1155/IMG_6721`) after Benoy's fix did not save; a knife labeled cup was relabeled knife
  (`2026-10-24-0036/IMG_4385`); two forks were relabeled (`IMG_4476` cup → fork, `IMG_4509`
  knife → fork). Benoy was shown the first two. Earlier `reviews.json` versions were kept as backups
  until 2026-10-28, when they were deleted at Benoy's request.
- **Confirmed by Benoy as correct:** `IMG_6571`–`6610` have no glass (not yet on the counter);
  `IMG_6718` has no fork; `IMG_4479` has no spoon; `IMG_4371` has no knife (Benoy's call:
  Claude had read a steel handle under the plate as a knife).
- **Photos viewed:** `IMG_6571`, `IMG_6611`, `IMG_6723` (labels and suggestions); first-kitchen
  `2322/232937`, `2322/232809`, `1643/164614` (detector mistakes drawn); 6 validation and test
  photos whose labels did not match the object list, plus `IMG_6664` and `IMG_6721`;
  third-kitchen `IMG_4364`, `4371`, `4378`, `4385`, `4396`, `4427`, `4456`.

## Known gaps

- The motivating queries mention milk and dish soap, which no candidate dataset labels. The
  final class list must be chosen with the scenarios in mind, and those classes need team
  annotations on the provided sequences.
- Public images are mostly close-up single objects. The provided counter sequences, shot from
  any angle, are needed for in-domain training and must be used for all evaluation.
- Consecutive snapshots from one session are nearly identical, which inflates metrics if they
  are split randomly.

## Provided sequences

Record for each delivered batch: number of sessions, images per session, snapshot interval,
camera position, resolution, where capture times came from (EXIF or manifest), and the time
zone.

- `2026-10-12-counter-test` (test batch): 2 photos, 43 s apart; handheld phone (Galaxy S25
  Ultra) from standing height, so the framing shifts between photos; 4000x2252 JPEG; capture
  times from EXIF `DateTimeOriginal` with offset -04:00; change log `changes.xlsx`. Sessions without reliable capture times cannot support time-based answers.
  Those two photos were later replaced by the 100 photos below, which were copied into six
  sessions; the folder was then deleted. The pipeline-test proposals and reviews for the two
  photos were removed on 2026-10-28, when `data/interim/counter-labels/` became the working
  folder for all counter photos. Earlier `reviews.json` versions were deleted the same day at
  Benoy's request; only the latest reviews are kept.
- `2026-10-12-1618`, `-1624`, `-1628`, `-1635`, `-1643`, `-1647` (first labeled batch): 100
  photos in 6 sessions of 16, 9, 18, 28, 16, and 13 photos, about 15 s apart (median); same
  phone, handheld; 4000x2252 JPEG; capture times from EXIF `DateTimeOriginal`,
  `SubsecTimeOriginal`, and `OffsetTimeOriginal` (-04:00). Change logs list the same seven
  objects as present in every photo.
- `2026-10-18-1055`, `-1140`, `-1155` (second kitchen, 2026-10-18): 155 photos (IMG_6571–6726,
  IMG_6628 absent) taken by Benoy in a second kitchen with an iPhone, 4032x3024 JPEG (3
  portrait), and delivered as a Google Drive zip (exported 2026-10-19 03:26 UTC). The photos
  carry no EXIF capture time, camera model, or GPS.
  - **Capture times:** the manifests use the zip's file times with offset -04:00, which Benoy
    checked and confirmed as the real capture times (2026-10-18). They are whole seconds. In
    `-1055`, the times of the 11 photos IMG_6600–6610 do not follow their file numbers,
    so its manifest lists photos in time order.
  - **Splits** (by Claude, at Benoy's request), cut at a 16-minute pause in the shoot:
    `-1055` = train (84 photos, 10:56–11:24, proposed by `yolo26s-counter-final-1280-e15`);
    `-1140` = test (48 photos, 11:40–11:54) and `-1155` = validation (23 photos,
    11:56–12:02). YOLOE's suggestions for these two were unusable (13.6 boxes per photo), so at
    Benoy's request a throwaway copy of `yolo26s-counter-final-1280-e15` (`f9e10aa9…`)
    suggested their boxes (6.5 per photo) and was then deleted; `proposals.meta.json` records it
    as `helper-detector`. Later models train from the official `yolo26s.pt`, never from it.
    These splits were replaced by the
    [random splits](#random-splits-by-photo-2026-10-21-and-2026-10-27).
  - **Domain shift:** before any review, the YOLO26s detector found exactly the seven-object
    list (2 cups, 1 each of fork, glass, knife, plate, spoon) in only 28 of the 155 photos.
    Missed glasses were the most common difference. Claude did not view the photos for this
    count.
- `2026-10-24-0007`, `-0026`, `-0036`, `-0040`, `-0047`, `-0108`, `-0120`, `-0133` (third
  kitchen, night of 2026-10-23/24): 150 photos (IMG_4364–4514; 8, 6, 10, 8, 23, 7, 29, and 59
  per session, split at pauses in shooting), provided by Benoy, who allowed Claude to view them.
  iPhone 17 Pro, shot top-down over a white counter, with an all-steel dinner knife. Capture
  times come from EXIF with a UTC offset; the raw files also hold GPS, which stays local.
  Boxes were suggested first by a throwaway copy of `f9e10aa9…`, then, for the 114 photos not
  yet reviewed, by `helper-2026-10-24` (fine-tuned from it on the reviewed photos; weights
  deleted after use). Benoy reviewed all 150.

## Class list

Not decided. The Kaggle export uses `cup, fork, glass, knife, plate, spoon` (indices 0-5,
alphabetical) as a provisional list; new classes should be appended so these indices stay valid. Record the final ordered class list here and keep it identical to the training
`data.yaml`; class indices are part of the shared contract with tracking and evaluation.

## Splits

- Group by capture session: every frame of one session or scripted scenario belongs to exactly
  one split.
- Freeze validation and test splits before any tuning; record their image lists and hashes.
- Scripted evaluation scenarios are held out from detector training entirely.

## Versioning

For each dataset version, record here: source URLs and export versions, download date, class
list, image counts per split, label format, and a SHA-256 of the archive or image list.

## Privacy

Images of a shared kitchen can include people. Avoid capturing faces where possible, keep raw
captures out of the repository and external services, and remove frames whose subjects have
not agreed to be recorded.
