"""Record detector training runs in a local MLflow store so they can be compared in the MLflow UI.

Run from the repository root:

    uv run python -m objhist_detection.experiment_tracking init
    uv run python -m objhist_detection.experiment_tracking import-runs runs/detect runs/detect/old
    uv run python -m objhist_detection.experiment_tracking log-models --current-best <run name>
    uv run python -m objhist_detection.experiment_tracking import-scores data/interim/scores/*.json
    make mlflow-ui      # http://127.0.0.1:5000

`make train` creates the experiment, refuses a run name that is already taken, lets Ultralytics'
MLflow callback log the run live (it reads `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME`, which
the Makefile sets), and then runs `import-runs` on the new folder to record the weight hashes. For
runs trained before MLflow was set up, `import-runs` rebuilds the run from its `args.yaml` and
`results.csv`. It never copies weights into the store; it records their paths and SHA-256 instead.
`log-models` adds one model per run to the experiment's Models tab: the run's `best.pt` (or the
checkpoint chosen with `--checkpoint`), with the validation metrics of its epoch, its path, and its
SHA-256. `import-scores` attaches `objhist_evaluation.score_detections --json-out` results to the
run, and the model, whose weights produced them; run it after `log-models` to link earlier scores.
Every step can be rerun; the imports exit 1 if anything was left unrecorded.
"""

import argparse
import csv
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml
from mlflow.entities import LoggedModel, LoggedModelStatus, Metric, Param, Run, RunStatus
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from objhist_detection.annotation import file_sha256

DEFAULT_TRACKING_URI = "sqlite:///runs/mlflow/mlflow.db"
DEFAULT_EXPERIMENT_NAME = "objhist-detector"
SQLITE_URI_PREFIX = "sqlite:///"
ARTIFACTS_DIRNAME = "artifacts"
"""Artifacts of a SQLite store go next to its database, so the whole store is one folder."""

RUN_DIR_TAG = "objhist.run_dir"
BACKFILLED_TAG = "objhist.backfilled"
BACKFILL_INCOMPLETE = "incomplete"
"""`BACKFILLED_TAG` value until every param, metric, and artifact is logged; reruns replace it."""
EPOCHS_COMPLETED_TAG = "objhist.epochs_completed"
STOP_REASON_TAG = "objhist.stop_reason"
WEIGHTS_TAG_PREFIX = "objhist.weights."
SCORES_ARTIFACT_DIR = "scores"
EXPERIMENT_KIND_TAG = "mlflow.experimentKind"
MODEL_TRAINING_KIND = "custom_model_development"
"""Without this tag, MLflow 3's UI may guess a GenAI layout (traces) that hides run details."""

# Logged models (the experiment's Models tab). The checkpoint tag is set on runs; the rest on models.
MODEL_TYPE = "yolo26-detector"
BEST_CHECKPOINT = "best"
MODEL_CHECKPOINT_TAG = "objhist.model_checkpoint"
MODEL_SHA256_TAG = "objhist.checkpoint_sha256"
MODEL_PATH_TAG = "objhist.checkpoint_path"
CURRENT_BEST_TAG = "objhist.current_best"
MODEL_PARAM_KEYS = ("model", "data", "epochs", "imgsz", "batch", "seed", "optimizer", "flipud")
CHECKPOINT_METRIC_PREFIXES = ("metrics/", "val/")
"""Validation metrics, copied to a model from the epoch its checkpoint holds."""

# Ultralytics' callback logs these file types from the run folder; weights are left out on purpose.
ARTIFACT_SUFFIXES = frozenset({".png", ".jpg", ".csv", ".yaml"})
UNLOGGED_RESULT_COLUMNS = frozenset({"epoch", "time"})
FITNESS_COLUMN = "metrics/mAP50-95(B)"
"""Ultralytics 8.4 detection fitness (what `patience` early stopping watches) is mAP50-95 alone."""
FITNESS_METRIC = "metrics/mAP50-95B"
"""`FITNESS_COLUMN` as logged to MLflow."""
CUTLERY_CLASSES = ("fork", "knife", "spoon")
COUNT_FIELDS = ("objects", "correct", "wrong_class", "missed", "false_boxes")

# MLflow request limits: log_batch entities per call and search_runs page size.
MAX_PARAMS_PER_BATCH = 100
MAX_METRICS_PER_BATCH = 1000
SEARCH_PAGE_SIZE = 1000
LOGGED_MODEL_PAGE_SIZE = 100


class TrackingError(ValueError):
    """A run folder or score file cannot be recorded as given."""


@dataclass(frozen=True)
class TrainingRun:
    """One Ultralytics run folder (`runs/detect/<name>/`)."""

    run_dir: Path
    args: Mapping[str, object]
    epochs: Sequence[Mapping[str, float]]
    finished_at_ms: int
    """Modification time of `results.csv`, written after every epoch."""

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def planned_epochs(self) -> int | None:
        epochs = self.args.get("epochs")
        return epochs if isinstance(epochs, int) else None

    @property
    def weights(self) -> list[Path]:
        return sorted((self.run_dir / "weights").glob("*.pt"))


def metric_name(results_column: str) -> str:
    """The name Ultralytics' MLflow callback uses for a `results.csv` column."""
    return results_column.strip().replace("(", "").replace(")", "")


def path_label(path: Path) -> str:
    """A path as recorded in tags: relative to the working directory when inside it."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def artifact_location(tracking_uri: str) -> str | None:
    """Where a new experiment keeps artifacts: beside a SQLite database, else the server decides."""
    if not tracking_uri.startswith(SQLITE_URI_PREFIX):
        return None
    database = Path(tracking_uri.removeprefix(SQLITE_URI_PREFIX))
    return (database.parent / ARTIFACTS_DIRNAME).resolve().as_uri()


def ensure_experiment(client: MlflowClient, tracking_uri: str, name: str) -> str:
    """Return the experiment id, creating it (and a SQLite store's folder) if needed."""
    if tracking_uri.startswith(SQLITE_URI_PREFIX):
        Path(tracking_uri.removeprefix(SQLITE_URI_PREFIX)).parent.mkdir(parents=True, exist_ok=True)
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        return client.create_experiment(
            name,
            artifact_location=artifact_location(tracking_uri),
            tags={EXPERIMENT_KIND_TAG: MODEL_TRAINING_KIND},
        )
    if experiment.lifecycle_stage != "active":
        raise TrackingError(
            f"experiment {name} is deleted; restore it with `uv run mlflow experiments restore "
            f"--experiment-id {experiment.experiment_id}` or pass another --experiment"
        )
    if EXPERIMENT_KIND_TAG not in experiment.tags:
        # Also covers experiments Ultralytics' callback created before `init` ran.
        client.set_experiment_tag(
            experiment.experiment_id, EXPERIMENT_KIND_TAG, MODEL_TRAINING_KIND
        )
    return experiment.experiment_id


def search_all_runs(client: MlflowClient, experiment_id: str, filter_string: str = "") -> list[Run]:
    runs: list[Run] = []
    page_token: str | None = None
    while True:
        page = client.search_runs(
            [experiment_id],
            filter_string=filter_string,
            max_results=SEARCH_PAGE_SIZE,
            page_token=page_token,
        )
        runs.extend(page)
        page_token = page.token
        if not page_token:
            return runs


def runs_named(client: MlflowClient, experiment_id: str, run_name: str) -> list[Run]:
    if "'" in run_name:
        raise TrackingError(f"run name {run_name!r} contains a quote, which MLflow filters reject")
    return search_all_runs(client, experiment_id, f"attributes.run_name = '{run_name}'")


def check_new_run_dir(client: MlflowClient, experiment_id: str, run_dir: Path) -> None:
    """Refuse a training run whose folder or name already exists, before any GPU time is spent.

    Ultralytics would rename a clashing folder to `<name>2`, and a second MLflow run of the same
    name could not be told apart from the first by name.
    """
    if run_dir.exists():
        raise TrackingError(f"{run_dir} exists; pick another RUN_NAME")
    taken = runs_named(client, experiment_id, run_dir.name)
    if taken:
        folder = taken[0].data.tags.get(RUN_DIR_TAG, "no folder recorded")
        raise TrackingError(
            f"MLflow already has a run named {run_dir.name} ({folder}); pick another RUN_NAME"
        )


def find_run_dirs(paths: Iterable[Path]) -> list[Path]:
    """Run folders among `paths`: each path is a run folder or a folder of run folders."""
    run_dirs: list[Path] = []
    for path in paths:
        if (path / "args.yaml").is_file():
            run_dirs.append(path)
            continue
        children = sorted(child for child in path.iterdir() if (child / "args.yaml").is_file())
        if not children:
            raise TrackingError(
                f"{path} is not a run folder and holds no run folders (no args.yaml)"
            )
        run_dirs.extend(children)
    return run_dirs


def read_training_run(run_dir: Path) -> TrainingRun:
    args_path, results_path = run_dir / "args.yaml", run_dir / "results.csv"
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(args, dict) or not all(isinstance(key, str) for key in args):
        raise TrackingError(f"{args_path} is not a mapping of training arguments")
    if not results_path.is_file():
        raise TrackingError(f"{run_dir} has no results.csv; the run did not finish an epoch")
    with results_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise TrackingError(f"{results_path} has no epochs")
    epochs: list[dict[str, float]] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            epochs.append({column.strip(): float(value) for column, value in row.items()})
        except (TypeError, ValueError) as error:
            raise TrackingError(f"{results_path}:{line_number} has a non-numeric value") from error
    if "epoch" not in epochs[0]:
        raise TrackingError(f"{results_path} has no epoch column")
    finished_at_ms = int(results_path.stat().st_mtime * 1000)
    return TrainingRun(run_dir, args, epochs, finished_at_ms)


def stop_reason(run: TrainingRun) -> str | None:
    """Why a run has fewer epochs than planned: `patience`, `time budget`, or `interrupted`.

    `results.csv` does not record the reason, so it is inferred the way Ultralytics decides to
    stop: `patience` epochs after the first epoch with the best fitness, or a `time=` budget.
    """
    planned = run.planned_epochs
    if planned is None or len(run.epochs) >= planned:
        return None
    if run.args.get("time"):
        return "time budget"
    patience = run.args.get("patience")
    fitness = [epoch.get(FITNESS_COLUMN) for epoch in run.epochs]
    if isinstance(patience, int) and all(value is not None for value in fitness):
        best_epoch = 1 + max(range(len(fitness)), key=lambda index: fitness[index] or 0.0)
        if len(run.epochs) - best_epoch >= patience:
            return "patience"
    return "interrupted"


def _chunks[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    return (items[start : start + size] for start in range(0, len(items), size))


def backfill_run(client: MlflowClient, experiment_id: str, run: TrainingRun) -> str:
    """Rebuild an MLflow run from a run folder, with the names and steps Ultralytics would use."""
    # results.csv `time` is cumulative training seconds, so the run started that long before
    # results.csv was last written. Both are approximate if the folder was copied, not moved.
    duration_ms = int(run.epochs[-1].get("time", 0.0) * 1000)
    started_at_ms = run.finished_at_ms - duration_ms
    run_id = client.create_run(
        experiment_id,
        start_time=started_at_ms,
        tags={BACKFILLED_TAG: BACKFILL_INCOMPLETE, RUN_DIR_TAG: path_label(run.run_dir)},
        run_name=run.name,
    ).info.run_id
    params = [Param(key, str(value)) for key, value in run.args.items()]
    metrics = [
        Metric(
            metric_name(column),
            value,
            started_at_ms + int(epoch.get("time", 0.0) * 1000),
            # Ultralytics logs epoch N (1-based in results.csv) at step N - 1.
            int(epoch["epoch"]) - 1,
        )
        for epoch in run.epochs
        for column, value in epoch.items()
        if column not in UNLOGGED_RESULT_COLUMNS
    ]
    try:
        for batch in _chunks(params, MAX_PARAMS_PER_BATCH):
            client.log_batch(run_id, params=batch)
        for batch in _chunks(metrics, MAX_METRICS_PER_BATCH):
            client.log_batch(run_id, metrics=batch)
        for path in sorted(run.run_dir.iterdir()):
            if path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
                client.log_artifact(run_id, str(path))
    except BaseException:
        # Also on Ctrl+C: leave the run visibly failed; the incomplete tag makes a rerun replace it.
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        raise
    status = RunStatus.KILLED if stop_reason(run) == "interrupted" else RunStatus.FINISHED
    client.set_tag(run_id, BACKFILLED_TAG, "true")
    client.set_terminated(run_id, RunStatus.to_string(status), end_time=run.finished_at_ms)
    return run_id


def weight_hashes(run: TrainingRun) -> dict[str, str]:
    return {path.stem: file_sha256(path) for path in run.weights}


def recorded_weight_hashes(tags: Mapping[str, str]) -> dict[str, str]:
    return {
        key.removeprefix(WEIGHTS_TAG_PREFIX).removesuffix(".sha256"): value
        for key, value in tags.items()
        if key.startswith(WEIGHTS_TAG_PREFIX) and key.endswith(".sha256")
    }


def find_logged_run(
    client: MlflowClient, experiment_id: str, run: TrainingRun, hashes: Mapping[str, str]
) -> Run | None:
    """The MLflow run that records this folder, trying the most specific evidence first.

    A run tagged with this folder; else one whose recorded weights match (the folder was moved,
    for example into `runs/detect/old/`); else an untagged run of that name, which Ultralytics'
    callback logged live. A same-named run of another folder with other weights is a different
    run, so no match is returned and the folder gets its own run.
    """
    folder = path_label(run.run_dir)
    candidates = runs_named(client, experiment_id, run.name)
    same_folder = [c for c in candidates if c.data.tags.get(RUN_DIR_TAG) == folder]
    moved = [
        c
        for c in candidates
        if RUN_DIR_TAG in c.data.tags
        and c.data.tags[RUN_DIR_TAG] != folder
        and recorded_weight_hashes(c.data.tags)
        and recorded_weight_hashes(c.data.tags) == hashes
    ]
    untagged = [c for c in candidates if RUN_DIR_TAG not in c.data.tags]
    for group in (same_folder, moved, untagged):
        if len(group) > 1:
            ids = ", ".join(found.info.run_id for found in group)
            raise TrackingError(f"MLflow runs {ids} all match {folder}; delete the extra ones")
        if group:
            return group[0]
    return None


def record_run_folder(
    client: MlflowClient, run_id: str, run: TrainingRun, hashes: Mapping[str, str]
) -> None:
    """Tag the run with its folder, epochs, stop reason, and checkpoint hashes (for import-scores)."""
    client.set_tag(run_id, RUN_DIR_TAG, path_label(run.run_dir))
    client.set_tag(run_id, EPOCHS_COMPLETED_TAG, str(len(run.epochs)))
    reason = stop_reason(run)
    if reason is not None:
        client.set_tag(run_id, STOP_REASON_TAG, reason)
    for path in run.weights:
        client.set_tag(run_id, f"{WEIGHTS_TAG_PREFIX}{path.stem}.sha256", hashes[path.stem])
        client.set_tag(run_id, f"{WEIGHTS_TAG_PREFIX}{path.stem}.path", path_label(path))


def import_run(client: MlflowClient, experiment_id: str, run_dir: Path) -> str:
    """Backfill a run folder unless it is already logged, then record its weights; returns a note."""
    run = read_training_run(run_dir)
    hashes = weight_hashes(run)
    logged = find_logged_run(client, experiment_id, run, hashes)
    replaced = logged is not None and logged.data.tags.get(BACKFILLED_TAG) == BACKFILL_INCOMPLETE
    if logged is not None and replaced:
        client.delete_run(logged.info.run_id)
        logged = None
    if logged is None:
        run_id = backfill_run(client, experiment_id, run)
        note = f"backfilled {len(run.epochs)} epochs"
        if replaced:
            note += " (replaced an incomplete import)"
    else:
        run_id, note = logged.info.run_id, "already logged"
    record_run_folder(client, run_id, run, hashes)
    return f"{run.name}: {note}, {len(run.weights)} weight files recorded (run {run_id})"


@dataclass(frozen=True)
class Note:
    text: str
    recorded: bool
    """False when something was left out (no matching run or checkpoint, or a conflict)."""


def search_all_logged_models(client: MlflowClient, experiment_id: str) -> list[LoggedModel]:
    models: list[LoggedModel] = []
    page_token: str | None = None
    while True:
        page = client.search_logged_models(
            [experiment_id], max_results=LOGGED_MODEL_PAGE_SIZE, page_token=page_token
        )
        models.extend(page)
        page_token = page.token
        if not page_token:
            return models


def models_by_sha256(models: Iterable[LoggedModel]) -> dict[str, LoggedModel]:
    return {
        model.tags[MODEL_SHA256_TAG]: model for model in models if MODEL_SHA256_TAG in model.tags
    }


def checkpoint_step(client: MlflowClient, run_id: str, checkpoint: str) -> int:
    """The step (epoch - 1) whose weights a checkpoint holds: `best` or the last epoch."""
    history = client.get_metric_history(run_id, FITNESS_METRIC)
    if not history:
        raise TrackingError(f"run {run_id} has no {FITNESS_METRIC} history")
    if checkpoint != BEST_CHECKPOINT:
        return max(metric.step for metric in history)
    # Ultralytics rewrites best.pt whenever fitness equals the best so far, so ties go to the
    # later epoch.
    return max(history, key=lambda metric: (metric.value, metric.step)).step


def set_model_checkpoint(
    client: MlflowClient, runs: Sequence[Run], run_name: str, stem: str
) -> None:
    """Choose which checkpoint represents a run in the Models tab (default `best`)."""
    named = [found for found in runs if found.info.run_name == run_name]
    if len(named) != 1:
        raise TrackingError(f"{len(named)} MLflow runs are named {run_name}; expected one")
    if f"{WEIGHTS_TAG_PREFIX}{stem}.sha256" not in named[0].data.tags:
        raise TrackingError(f"{run_name} has no {stem}.pt recorded; run import-runs on its folder")
    client.set_tag(named[0].info.run_id, MODEL_CHECKPOINT_TAG, stem)


def log_model(
    client: MlflowClient, experiment_id: str, run: Run, known: dict[str, LoggedModel]
) -> Note:
    """Add the run's chosen checkpoint to the Models tab, with the metrics of its epoch.

    The model records the checkpoint's path and SHA-256; the weights stay in the run folder.
    """
    name, run_id = run.info.run_name, run.info.run_id
    checkpoint = run.data.tags.get(MODEL_CHECKPOINT_TAG, BEST_CHECKPOINT)
    sha256 = run.data.tags.get(f"{WEIGHTS_TAG_PREFIX}{checkpoint}.sha256")
    if sha256 is None:
        return Note(f"{name}: no {checkpoint}.pt recorded; run import-runs on its folder", False)
    existing = known.get(sha256)
    if existing is not None and existing.status == LoggedModelStatus.READY:
        return Note(f"{name}: {checkpoint}.pt is already model {existing.model_id}", True)
    if existing is not None:
        client.delete_logged_model(existing.model_id)  # left over from an interrupted call
    params = {key: run.data.params[key] for key in MODEL_PARAM_KEYS if key in run.data.params}
    model = client.create_logged_model(
        experiment_id,
        name=name,
        source_run_id=run_id,
        tags={
            MODEL_SHA256_TAG: sha256,
            MODEL_PATH_TAG: run.data.tags.get(f"{WEIGHTS_TAG_PREFIX}{checkpoint}.path", ""),
        },
        params=params | {"checkpoint": checkpoint},
        model_type=MODEL_TYPE,
    )
    try:
        step = checkpoint_step(client, run_id, checkpoint)
        # Same key, value, timestamp, and step as the run's own points, so MLflow only links them
        # to the model and the run's history gains no duplicates.
        metrics = [
            Metric(point.key, point.value, point.timestamp, point.step, model_id=model.model_id)
            for key in sorted(run.data.metrics)
            if key.startswith(CHECKPOINT_METRIC_PREFIXES)
            for point in client.get_metric_history(run_id, key)
            if point.step == step
        ]
        for batch in _chunks(metrics, MAX_METRICS_PER_BATCH):
            client.log_batch(run_id, metrics=batch)
    except BaseException:
        client.finalize_logged_model(model.model_id, LoggedModelStatus.FAILED)
        raise
    known[sha256] = client.finalize_logged_model(model.model_id, LoggedModelStatus.READY)
    return Note(f"{name}: added {checkpoint}.pt (epoch {step + 1}) as model {model.model_id}", True)


def mark_current_best(
    client: MlflowClient, runs: Sequence[Run], known: Mapping[str, LoggedModel], run_name: str
) -> str:
    """Tag the model of `run_name` as the current best and remove the tag from every other model."""
    named = [found for found in runs if found.info.run_name == run_name]
    if len(named) != 1:
        raise TrackingError(f"{len(named)} MLflow runs are named {run_name}; expected one")
    tags = named[0].data.tags
    checkpoint = tags.get(MODEL_CHECKPOINT_TAG, BEST_CHECKPOINT)
    best = known.get(tags.get(f"{WEIGHTS_TAG_PREFIX}{checkpoint}.sha256", ""))
    if best is None:
        raise TrackingError(f"{run_name} has no model yet; run log-models without --current-best")
    move_current_best(client, known.values(), best)
    return f"current best: {run_name} {checkpoint}.pt (model {best.model_id})"


def move_current_best(
    client: MlflowClient, models: Iterable[LoggedModel], best: LoggedModel
) -> None:
    for model in models:
        if model.model_id == best.model_id:
            client.set_logged_model_tags(model.model_id, {CURRENT_BEST_TAG: "true"})
        elif CURRENT_BEST_TAG in model.tags:
            client.delete_logged_model_tag(model.model_id, CURRENT_BEST_TAG)


def model_fitness(model: LoggedModel) -> float | None:
    """Validation mAP50-95 at the model's checkpoint epoch."""
    values = [metric.value for metric in model.metrics or [] if metric.key == FITNESS_METRIC]
    return max(values) if values else None


def pick_best_by_validation(
    client: MlflowClient, known: Mapping[str, LoggedModel], new_sha256s: Iterable[str]
) -> Note:
    """Move the current-best tag to a new model that beats it on validation mAP50-95.

    Only models added in this call challenge the current best, so a best chosen by hand
    (`--current-best`) is never handed back to an older model it was preferred over. Challengers
    need the current best's `data` config: another dataset has another validation set, so its
    scores are not comparable and a person decides. Ties keep the current best. The test split
    is never used here.
    """
    models = [model for model in known.values() if model.status == LoggedModelStatus.READY]
    holders = [model for model in models if model.tags.get(CURRENT_BEST_TAG) == "true"]
    if len(holders) != 1:
        return Note("current best: none set yet; choose one with --current-best <run name>", True)
    holder = holders[0]
    new_ids = {known[sha256].model_id for sha256 in new_sha256s if sha256 in known}
    rivals = [
        model
        for model in models
        if (model.model_id == holder.model_id or model.model_id in new_ids)
        and model.params.get("data") == holder.params.get("data")
        and model_fitness(model) is not None
    ]
    if not rivals:
        return Note(f"current best: {holder.name} kept (no validation mAP50-95 to compare)", True)
    challenger = max(
        rivals, key=lambda model: (model_fitness(model), model.model_id == holder.model_id)
    )
    if challenger.model_id == holder.model_id:
        return Note(
            f"current best: {holder.name} kept (val mAP50-95 {model_fitness(holder)})", True
        )
    move_current_best(client, models, challenger)
    return Note(
        f"current best: {challenger.name} (val mAP50-95 {model_fitness(challenger)}) replaces "
        f"{holder.name} ({model_fitness(holder)})",
        True,
    )


@dataclass
class ScoredRuns:
    """Runs and models by checkpoint hash, with their metrics, read once per `import-scores` call."""

    run_by_sha256: dict[str, str]
    metrics: dict[str, dict[str, float]]
    model_by_sha256: dict[str, str] = field(default_factory=dict)
    model_metric_keys: dict[str, set[str]] = field(default_factory=dict)
    scores_logged: set[tuple[str, str]] = field(default_factory=set)
    """(run id, score file name) pairs whose file is already attached in this call."""

    @classmethod
    def read(cls, client: MlflowClient, experiment_id: str) -> "ScoredRuns":
        run_by_sha256: dict[str, str] = {}
        metrics: dict[str, dict[str, float]] = {}
        for found in search_all_runs(client, experiment_id):
            metrics[found.info.run_id] = dict(found.data.metrics)
            for sha256 in recorded_weight_hashes(found.data.tags).values():
                run_by_sha256[sha256] = found.info.run_id
        scored = cls(run_by_sha256, metrics)
        models = search_all_logged_models(client, experiment_id)
        for sha256, model in models_by_sha256(models).items():
            if model.status == LoggedModelStatus.READY:
                scored.model_by_sha256[sha256] = model.model_id
                scored.model_metric_keys[model.model_id] = {m.key for m in model.metrics or []}
        return scored


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TrackingError(f"{where} is not an object")
    return value


def _count(value: object, where: str) -> int:
    if not isinstance(value, int):
        raise TrackingError(f"{where} is not a whole number")
    return value


def score_label(detector: Mapping[str, object], min_confidence: object) -> str:
    """Name the inference settings, e.g. `single1280-merged` from `<run>-<sha12>-single1280-merged`.

    Detector versions written by `objhist_detection.detect` end in those settings; the confidence
    is appended only when scoring used a higher threshold than detection did.
    """
    version, sha256 = str(detector.get("version", "")), str(detector.get("weights_sha256", ""))
    _, separator, settings = version.partition(f"-{sha256[:12]}-")
    label = settings if separator else version
    if min_confidence != detector.get("confidence_threshold"):
        label += f"-conf{min_confidence}"
    return label


def score_metrics(result: Mapping[str, object], prefix: str, where: str) -> dict[str, int]:
    totals = _mapping(result.get("totals"), f"{where} totals")
    per_class = _mapping(result.get("per_class"), f"{where} per_class")
    metrics = {
        f"{prefix}/{field}": _count(totals.get(field), f"{where} {field}") for field in COUNT_FIELDS
    }
    cutlery = dict.fromkeys(COUNT_FIELDS, 0)
    for class_name, counts in per_class.items():
        counts = _mapping(counts, f"{where} {class_name}")
        for field_name in COUNT_FIELDS:
            count = _count(counts.get(field_name), f"{where} {class_name} {field_name}")
            metrics[f"{prefix}/{class_name}/{field_name}"] = count
            if class_name in CUTLERY_CLASSES:
                cutlery[field_name] += count
    metrics |= {f"{prefix}/cutlery/{name}": count for name, count in cutlery.items()}
    return metrics


def _created_at_ms(record: Mapping[str, object], score_path: Path) -> int:
    created_at = record.get("created_at")
    if not isinstance(created_at, str):
        raise TrackingError(f"{score_path} has no created_at time")
    try:
        return int(datetime.fromisoformat(created_at).timestamp() * 1000)
    except ValueError as error:
        raise TrackingError(f"{score_path} created_at {created_at!r} is not ISO 8601") from error


def _logged_timestamp_ms(client: MlflowClient, run_id: str, key: str, value: float) -> int:
    """When the run logged this score, so linking it to a model reuses the run's own point."""
    points = [p for p in client.get_metric_history(run_id, key) if p.step == 0 and p.value == value]
    if not points:
        raise TrackingError(f"run {run_id} has no {key} = {value} to link to its model")
    return points[0].timestamp


def import_scores(client: MlflowClient, scored: ScoredRuns, score_path: Path) -> list[Note]:
    """Log each result in a score file to the run, and the model, whose weights it scored."""
    record = _mapping(json.loads(score_path.read_text(encoding="utf-8")), str(score_path))
    dataset, split = record.get("dataset"), record.get("split")
    results = record.get("results")
    if not isinstance(dataset, str) or not isinstance(split, str) or not isinstance(results, list):
        raise TrackingError(f"{score_path} is not a score_detections --json-out file")
    timestamp_ms = _created_at_ms(record, score_path)
    dataset_name = Path(dataset.replace("\\", "/")).name
    notes: list[Note] = []
    for index, result in enumerate(results):
        where = f"{score_path} result {index}"
        result = _mapping(result, where)
        detector = _mapping(result.get("detector"), f"{where} detector")
        sha256 = str(detector.get("weights_sha256", ""))
        version = detector.get("version")
        run_id = scored.run_by_sha256.get(sha256)
        if run_id is None:
            text = f"{version}: no MLflow run has weights {sha256[:12]}; import its run first"
            notes.append(Note(text, recorded=False))
            continue
        prefix = (
            f"score/{dataset_name}/{split}/{score_label(detector, result.get('min_confidence'))}"
        )
        metrics = score_metrics(result, prefix, where)
        logged = scored.metrics[run_id]
        on_run = all(logged.get(key) == value for key, value in metrics.items())
        if not on_run and any(key in logged for key in metrics):
            text = (
                f"{version}: {prefix} is already logged with other counts; not changed. "
                "Score relabeled data under a new dataset name"
            )
            notes.append(Note(text, recorded=False))
            continue
        model_id = scored.model_by_sha256.get(sha256)
        on_model = model_id is None or set(metrics) <= scored.model_metric_keys[model_id]
        if on_run and on_model:
            notes.append(Note(f"{version}: {prefix} already logged", recorded=True))
            continue
        batch = [
            Metric(
                key,
                value,
                _logged_timestamp_ms(client, run_id, key, value) if on_run else timestamp_ms,
                0,
                model_id=model_id,
            )
            for key, value in metrics.items()
        ]
        for chunk in _chunks(batch, MAX_METRICS_PER_BATCH):
            client.log_batch(run_id, metrics=chunk)
        logged.update(metrics)
        if model_id is not None:
            scored.model_metric_keys[model_id] |= set(metrics)
        if not on_run and (run_id, score_path.name) not in scored.scores_logged:
            client.log_artifact(run_id, str(score_path), SCORES_ARTIFACT_DIR)
            scored.scores_logged.add((run_id, score_path.name))
        action = "linked to its model" if on_run else "logged"
        notes.append(Note(f"{version}: {action} {prefix} (run {run_id})", recorded=True))
    return notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help=f"MLflow store (default: $MLFLOW_TRACKING_URI or {DEFAULT_TRACKING_URI})",
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
        help=f"experiment name (default: $MLFLOW_EXPERIMENT_NAME or {DEFAULT_EXPERIMENT_NAME})",
    )
    steps = parser.add_subparsers(dest="step", required=True)
    init = steps.add_parser("init", help="create the store and the experiment")
    init.add_argument(
        "--new-run-dir",
        type=Path,
        help="fail if this training run folder, or an MLflow run of its name, already exists",
    )
    runs = steps.add_parser("import-runs", help="backfill run folders and record their weights")
    runs.add_argument("paths", nargs="+", type=Path, help="run folders or folders of run folders")
    models = steps.add_parser("log-models", help="add each run's checkpoint to the Models tab")
    models.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        type=_run_checkpoint,
        metavar="RUN=STEM",
        help="represent RUN by STEM.pt instead of best.pt from now on (repeatable)",
    )
    best = models.add_mutually_exclusive_group()
    best.add_argument(
        "--current-best", metavar="RUN", help="tag RUN's model as the current best detector"
    )
    best.add_argument(
        "--auto-best",
        action="store_true",
        help="move the current-best tag to a new model with higher validation mAP50-95",
    )
    scores = steps.add_parser("import-scores", help="attach score_detections results to runs")
    scores.add_argument("paths", nargs="+", type=Path, help="score_detections --json-out files")
    return parser


def _run_checkpoint(value: str) -> tuple[str, str]:
    run_name, separator, stem = value.partition("=")
    if not separator or not run_name or not stem:
        raise argparse.ArgumentTypeError(f"expected RUN=STEM, for example my-run=last; got {value}")
    return run_name, stem


def run_log_models(
    client: MlflowClient,
    experiment_id: str,
    checkpoints: Sequence[tuple[str, str]],
    current_best: str | None,
    auto_best: bool = False,
) -> list[Note]:
    for run_name, stem in checkpoints:
        set_model_checkpoint(client, search_all_runs(client, experiment_id), run_name, stem)
    runs = sorted(search_all_runs(client, experiment_id), key=lambda found: found.info.start_time)
    known = models_by_sha256(search_all_logged_models(client, experiment_id))
    existing = {
        sha256 for sha256, model in known.items() if model.status == LoggedModelStatus.READY
    }
    notes = [log_model(client, experiment_id, found, known) for found in runs]
    # Read again so every model carries its current tags, params, and metrics.
    known = models_by_sha256(search_all_logged_models(client, experiment_id))
    if current_best is not None:
        notes.append(Note(mark_current_best(client, runs, known, current_best), True))
    elif auto_best:
        notes.append(pick_best_by_validation(client, known, set(known) - existing))
    return notes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    try:
        experiment_id = ensure_experiment(client, args.tracking_uri, args.experiment)
        if args.step == "init":
            if args.new_run_dir is not None:
                check_new_run_dir(client, experiment_id, args.new_run_dir)
            print(f"experiment {args.experiment} ({experiment_id}) in {args.tracking_uri}")
            return 0
        if args.step == "import-runs":
            for run_dir in find_run_dirs(args.paths):
                print(import_run(client, experiment_id, run_dir))
            return 0
        if args.step == "log-models":
            notes = run_log_models(
                client, experiment_id, args.checkpoint, args.current_best, args.auto_best
            )
        else:
            scored = ScoredRuns.read(client, experiment_id)
            notes = [note for path in args.paths for note in import_scores(client, scored, path)]
    except (TrackingError, MlflowException, OSError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for note in notes:
        print(note.text if note.recorded else f"not logged: {note.text}")
    unrecorded = sum(not note.recorded for note in notes)
    if unrecorded:
        print(f"error: {unrecorded} items were not logged; see above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
