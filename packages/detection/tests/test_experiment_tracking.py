import json
import shutil
from pathlib import Path

import pytest
from mlflow.entities import LoggedModelStatus
from mlflow.tracking import MlflowClient

from objhist_detection import experiment_tracking
from objhist_detection.annotation import file_sha256
from objhist_detection.experiment_tracking import (
    BACKFILL_INCOMPLETE,
    BACKFILLED_TAG,
    CURRENT_BEST_TAG,
    EXPERIMENT_KIND_TAG,
    MODEL_SHA256_TAG,
    MODEL_TRAINING_KIND,
    RUN_DIR_TAG,
    STOP_REASON_TAG,
    ScoredRuns,
    TrackingError,
    check_new_run_dir,
    ensure_experiment,
    find_run_dirs,
    import_run,
    import_scores,
    main,
    metric_name,
    path_label,
    read_training_run,
    run_log_models,
    score_label,
    stop_reason,
)

RESULTS_HEADER = "epoch,time,train/box_loss,metrics/mAP50-95(B),lr/pg0"


@pytest.fixture
def store(tmp_path: Path) -> tuple[MlflowClient, str, str]:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow' / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    return client, tracking_uri, ensure_experiment(client, tracking_uri, "test-detector")


def make_run_dir(
    parent: Path,
    name: str,
    fitness: tuple[float, ...] = (0.6, 0.7),
    epochs_planned: int | None = None,
    patience: int = 100,
    weights: bytes = b"w",
) -> Path:
    run_dir = parent / name
    (run_dir / "weights").mkdir(parents=True)
    planned = len(fitness) if epochs_planned is None else epochs_planned
    (run_dir / "args.yaml").write_text(
        f"model: yolo26s.pt\nepochs: {planned}\npatience: {patience}\ntime: null\n"
        f"imgsz: 1280\nname: {name}\n",
        encoding="utf-8",
    )
    rows = [
        f"{epoch},{epoch * 10.0},{1.0 / epoch},{value},0.001"
        for epoch, value in enumerate(fitness, start=1)
    ]
    (run_dir / "results.csv").write_text(
        "\n".join([RESULTS_HEADER, *rows]) + "\n", encoding="utf-8"
    )
    (run_dir / "results.png").write_bytes(b"png")
    # Synthetic stand-in: hashed only, never loaded as a checkpoint.
    (run_dir / "weights" / "best.pt").write_bytes(weights)
    return run_dir


def write_scores(path: Path, sha256: str, correct: int = 9, created_at: str | None = None) -> Path:
    result = {
        "detector": {
            "version": f"run-a-{sha256[:12]}-single1280-merged",
            "weights_sha256": sha256,
            "confidence_threshold": 0.25,
        },
        "min_confidence": 0.25,
        "totals": {
            "objects": 12,
            "correct": correct,
            "wrong_class": 1,
            "missed": 2,
            "false_boxes": 0,
        },
        "per_class": {
            name: {"objects": 4, "correct": 3, "wrong_class": 0, "missed": 1, "false_boxes": 0}
            for name in ("cup", "fork", "spoon")
        },
    }
    record = {
        "created_at": created_at or "2026-10-13T02:22:43+00:00",
        "dataset": "data\\processed\\counter-2026-10-12",
        "split": "test",
        "results": [result],
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def sha(run_dir: Path) -> str:
    return file_sha256(run_dir / "weights" / "best.pt")


def test_metric_names_match_the_ultralytics_callback() -> None:
    assert metric_name(" metrics/mAP50-95(B)") == "metrics/mAP50-95B"
    assert metric_name("train/box_loss") == "train/box_loss"


def test_sqlite_store_keeps_artifacts_beside_the_database(store, tmp_path: Path) -> None:
    client, _, experiment_id = store

    location = client.get_experiment(experiment_id).artifact_location

    assert location == (tmp_path / "mlflow" / "artifacts").resolve().as_uri()


def test_experiments_are_marked_as_model_training_for_the_ui(store) -> None:
    client, tracking_uri, experiment_id = store
    untagged = client.create_experiment("made-by-ultralytics")

    ensure_experiment(client, tracking_uri, "made-by-ultralytics")

    assert client.get_experiment(experiment_id).tags[EXPERIMENT_KIND_TAG] == MODEL_TRAINING_KIND
    assert client.get_experiment(untagged).tags[EXPERIMENT_KIND_TAG] == MODEL_TRAINING_KIND


def test_deleted_experiment_is_an_actionable_error(store) -> None:
    client, tracking_uri, experiment_id = store
    client.delete_experiment(experiment_id)

    with pytest.raises(TrackingError, match="is deleted"):
        ensure_experiment(client, tracking_uri, "test-detector")


def test_backfill_logs_params_epoch_metrics_artifacts_and_weights(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path / "runs", "run-a")

    import_run(client, experiment_id, run_dir)

    (found,) = client.search_runs([experiment_id])
    run_id = found.info.run_id
    assert found.info.run_name == "run-a"
    assert found.info.status == "FINISHED"
    assert found.data.params["imgsz"] == "1280"
    assert found.data.tags[BACKFILLED_TAG] == "true"
    assert found.data.tags[RUN_DIR_TAG] == path_label(run_dir)
    assert found.data.tags["objhist.weights.best.sha256"] == sha(run_dir)
    history = client.get_metric_history(run_id, "metrics/mAP50-95B")
    assert [(m.step, m.value) for m in history] == [(0, 0.6), (1, 0.7)]
    assert "epoch" not in found.data.metrics and "time" not in found.data.metrics
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert artifacts == {"args.yaml", "results.csv", "results.png"}


def test_reimport_does_not_duplicate_and_tags_an_already_logged_run(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path / "runs", "run-a")
    # A run Ultralytics' callback logged live has the folder's name but no folder tag.
    live_id = client.create_run(experiment_id, run_name="run-a").info.run_id

    import_run(client, experiment_id, run_dir)
    import_run(client, experiment_id, run_dir)

    (found,) = client.search_runs([experiment_id])
    assert found.info.run_id == live_id
    assert BACKFILLED_TAG not in found.data.tags
    assert found.data.tags["objhist.weights.best.sha256"] == sha(run_dir)


def test_moved_folder_keeps_its_run_and_updates_the_recorded_path(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path / "detect", "run-a")
    import_run(client, experiment_id, run_dir)
    moved = tmp_path / "detect" / "old" / "run-a"
    moved.parent.mkdir()
    shutil.move(run_dir, moved)

    import_run(client, experiment_id, moved)

    (found,) = client.search_runs([experiment_id])
    assert found.data.tags[RUN_DIR_TAG] == path_label(moved)


def test_same_named_folders_with_other_weights_get_separate_runs(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    first = make_run_dir(tmp_path / "detect", "run-a", weights=b"first")
    second = make_run_dir(tmp_path / "detect" / "old", "run-a", weights=b"second")

    import_run(client, experiment_id, first)
    import_run(client, experiment_id, second)
    import_run(client, experiment_id, first)

    runs = client.search_runs([experiment_id])
    recorded = {r.data.tags[RUN_DIR_TAG]: r.data.tags["objhist.weights.best.sha256"] for r in runs}
    assert recorded == {path_label(first): sha(first), path_label(second): sha(second)}


def test_failed_backfill_is_marked_failed_and_replaced_on_rerun(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path / "runs", "run-a")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(client, "log_artifact", fail)
        with pytest.raises(OSError, match="disk full"):
            import_run(client, experiment_id, run_dir)
    (failed,) = client.search_runs([experiment_id])
    assert failed.info.status == "FAILED"
    assert failed.data.tags[BACKFILLED_TAG] == BACKFILL_INCOMPLETE

    note = import_run(client, experiment_id, run_dir)

    (found,) = client.search_runs([experiment_id])
    assert "replaced an incomplete import" in note
    assert found.info.run_id != failed.info.run_id
    assert found.info.status == "FINISHED"
    assert found.data.tags[BACKFILLED_TAG] == "true"


@pytest.mark.parametrize(
    ("fitness", "planned", "patience", "reason", "status"),
    [
        ((0.6, 0.7), 2, 100, None, "FINISHED"),
        ((0.6, 0.8, 0.7, 0.7), 10, 2, "patience", "FINISHED"),
        ((0.6, 0.8, 0.7), 10, 2, "interrupted", "KILLED"),
    ],
)
def test_short_runs_get_an_inferred_stop_reason(
    store, tmp_path: Path, fitness, planned, patience, reason, status
) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path, "run", fitness, planned, patience)

    import_run(client, experiment_id, run_dir)

    (found,) = client.search_runs([experiment_id])
    assert stop_reason(read_training_run(run_dir)) == reason
    assert found.data.tags.get(STOP_REASON_TAG) == reason
    assert found.info.status == status


def test_time_budget_counts_as_a_normal_stop(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, "run", (0.6, 0.7), epochs_planned=10)
    args = run_dir / "args.yaml"
    args.write_text(args.read_text(encoding="utf-8").replace("time: null", "time: 0.5"), "utf-8")

    assert stop_reason(read_training_run(run_dir)) == "time budget"


def test_new_run_dir_check_covers_folders_and_names_already_in_mlflow(
    store, tmp_path: Path
) -> None:
    client, _, experiment_id = store
    old = make_run_dir(tmp_path / "detect" / "old", "run-a")
    import_run(client, experiment_id, old)

    check_new_run_dir(client, experiment_id, tmp_path / "detect" / "run-b")
    with pytest.raises(TrackingError, match="already has a run named run-a"):
        check_new_run_dir(client, experiment_id, tmp_path / "detect" / "run-a")
    with pytest.raises(TrackingError, match="exists"):
        check_new_run_dir(client, experiment_id, old)


def test_find_run_dirs_accepts_run_folders_and_their_parents(tmp_path: Path) -> None:
    first = make_run_dir(tmp_path / "detect", "a")
    second = make_run_dir(tmp_path / "detect" / "old", "b")

    assert find_run_dirs([tmp_path / "detect", second]) == [first, second]
    with pytest.raises(TrackingError, match="no run folders"):
        find_run_dirs([tmp_path / "detect" / "a" / "weights"])


def test_run_without_results_is_an_error(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path, "empty")
    (run_dir / "results.csv").unlink()

    with pytest.raises(TrackingError, match="no results.csv"):
        import_run(client, experiment_id, run_dir)


def test_scores_attach_to_the_run_with_matching_weights_once(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path / "runs", "run-a")
    import_run(client, experiment_id, run_dir)
    scores = write_scores(tmp_path / "scores.json", sha(run_dir))
    scored = ScoredRuns.read(client, experiment_id)

    first = import_scores(client, scored, scores)
    second = import_scores(client, ScoredRuns.read(client, experiment_id), scores)

    (found,) = client.search_runs([experiment_id])
    prefix = "score/counter-2026-10-12/test/single1280-merged"
    assert found.data.metrics[f"{prefix}/correct"] == 9
    assert found.data.metrics[f"{prefix}/cutlery/objects"] == 8
    assert found.data.metrics[f"{prefix}/cup/missed"] == 1
    assert len(client.get_metric_history(found.info.run_id, f"{prefix}/correct")) == 1
    assert first[0].recorded and "logged" in first[0].text
    assert second[0].recorded and "already logged" in second[0].text
    assert [a.path for a in client.list_artifacts(found.info.run_id, "scores")] == [
        "scores/scores.json"
    ]


def test_scores_with_conflicting_counts_or_unknown_weights_are_not_recorded(
    store, tmp_path: Path
) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path / "runs", "run-a")
    import_run(client, experiment_id, run_dir)
    scored = ScoredRuns.read(client, experiment_id)
    import_scores(client, scored, write_scores(tmp_path / "a.json", sha(run_dir)))

    conflict = import_scores(client, scored, write_scores(tmp_path / "b.json", sha(run_dir), 5))
    unknown = import_scores(client, scored, write_scores(tmp_path / "c.json", "f" * 64))

    assert not conflict[0].recorded and "other counts" in conflict[0].text
    assert not unknown[0].recorded and "no MLflow run" in unknown[0].text
    metrics = client.search_runs([experiment_id])[0].data.metrics
    assert metrics["score/counter-2026-10-12/test/single1280-merged/correct"] == 9


def test_score_lookup_pages_through_every_run(
    store, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, experiment_id = store
    run_dirs = [make_run_dir(tmp_path, f"run-{i}", weights=bytes([i])) for i in range(3)]
    for run_dir in run_dirs:
        import_run(client, experiment_id, run_dir)
    monkeypatch.setattr(experiment_tracking, "SEARCH_PAGE_SIZE", 2)

    scored = ScoredRuns.read(client, experiment_id)

    assert set(scored.run_by_sha256) == {sha(run_dir) for run_dir in run_dirs}


def test_score_label_names_settings_and_a_raised_confidence() -> None:
    detector = {
        "version": "run-a-0123456789ab-tiled",
        "weights_sha256": "0123456789ab" + "0" * 52,
        "confidence_threshold": 0.25,
    }

    assert score_label(detector, 0.25) == "tiled"
    assert score_label(detector, 0.5) == "tiled-conf0.5"
    assert score_label({"version": "yoloe", "confidence_threshold": 0.1}, 0.1) == "yoloe"


def test_cli_exits_nonzero_when_a_score_is_not_recorded(tmp_path: Path, capsys) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_dir = make_run_dir(tmp_path / "runs", "run-a")
    assert main(["--tracking-uri", uri, "--experiment", "e", "import-runs", str(run_dir)]) == 0
    known = write_scores(tmp_path / "known.json", sha(run_dir))
    unknown = write_scores(tmp_path / "unknown.json", "f" * 64)

    code = main(
        ["--tracking-uri", uri, "--experiment", "e", "import-scores", str(known), str(unknown)]
    )

    output = capsys.readouterr()
    assert code == 1
    assert "not logged: " in output.out and "1 items were not logged" in output.err


@pytest.mark.parametrize(
    ("content", "message"),
    [("[]", "is not an object"), (None, "is not ISO 8601")],
)
def test_cli_reports_unreadable_score_files(tmp_path: Path, capsys, content, message) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    path = tmp_path / "bad.json"
    if content is None:
        write_scores(path, "f" * 64, created_at="yesterday")
    else:
        path.write_text(content, encoding="utf-8")

    assert main(["--tracking-uri", uri, "import-scores", str(path)]) == 1
    assert message in capsys.readouterr().err


def add_last_checkpoint(run_dir: Path) -> None:
    (run_dir / "weights" / "last.pt").write_bytes(b"last " + run_dir.name.encode())


def models_by_name(client: MlflowClient, experiment_id: str) -> dict[str, object]:
    return {m.name: m for m in client.search_logged_models([experiment_id])}


def test_log_models_adds_each_run_with_its_checkpoint_epoch_metrics_once(
    store, tmp_path: Path
) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path, "run-a", fitness=(0.6, 0.8, 0.8, 0.7))
    import_run(client, experiment_id, run_dir)
    run_id = client.search_runs([experiment_id])[0].info.run_id

    first = run_log_models(client, experiment_id, [], None)
    second = run_log_models(client, experiment_id, [], None)

    (model,) = client.search_logged_models([experiment_id])
    assert model.name == "run-a" and model.source_run_id == run_id
    assert model.status == LoggedModelStatus.READY
    assert model.tags[MODEL_SHA256_TAG] == sha(run_dir)
    assert model.params["checkpoint"] == "best" and model.params["imgsz"] == "1280"
    # A tie at 0.8 goes to the later epoch, as Ultralytics rewrites best.pt on ties.
    assert {(m.key, m.step, m.value) for m in model.metrics} == {("metrics/mAP50-95B", 2, 0.8)}
    assert len(client.get_metric_history(run_id, "metrics/mAP50-95B")) == 4
    assert "added best.pt (epoch 3)" in first[0].text
    assert "already model" in second[0].text


def test_checkpoint_choice_persists_and_current_best_moves(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    for name in ("run-a", "run-b"):
        run_dir = make_run_dir(tmp_path, name, weights=name.encode())
        add_last_checkpoint(run_dir)
        import_run(client, experiment_id, run_dir)

    run_log_models(client, experiment_id, [("run-a", "last")], "run-a")
    run_log_models(client, experiment_id, [], "run-b")

    models = models_by_name(client, experiment_id)
    assert models["run-a"].params["checkpoint"] == "last"
    assert {m.step for m in models["run-a"].metrics} == {1}
    assert CURRENT_BEST_TAG not in models["run-a"].tags
    assert models["run-b"].tags[CURRENT_BEST_TAG] == "true"
    assert len(models) == 2
    with pytest.raises(TrackingError, match="no nope.pt"):
        run_log_models(client, experiment_id, [("run-a", "nope")], None)


def test_scores_logged_before_the_model_are_linked_to_it_without_duplicates(
    store, tmp_path: Path
) -> None:
    client, _, experiment_id = store
    run_dir = make_run_dir(tmp_path, "run-a")
    import_run(client, experiment_id, run_dir)
    scores = write_scores(tmp_path / "scores.json", sha(run_dir))
    import_scores(client, ScoredRuns.read(client, experiment_id), scores)
    run_log_models(client, experiment_id, [], None)

    notes = import_scores(client, ScoredRuns.read(client, experiment_id), scores)
    again = import_scores(client, ScoredRuns.read(client, experiment_id), scores)

    key = "score/counter-2026-10-12/test/single1280-merged/cutlery/correct"
    (model,) = client.search_logged_models([experiment_id])
    run_id = client.search_runs([experiment_id])[0].info.run_id
    assert any(m.key == key and m.value == 6 for m in model.metrics)
    assert len(client.get_metric_history(run_id, key)) == 1
    assert "linked to its model" in notes[0].text and "already logged" in again[0].text


def import_with_data(client, experiment_id, run_dir: Path, data: str) -> None:
    args = run_dir / "args.yaml"
    args.write_text(args.read_text(encoding="utf-8") + f"data: {data}\n", encoding="utf-8")
    import_run(client, experiment_id, run_dir)


def test_auto_best_moves_only_to_a_new_model_with_higher_validation(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    # "higher" already beats "holder" on validation, but a person chose "holder" over it.
    for name, fitness in (("holder", (0.6, 0.8)), ("higher", (0.6, 0.85))):
        run_dir = make_run_dir(tmp_path, name, fitness, weights=name.encode())
        import_with_data(client, experiment_id, run_dir, "same/data.yaml")
    run_log_models(client, experiment_id, [], "holder")

    kept = run_log_models(client, experiment_id, [], None, auto_best=True)
    for name, fitness in (("tied", (0.8, 0.6)), ("lower", (0.6, 0.7))):
        run_dir = make_run_dir(tmp_path, name, fitness, weights=name.encode())
        import_with_data(client, experiment_id, run_dir, "same/data.yaml")
    still_kept = run_log_models(client, experiment_id, [], None, auto_best=True)
    better = make_run_dir(tmp_path, "better", (0.6, 0.9), weights=b"better")
    import_with_data(client, experiment_id, better, "same/data.yaml")
    moved = run_log_models(client, experiment_id, [], None, auto_best=True)

    models = models_by_name(client, experiment_id)
    assert "holder kept" in kept[-1].text and "holder kept" in still_kept[-1].text
    assert "better" in moved[-1].text and "replaces holder" in moved[-1].text
    assert [name for name, m in models.items() if CURRENT_BEST_TAG in m.tags] == ["better"]


def test_auto_best_ignores_models_trained_on_other_data(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    holder = make_run_dir(tmp_path, "holder", (0.6, 0.7), weights=b"holder")
    import_with_data(client, experiment_id, holder, "same/data.yaml")
    run_log_models(client, experiment_id, [], "holder")
    other = make_run_dir(tmp_path, "other", (0.6, 0.99), weights=b"other")
    import_with_data(client, experiment_id, other, "other/data.yaml")

    notes = run_log_models(client, experiment_id, [], None, auto_best=True)

    assert "holder kept" in notes[-1].text
    assert CURRENT_BEST_TAG not in models_by_name(client, experiment_id)["other"].tags


def test_auto_best_without_a_current_best_changes_nothing(store, tmp_path: Path) -> None:
    client, _, experiment_id = store
    import_run(client, experiment_id, make_run_dir(tmp_path, "run-a"))

    notes = run_log_models(client, experiment_id, [], None, auto_best=True)

    assert "none set yet" in notes[-1].text and notes[-1].recorded
    assert CURRENT_BEST_TAG not in models_by_name(client, experiment_id)["run-a"].tags
