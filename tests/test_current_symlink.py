"""Tests for the training_output/current symlink helpers."""

import os
import time

import pytest

from openadapt_ml.training.trainer import (
    get_current_job_directory,
    update_current_symlink_to_latest,
)


def _can_symlink(tmp_path) -> bool:
    """Symlinks need Developer Mode or admin rights on Windows."""
    try:
        (tmp_path / "_probe").symlink_to(tmp_path)
    except OSError:
        return False
    (tmp_path / "_probe").unlink()
    return True


@pytest.fixture
def base(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not permitted on this system")
    return tmp_path / "training_output"


def _make_run(base, name: str, mtime: float, with_log: bool = True):
    run = base / name
    run.mkdir(parents=True)
    target = run / "training_log.json" if with_log else run
    if with_log:
        target.write_text("{}")
    os.utime(target, (mtime, mtime))
    return run


def test_points_current_to_most_recent_run(base):
    now = time.time()
    _make_run(base, "job_old", now - 100)
    newest = _make_run(base, "job_new", now)

    assert update_current_symlink_to_latest(base) == newest
    assert (base / "current").is_symlink()
    assert get_current_job_directory(base).resolve() == newest.resolve()


def test_prefers_training_log_mtime_over_directory_mtime(base):
    now = time.time()
    logged = _make_run(base, "job_logged", now)
    _make_run(base, "job_empty", now - 1000, with_log=False)

    assert update_current_symlink_to_latest(base) == logged


def test_repoints_existing_symlink_and_ignores_hidden_dirs(base):
    now = time.time()
    _make_run(base, "job_a", now - 100)
    update_current_symlink_to_latest(base)
    newest = _make_run(base, "job_b", now)
    _make_run(base, ".cache", now + 100)

    assert update_current_symlink_to_latest(base) == newest
    assert get_current_job_directory(base).resolve() == newest.resolve()


def test_returns_none_without_runs(tmp_path):
    assert update_current_symlink_to_latest(tmp_path / "missing") is None
    (tmp_path / "empty").mkdir()
    assert update_current_symlink_to_latest(tmp_path / "empty") is None
