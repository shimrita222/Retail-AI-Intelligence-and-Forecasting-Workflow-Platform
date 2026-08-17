from pathlib import Path

from src.services.run_registry import is_real_run_id, list_real_run_dirs


def test_real_run_id_is_recognized():
    assert is_real_run_id("20260814T005931Z-b3cd361a")


def test_fixture_run_id_is_not_recognized():
    # This is exactly the run_id used by the manually-built UI smoke-test
    # fixture; it must never be mistaken for a genuine workflow run.
    assert not is_real_run_id("ui-smoketest")


def test_other_non_conforming_ids_are_rejected():
    for bad_id in ["demo", "20260814", "20260814T005931Z", "20260814T005931Z-tooshort", ""]:
        assert not is_real_run_id(bad_id)


def test_list_real_run_dirs_excludes_fixture(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    real_run = artifacts_dir / "20260814T005931Z-b3cd361a"
    fixture_run = artifacts_dir / "ui-smoketest"
    real_run.mkdir(parents=True)
    fixture_run.mkdir(parents=True)

    discovered = list_real_run_dirs(artifacts_dir)

    assert real_run in discovered
    assert fixture_run not in discovered


def test_list_real_run_dirs_still_finds_legitimate_runs(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    run_a = artifacts_dir / "20260814T004807Z-fe642b4b"
    run_b = artifacts_dir / "20260814T060614Z-7ddf5d40"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    discovered = list_real_run_dirs(artifacts_dir)

    assert set(discovered) == {run_a, run_b}


def test_list_real_run_dirs_on_missing_directory(tmp_path: Path):
    assert list_real_run_dirs(tmp_path / "does_not_exist") == []
