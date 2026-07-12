from ml_homework import DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR


def test_project_directories_are_anchored_at_project_root() -> None:
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DATA_DIR == DATA_DIR / "raw"
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"


def test_required_project_directories_exist() -> None:
    assert RAW_DATA_DIR.is_dir()
    assert PROCESSED_DATA_DIR.is_dir()
