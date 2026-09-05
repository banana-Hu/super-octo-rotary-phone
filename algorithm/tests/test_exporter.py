from pathlib import Path

import pytest

from momentmaker_cv.contracts import PersonCutout, SubjectCutout
from momentmaker_cv.exporter import cleanup_stale_people, cleanup_stale_subjects


def _person(path: Path, person_id: int = 1) -> PersonCutout:
    return PersonCutout(
        person_id=person_id,
        confidence=0.9,
        source_box=(0, 0, 10, 20),
        output_path=path,
        pixel_area=100,
    )


def test_cleanup_removes_only_obsolete_algorithm_artifacts(tmp_path: Path) -> None:
    people_dir = tmp_path / "people"
    people_dir.mkdir()
    current = people_dir / "person_01.png"
    stale = people_dir / "person_04.png"
    unrelated = people_dir / "notes.txt"
    similar_name = people_dir / "person_final.png"
    for path in (current, stale, unrelated, similar_name):
        path.write_bytes(b"content")

    warnings = cleanup_stale_people(tmp_path, (_person(current),))

    assert warnings == ()
    assert current.exists()
    assert not stale.exists()
    assert unrelated.exists()
    assert similar_name.exists()


def test_cleanup_removes_all_old_people_when_current_result_is_empty(tmp_path: Path) -> None:
    people_dir = tmp_path / "people"
    people_dir.mkdir()
    stale_paths = [people_dir / "person_01.png", people_dir / "person_100.png"]
    for path in stale_paths:
        path.write_bytes(b"content")

    warnings = cleanup_stale_people(tmp_path, ())

    assert warnings == ()
    assert all(not path.exists() for path in stale_paths)


def test_cleanup_reports_delete_failure_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    people_dir = tmp_path / "people"
    people_dir.mkdir()
    stale = people_dir / "person_02.png"
    stale.write_bytes(b"content")
    original_unlink = Path.unlink

    def fail_stale(path: Path, *args: object, **kwargs: object) -> None:
        if path == stale:
            raise PermissionError("file is locked")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_stale)

    warnings = cleanup_stale_people(tmp_path, ())

    assert stale.exists()
    assert len(warnings) == 1
    assert "people/person_02.png" in warnings[0]
    assert "file is locked" in warnings[0]


def test_cleanup_removes_only_obsolete_subject_artifacts(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()
    current = subjects_dir / "subject_01.png"
    stale = subjects_dir / "subject_02.png"
    unrelated = subjects_dir / "layout.json"
    for path in (current, stale, unrelated):
        path.write_bytes(b"content")
    subject = SubjectCutout(1, (1, 2), "people", (0, 0, 10, 20), current, 100)

    warnings = cleanup_stale_subjects(tmp_path, (subject,))

    assert warnings == ()
    assert current.exists()
    assert not stale.exists()
    assert unrelated.exists()
