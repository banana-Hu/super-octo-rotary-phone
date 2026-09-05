import numpy as np

from momentmaker_cv.mask_processing import ProcessedMask
from momentmaker_cv.subject_processing import build_subject_masks, group_nearby_people


def _person(box: tuple[int, int, int, int], size: tuple[int, int] = (100, 80)) -> ProcessedMask:
    alpha = np.zeros((size[1], size[0]), dtype=np.uint8)
    left, top, right, bottom = box
    alpha[top:bottom, left:right] = 255
    return ProcessedMask(0.9, box, box, alpha, int((alpha > 0).sum()))


def test_groups_close_people_but_keeps_distant_people_separate() -> None:
    masks = [_person((5, 10, 25, 70)), _person((28, 12, 48, 70)), _person((75, 8, 95, 70))]

    assert group_nearby_people(masks) == [(1, 2), (3,)]


def test_grouping_is_transitive() -> None:
    masks = [_person((5, 10, 25, 70)), _person((28, 10, 48, 70)), _person((51, 10, 71, 70))]

    assert group_nearby_people(masks) == [(1, 2, 3)]


def test_does_not_group_small_background_person_with_large_foreground_person() -> None:
    foreground_person = _person((5, 5, 75, 78))
    background_person = _person((65, 30, 78, 65))

    assert group_nearby_people([foreground_person, background_person]) == [(1,), (2,)]


def test_foreground_keeps_connected_object_and_rejects_remote_region() -> None:
    person = _person((10, 15, 30, 65))
    foreground = np.zeros((80, 100), dtype=np.float32)
    foreground[15:65, 10:30] = 1.0
    foreground[35:55, 28:55] = 0.9  # held object touches the person
    foreground[10:30, 75:95] = 1.0  # unrelated remote foreground

    subjects = build_subject_masks((100, 80), [person], foreground)

    assert len(subjects) == 1
    assert subjects[0].mode == "foreground"
    assert subjects[0].alpha[45, 50] > 0
    assert subjects[0].alpha[20, 80] == 0


def test_people_only_subject_unions_group_members() -> None:
    people = [_person((5, 10, 25, 70)), _person((28, 12, 48, 70))]

    subject = build_subject_masks((100, 80), people)[0]

    assert subject.member_person_ids == (1, 2)
    assert subject.mode == "people"
    assert subject.alpha[20, 10] == 255
    assert subject.alpha[20, 35] == 255


def test_foreground_subject_excludes_people_from_other_groups() -> None:
    people = [_person((10, 10, 30, 70)), _person((60, 10, 80, 70))]
    foreground = np.ones((80, 100), dtype=np.float32)

    subjects = build_subject_masks((100, 80), people, foreground)

    assert len(subjects) == 2
    assert subjects[0].alpha[20, 20] == 255
    assert subjects[0].alpha[20, 70] == 0
    assert subjects[1].alpha[20, 70] == 255
    assert subjects[1].alpha[20, 20] == 0


def test_rejects_foreground_mask_with_wrong_shape() -> None:
    with np.testing.assert_raises_regex(ValueError, "foreground mask shape"):
        build_subject_masks((100, 80), [_person((10, 10, 30, 70))], np.ones((20, 20)))
