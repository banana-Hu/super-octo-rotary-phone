from momentmaker_cv import RESULT_SCHEMA_VERSION, CutoutOptions, process_image


def test_package_exposes_backend_facing_api() -> None:
    assert callable(process_image)
    assert CutoutOptions().max_people == 5
    assert RESULT_SCHEMA_VERSION == "1.0"
