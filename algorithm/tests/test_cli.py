from pathlib import Path

import pytest

from momentmaker_cv.cli import build_parser, main


def test_parser_requires_output_directory() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["photo.jpg"])


def test_cli_reports_missing_input_without_loading_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([str(tmp_path / "missing.jpg"), "-o", str(tmp_path / "out")])

    assert exit_code == 2
    assert '"status": "invalid_input"' in capsys.readouterr().out
