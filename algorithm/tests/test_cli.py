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


def test_parser_can_keep_partial_people() -> None:
    args = build_parser().parse_args(["photo.jpg", "--output", "output", "--keep-partial-people"])

    assert args.keep_partial_people is True


def test_parser_accepts_soft_alpha_mode() -> None:
    args = build_parser().parse_args(["photo.jpg", "--output", "output", "--alpha-mode", "soft"])

    assert args.alpha_mode == "soft"


def test_parser_uses_soft_alpha_by_default() -> None:
    args = build_parser().parse_args(["photo.jpg", "--output", "output"])

    assert args.alpha_mode == "soft"
