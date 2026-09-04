import re
import tomllib
from pathlib import Path

ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
PIN_PATTERN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\s]+)$")


def _requirement_name(requirement: str) -> str:
    return re.split(r"[<>=!~\s]", requirement, maxsplit=1)[0].lower()


def test_demo_constraints_pin_every_direct_dependency() -> None:
    with (ALGORITHM_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    declared = {
        _requirement_name(requirement) for requirement in project["project"]["dependencies"]
    }
    for requirements in project["project"]["optional-dependencies"].values():
        declared.update(_requirement_name(requirement) for requirement in requirements)

    pinned: dict[str, str] = {}
    for raw_line in (
        (ALGORITHM_ROOT / "constraints-demo.txt").read_text(encoding="utf-8").splitlines()
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN_PATTERN.fullmatch(line)
        assert match is not None, f"demo constraint must be an exact pin: {line}"
        name = match.group("name").lower()
        assert name not in pinned, f"duplicate demo constraint: {name}"
        pinned[name] = match.group("version")

    assert set(pinned) == declared
