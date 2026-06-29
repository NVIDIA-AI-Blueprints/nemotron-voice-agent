"""Skill governance tests — validates skill version compatibility with the repo."""

# ruff: noqa: D103

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import yaml

REPO_ROOT = Path(__file__).parent.parent
SKILLS_DIR = REPO_ROOT / "skills"
SKILL_DIRS = [d for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists()]

# Match a leading ``X.Y.Z`` and ignore any PEP 440 / semver pre-release suffix
# (e.g. ``2.1.0rc1``, ``2.0.0-rc1``, ``2.1.0.dev0``) so a release-candidate version
# in pyproject.toml does not break the skill-vs-repo governance checks.
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


def _version_parts(ver: str) -> tuple[int, int] | None:
    """Return ``(major, minor)`` ints from a version starting with ``X.Y.Z``, or ``None`` if malformed."""
    match = _SEMVER_RE.match(ver.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _parse_frontmatter(skill_dir: Path) -> dict:
    text = (skill_dir / "SKILL.md").read_text()
    if not text.startswith("---"):
        return {}
    end = text.index("---", 3)
    return yaml.safe_load(text[3:end])


def _repo_version() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def pytest_generate_tests(metafunc):
    if "skill_dir" in metafunc.fixturenames:
        metafunc.parametrize("skill_dir", SKILL_DIRS, ids=[d.name for d in SKILL_DIRS])


def test_skill_version_major_matches_repo(skill_dir):
    fm = _parse_frontmatter(skill_dir)
    skill_ver = str(fm.get("version", ""))
    repo_ver = _repo_version()

    assert skill_ver, f"{skill_dir.name}: no version field in SKILL.md frontmatter"

    skill_parts = _version_parts(skill_ver)
    assert skill_parts is not None, f"{skill_dir.name}: version '{skill_ver}' must start with X.Y.Z (major.minor.patch)"
    repo_parts = _version_parts(repo_ver)
    assert repo_parts is not None, f"pyproject.toml version '{repo_ver}' must start with X.Y.Z (major.minor.patch)"

    skill_major, repo_major = skill_parts[0], repo_parts[0]

    assert skill_major == repo_major, (
        f"{skill_dir.name}: skill major version ({skill_major}) != "
        f"repo major version ({repo_major}) from pyproject.toml. "
        f"Update skill version to match '{repo_ver}'."
    )


def test_skill_minor_not_ahead_of_repo(skill_dir):
    fm = _parse_frontmatter(skill_dir)
    skill_ver = str(fm.get("version", "0.0.0"))
    repo_ver = _repo_version()

    skill_parts = _version_parts(skill_ver)
    assert skill_parts is not None, f"{skill_dir.name}: version '{skill_ver}' must start with X.Y.Z (major.minor.patch)"
    repo_parts = _version_parts(repo_ver)
    assert repo_parts is not None, f"pyproject.toml version '{repo_ver}' must start with X.Y.Z (major.minor.patch)"

    skill_minor, repo_minor = skill_parts[1], repo_parts[1]

    assert skill_minor <= repo_minor, (
        f"{skill_dir.name}: skill minor version ({skill_minor}) is ahead of "
        f"repo minor version ({repo_minor}). Skills must not be newer than the Blueprint."
    )
