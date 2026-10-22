"""Keep the supported Python versions agreeing with the ``pyproject.toml`` classifiers."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib

import pytest

REPO_ROOT = Path(__file__).parent.parent
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PINNED_PYTHON_PATH = REPO_ROOT / ".python-version"

# The sdist ships the tests but not the CI workflow or the pin.
if not (CI_WORKFLOW_PATH.exists() and PINNED_PYTHON_PATH.exists()):
    pytest.skip("needs a repository checkout", allow_module_level=True)

PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
CI_WORKFLOW = CI_WORKFLOW_PATH.read_text()
PINNED_PYTHON = PINNED_PYTHON_PATH.read_text()

_CLASSIFIER_PATTERN = re.compile(r"Programming Language :: Python :: (\d+\.\d+)")
_MATRIX_PATTERN = re.compile(r"^\s*python-version:\s*\[(?P<versions>[^\]]*)\]", re.MULTILINE)
_LOWEST_PATTERN = re.compile(r"--resolution lowest-direct --python (\d+\.\d+)")

type Version = tuple[int, int]


def _parse(version: str) -> Version:
    """Parse a ``major.minor`` string."""
    major, minor = version.strip().strip("\"'").split(".")
    return int(major), int(minor)


def _supported() -> list[Version]:
    """Return the classified Python versions, oldest first."""
    classifiers: list[str] = PYPROJECT["project"]["classifiers"]
    versions = sorted(
        _parse(match.group(1))
        for classifier in classifiers
        if (match := _CLASSIFIER_PATTERN.fullmatch(classifier))
    )
    assert versions, "no Python version classifiers found in pyproject.toml"
    return versions


def test_classifiers_are_contiguous() -> None:
    supported = _supported()
    (major, oldest), (_, newest) = supported[0], supported[-1]

    assert supported == [(major, minor) for minor in range(oldest, newest + 1)]


def test_ci_matrix_tests_every_supported_version() -> None:
    match = _MATRIX_PATTERN.search(CI_WORKFLOW)
    assert match is not None, "no python-version matrix found in the CI workflow"

    matrix = {_parse(version) for version in match.group("versions").split(",")}

    assert matrix == set(_supported())


def test_requires_python_is_the_oldest_supported_version() -> None:
    major, minor = _supported()[0]

    assert PYPROJECT["project"]["requires-python"] == f">={major}.{minor}"


def test_lowest_bound_job_runs_the_oldest_supported_version() -> None:
    versions = _LOWEST_PATTERN.findall(CI_WORKFLOW)
    assert versions, "no lowest-direct sync found in the CI workflow"

    assert {_parse(version) for version in versions} == {_supported()[0]}


def test_pyrefly_checks_against_the_oldest_supported_version() -> None:
    assert _parse(PYPROJECT["tool"]["pyrefly"]["python-version"]) == _supported()[0]


def test_pinned_development_python_is_supported() -> None:
    assert _parse(PINNED_PYTHON) in _supported()
