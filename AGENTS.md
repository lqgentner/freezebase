# freezebase Development Setup

freezebase is a lightweight geospatial helpers package: reproducible MGRS grids, rasterio/GDAL-backed COG and VRT I/O, S3 credential handling, and HTTP download tooling. It is the shared IO layer consumed by sibling packages (`s1bursts`, `deep-glacier-mapping`, `glace-catalog`), so changes to its public API are a cross-repository concern, not just a local one.

Treat signatures, return types, exceptions, import paths, and documented behavior as public
API. Preserve compatibility unless a change is explicitly breaking.

## Environment

The project uses uv and supports Python 3.12–3.14. Dependencies are declared in
`pyproject.toml`, and the virtual environment is located at `.venv/`.

```bash
uv sync --all-extras
```

`--all-extras` matters: the `s3` extra is optional at runtime but the test suite imports `freezebase.s3`, so a bare `uv sync` leaves those tests skipped.

## Required checks

```bash
uv run pytest
uv run ruff format
uv run ruff check --fix
uv run ruff format
uv run mypy src/freezebase tests
```

After every code change, run tests, Ruff formatting and linting, and mypy. Rerun affected
checks after fixes. Add or update tests for behavior changes; bug fixes should include a
regression test.

Unit tests are offline. Tests marked `integration` need a live S3-compatible service and are
skipped unless `FREEZEBASE_TEST_S3_*` is set:

```bash
docker run -d --name minio -p 9000:9000 \
  -e MINIO_ROOT_USER=testkey -e MINIO_ROOT_PASSWORD=testsecret \
  minio/minio server /data

FREEZEBASE_TEST_S3_ENDPOINT=http://localhost:9000 \
FREEZEBASE_TEST_S3_KEY=testkey \
FREEZEBASE_TEST_S3_SECRET=testsecret \
  uv run pytest -q -m integration
```

Run integration tests after changing `s3.py` or S3 paths in `raster.py` or `vrt.py`.

### Coverage

PR patch coverage must be at least 90%; the project-wide floor is 80%. Check before opening
a PR:

```bash
uv run pytest -q --cov=freezebase --cov-branch --cov-report=term-missing
```

### Dependency lower bounds

Every dependency's lower bound is the oldest release that installs and works on Python 3.12.
When using a newer dependency API, raise its lower bound with a justification and run:

```bash
uv sync --all-extras --resolution lowest-direct --python 3.12
uv run --no-sync pytest -q
```

Keep `--no-sync`; otherwise `uv run` re-resolves the environment at the latest versions.

## Documentation

Great Docs uses Quarto, which must be installed separately. See
[`user_guide/03-contributing.qmd`](user_guide/03-contributing.qmd) for preview and freeze-cache
details.

```bash
uv sync --all-extras --group docs
uv run great-docs build      # writes great-docs/_site/
uv run great-docs preview    # serves it; does not rebuild on change
```

## Releasing

Versions come from git tags via `hatch-vcs`, so there is no version string to edit. Record
changes in [`CHANGELOG.md`](CHANGELOG.md) under `## Unreleased` as you go, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The project is pre-1.0, so minor releases may contain breaking changes — but they must be
labelled. Mark anything that changes existing behaviour with **Breaking:** and show the
before/after, so the entry is enough on its own to fix a downstream call site.

New backward-compatible functionality (a new keyword, a new function) is a **minor** bump; a
pure bug fix with no new public API is a **patch**.

## Code Style Guidelines

### Docstrings

- NumPy-style docstrings for public functions, classes, and methods. Ruff's `D` rules enforce
  `Parameters`, `Returns`, and `Raises` sections.
- Avoid `Examples` and `Notes` sections; keep descriptions concise and free of implementation
  detail the caller doesn't need.
- Private helpers, fixtures, and test doubles need only a one-line summary.

### Comments

- Keep comments brief and explain why, not what.
- Describe current behavior; only regression-test comments may reference the former bug, e.g.
  `# Previously raised IndexError for a scalar geometry query.`

### Type checking

- Python 3.12+ annotations: `type X = ...` instead of `TypeVar`, `collections.abc` instead of
  `typing` for collection types, `X | None` instead of `Optional[X]`.
- Keep ignores and casts to a minimum.
- Only annotate what mypy cannot infer on its own (function arguments, return values, empty
  collections).

### Imports

- Place imports at the top of the file in the standard import section.
- Every module starts with `from __future__ import annotations`.
