# Testing and Version Verification

This document is for maintainers, not normal end users.

The goal of the `nox` smoke matrix is to answer two practical questions:

- Does the project install in a clean environment?
- Does every requested Python version import the full codebase without missing-package surprises?

## What `nox` is doing

The repo's [noxfile.py](../noxfile.py) defines two smoke matrices:

- `smoke` for currently supported versions
- `smoke-next` for exploratory newer versions

The supported `smoke` matrix currently covers:

- `3.10`
- `3.11`
- `3.12`

The exploratory `smoke-next` matrix currently covers:

- `3.13`
- `3.14`

For each version, `nox`:

1. creates a fresh environment
2. installs the project with `pip install -e .`
3. compiles `src/` and `demo/`
4. runs [scripts/smoke_imports.py](../scripts/smoke_imports.py), which imports every local module and demo script

This is a smoke test, not a behavioral test suite. It is meant to catch packaging, dependency, and interpreter-compatibility problems quickly.

## Recommended way to run the full matrix

Use `uv` plus `nox` with the `uv` extra so `nox` can create environments with the `uv` backend and download any missing interpreters automatically:

```bash
uv tool run --with 'nox[uv]' nox -s smoke
```

If you want to probe newer interpreter versions too:

```bash
uv tool run --with 'nox[uv]' nox -s smoke-next
```

The `smoke-next` session enables a small Tkinter stub during imports. This keeps the exploratory smoke run useful on interpreter builds that lack `_tkinter`, while still checking the rest of the codebase for import-time compatibility problems.

## If you want to preinstall the interpreters first

You can also install the full version set up front:

```bash
uv python install 3.10 3.11 3.12
uv tool run --with 'nox[uv]' nox -s smoke
```

## Running pytest on `./tests/`

```bash
uv tool run --with 'nox[uv]' nox -s tests
```

## Slow Bayesian Bins Regression

Run this before changing Bayesian Bins, DenseTC latency defaults, or SDF offset
logic:

```bash
uv tool run --with 'nox[uv]' nox -s bayesian-bins
```

See [Bayesian Bins Notes](bayesian_bins.md) for benchmark commands, expected
frozen/manual demo scores, and parameter sweep examples.
