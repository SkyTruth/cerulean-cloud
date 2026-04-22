---
name: cerulean-cloud-test-env
description: Ensure tests in the cerulean-cloud repository run from the canonical checkout-local Conda prefix at .conda/ceru-ci, creating that environment when it is missing and deriving geospatial data paths dynamically. Use when running pytest, targeted pytest slices, or other repo test commands.
---

# Cerulean Cloud Test Env

Use this skill only for test execution in the active `cerulean-cloud` checkout.

## Rule

Before running any test command in this repo:

1. Derive the checkout path with `git rev-parse --show-toplevel`.
2. Use the canonical test environment at `$REPO_ROOT/.conda/ceru-ci`.
3. If `$REPO_ROOT/.conda/ceru-ci` does not exist, build it before running tests.
4. Derive rasterio's `proj_data` and `gdal_data` paths from that environment; do not hard-code a developer's absolute checkout path or Python minor version.
5. Do not run tests in another environment unless the user explicitly overrides this skill.

## Build the test environment

The standard test environment is a repo-local Conda prefix:

- path: `$REPO_ROOT/.conda/ceru-ci`
- Python: `3.11`
- dependency sources:
  - `requirements.txt`
  - `requirements-test.txt`
  - `cerulean_cloud/cloud_run_orchestrator/requirements.txt`
  - `cerulean_cloud/cloud_run_tipg/requirements.txt`

Build it when missing:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
CERU_CI="$REPO_ROOT/.conda/ceru-ci"

if [ ! -x "$CERU_CI/bin/python" ]; then
  conda create -y -p "$CERU_CI" python=3.11 pip
  conda run -p "$CERU_CI" python -m pip install --upgrade pip setuptools wheel
  conda run -p "$CERU_CI" python -m pip install -r "$REPO_ROOT/requirements.txt" -r "$REPO_ROOT/requirements-test.txt"
  conda run -p "$CERU_CI" python -m pip install -r "$REPO_ROOT/cerulean_cloud/cloud_run_orchestrator/requirements.txt" -r "$REPO_ROOT/cerulean_cloud/cloud_run_tipg/requirements.txt"
fi
```

If `conda` is not available, report that a Conda-compatible package manager is required to create `$REPO_ROOT/.conda/ceru-ci`; do not silently switch to the default shell Python.

## Run tests

Use this pattern from anywhere inside the checkout:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
CERU_CI="$REPO_ROOT/.conda/ceru-ci"
RASTERIO_DIR="$(conda run -p "$CERU_CI" python -c 'from pathlib import Path; import rasterio; print(Path(rasterio.__file__).resolve().parent)')"
PROJ_DATA="$RASTERIO_DIR/proj_data" \
PROJ_LIB="$RASTERIO_DIR/proj_data" \
GDAL_DATA="$RASTERIO_DIR/gdal_data" \
conda run -p "$CERU_CI" pytest
```

Run a targeted slice by replacing the final command:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
CERU_CI="$REPO_ROOT/.conda/ceru-ci"
RASTERIO_DIR="$(conda run -p "$CERU_CI" python -c 'from pathlib import Path; import rasterio; print(Path(rasterio.__file__).resolve().parent)')"
PROJ_DATA="$RASTERIO_DIR/proj_data" \
PROJ_LIB="$RASTERIO_DIR/proj_data" \
GDAL_DATA="$RASTERIO_DIR/gdal_data" \
conda run -p "$CERU_CI" pytest test/test_cerulean_cloud/test_cloud_run_orchestrator.py -k sea_ice_mask
```

## Notes

- Do not run repo tests with the default shell Python.
- Keep the geospatial env vars in front of the test command, or rasterio/PROJ imports may fail during collection.
- If PostgreSQL-backed tests fail because `pg_ctl` is missing from `ceru-ci`, report that as an environment limitation rather than a code regression.
- Do not add service-image-specific requirement files to the canonical test env unless a target test requires them; some service images intentionally pin conflicting rasterio stacks.
- Never commit examples that reveal a maintainer's home directory, username, or machine-specific environment layout.
