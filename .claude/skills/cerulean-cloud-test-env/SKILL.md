---
name: cerulean-cloud-test-env
description: Ensure tests in the cerulean-cloud repository run in the repo-local .conda/ceru-ci environment instead of the default interpreter. Use when running pytest, targeted pytest slices, or other test commands from /Users/jonathanraphael/git/cerulean-cloud.
---

# Cerulean Cloud Test Env

Use this skill only for test execution in `/Users/jonathanraphael/git/cerulean-cloud`.

## Rule

Before running any test command in this repo, invoke it through the repo-local `ceru-ci` environment with this prefix:

```bash
PROJ_DATA=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/proj_data \
PROJ_LIB=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/proj_data \
GDAL_DATA=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/gdal_data \
conda run -p /Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci
```

Append the actual test command after that prefix.

## Examples

Run the full suite:

```bash
PROJ_DATA=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/proj_data \
PROJ_LIB=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/proj_data \
GDAL_DATA=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/gdal_data \
conda run -p /Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci pytest
```

Run a targeted slice:

```bash
PROJ_DATA=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/proj_data \
PROJ_LIB=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/proj_data \
GDAL_DATA=/Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci/lib/python3.11/site-packages/rasterio/gdal_data \
conda run -p /Users/jonathanraphael/git/cerulean-cloud/.conda/ceru-ci pytest test/test_cerulean_cloud/test_cloud_run_orchestrator.py -k sea_ice_mask
```

## Notes

- Do not run repo tests with the default shell Python.
- Keep the geospatial env vars in front of `conda run`, or rasterio/PROJ imports may fail during collection.
- If PostgreSQL-backed tests fail because `pg_ctl` is missing from `ceru-ci`, report that as an environment limitation rather than a code regression.
