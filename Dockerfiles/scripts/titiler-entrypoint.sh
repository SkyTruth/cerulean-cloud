#!/bin/sh

echo "[INFO] Preparing Cerulean Titiler installation package ..."

test -d /local || (
  echo "[ERR] Required mount point /local not found"
  exit 1
)

cd /var/task || (
  echo "[ERR] Failed ..."
  exit 1
)

echo "[INFO] Installing base wheels (numpy 1.26) to avoid builds ..."
pip install \
    --no-input \
    --disable-pip-version-check \
    --no-warn-script-location \
    --no-cache-dir \
    --only-binary=:all: \
    --target /var/task \
    "numpy>=1.26,<2.0" || (
  echo "[ERR] Failed to preinstall numpy wheel"
  exit 1
)

echo "[INFO] Installing application requirements (binary-only, no build isolation) ..."
echo "[INFO] Preinstall pydantic-core and pydantic wheels explicitly"
pip install \
    --no-input \
    --disable-pip-version-check \
    --no-warn-script-location \
    --no-cache-dir \
    --only-binary=:all: \
    --target /var/task \
    'pydantic-core>=2.16,<2.19' 'pydantic>=2.5,<2.8' || (
  echo "[ERR] Failed to preinstall pydantic/pydantic-core"
  exit 1
)

PIP_NO_BUILD_ISOLATION=1 PIP_ONLY_BINARY=:all: pip install \
    --no-input \
    --disable-pip-version-check \
    --no-warn-script-location \
    --no-cache-dir \
    --target /var/task \
    -r requirements.txt || (
  echo "[ERR] Failed to install Python packages"
  exit 1
)

echo "[INFO] Reduce package size: ship top-level .pyc + app files"

# 1) Precompile all installed modules to bytecode into __pycache__
python - <<'PY'
import compileall
import sys

ok = compileall.compile_dir('/var/task', force=True, quiet=1, workers=0)
sys.exit(0 if ok else 1)
PY
if [ $? -ne 0 ]; then
  echo "[ERR] Failed to precompile python files"
  exit 1
fi

# 2) Remove source .py files except:
#    - all package markers (__init__.py) to preserve package identity
#    - our app entry files (handler.py, auth.py)
find /var/task -type f -name '*.py' \
  ! -path '/var/task/handler.py' \
  ! -path '/var/task/auth.py' \
  ! -path '/var/task/fastapi/*' \
  ! -path '/var/task/starlette/*' \
  ! -path '/var/task/pydantic/*' \
  ! -path '/var/task/pydantic_core/*' \
  ! -path '/var/task/titiler/*' \
  ! -path '/var/task/rio_tiler/*' \
  ! -path '/var/task/rio_tiler_pds/*' \
  ! -path '/var/task/mangum/*' \
  ! -path '/var/task/starlette_cramjam/*' \
  ! -path '/var/task/jinja2/*' \
  ! -path '/var/task/anyio/*' \
  ! -path '/var/task/sniffio/*' \
  ! -path '/var/task/numpy/*' \
  ! -path '/var/task/numpy*.py' \
  ! -path '/var/task/typing_extensions.py' \
  ! -path '/var/task/annotated_types.py' \
  ! -name '__init__.py' \
  -delete || (
  echo "[ERR] Failed to remove .py sources"
  exit 1
)

# 5) Remove test folders and large docs to trim size
find /var/task -type d -name 'tests' -print0 | xargs -0 rm -rf || (
  echo "[ERR] Failed to remove tests"
  exit 1
)
[ -d /var/task/numpy/doc ] && rm -rf /var/task/numpy/doc || true
[ -d /var/task/stack ] && rm -rf /var/task/stack || true

# 4) Remove dist-info metadata (not needed at runtime)
find /var/task -type d -name '*.dist-info' -print0 | xargs -0 rm -rf || (
  echo "[ERR] Failed to remove dist-info"
  exit 1
)

# 5) Sanity check imports before zipping
echo "[INFO] Listing fastapi package contents (top-level)"
find /var/task/fastapi -maxdepth 1 -type f -print | sort | head -n 50 || true
echo "[INFO] Listing pydantic_core contents"
ls -l /var/task/pydantic_core || true
echo "[INFO] Probing for pydantic_core extension (.so)"
ls -l /var/task/pydantic_core/*.so || true

# 5) Sanity check imports from /var/task (Lambda extracts zip to /var/task)
python - <<'PY'
from fastapi import Depends, FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
from mangum import Mangum
from titiler.core.factory import MultiBandTilerFactory
from rio_tiler_pds.sentinel.aws import S1L1CReader
import pydantic_core, jinja2, markupsafe, anyio, sniffio
import typing_extensions, annotated_types
import numpy as _np
print('Import sanity check: OK')
print('pydantic_core at:', getattr(pydantic_core, '__file__', None))
print('numpy at:', getattr(_np, '__file__', None), 'version:', _np.__version__)
PY
if [ $? -ne 0 ]; then
  echo "[ERR] Import sanity check failed"
  exit 1
fi

# shellcheck disable=SC2035
echo "[INFO] Creating package.zip"
zip -r9q /local/package.zip * || (
  echo "[ERR] Failed ..."
  exit 1
)
