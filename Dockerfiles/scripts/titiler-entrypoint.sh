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

# 2) Copy compiled bytecode out of __pycache__ to top-level as module.pyc
echo "[INFO] Promoting bytecode from __pycache__ to top-level .pyc"
find /var/task -type f -path '*/__pycache__/*.pyc' | while read -r f; do \
  dir=$(dirname "$f"); \
  parent=$(dirname "$dir"); \
  base=$(basename "$f"); \
  destname=$(echo "$base" | sed -E 's/\.cpython-[0-9]+\.pyc$/.pyc/'); \
  cp "$f" "$parent/$destname"; \
done || ( \
  echo "[ERR] Failed to place compiled bytecode"; \
  exit 1 \
)

# 3) Remove __pycache__ directories
find /var/task -type d -name '__pycache__' -print0 | xargs -0 rm -rf || (
  echo "[ERR] Failed to remove __pycache__"
  exit 1
)

# 4) Remove source .py files except our application entry and support files
find /var/task -type f -name '*.py' \
  ! -path '/var/task/handler.py' \
  ! -path '/var/task/auth.py' \
  ! -path '/var/task/fastapi/*' \
  ! -path '/var/task/starlette/*' \
  ! -path '/var/task/pydantic/*' \
  ! -path '/var/task/pydantic_core/*' \
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

# 6) Remove dist-info metadata (not needed at runtime)
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

# 7) Sanity check imports from a zipped archive (simulate Lambda zipimport)
echo "[INFO] Creating temporary test zip and validating imports from it"
rm -f /tmp/package-test.zip
zip -r9q /tmp/package-test.zip * || (
  echo "[ERR] Failed to create test zip"
  exit 1
)
python - <<'PY'
import sys
sys.path.insert(0, '/tmp/package-test.zip')
print('sys.path[0]=', sys.path[0])
from fastapi import Depends, FastAPI, Query
from starlette.middleware.cors import CORSMiddleware
from mangum import Mangum
from titiler.core.factory import MultiBandTilerFactory
from rio_tiler_pds.sentinel.aws import S1L1CReader
import jinja2, markupsafe, anyio, sniffio
print('Zipped import sanity check: OK')
PY
if [ $? -ne 0 ]; then
  echo "[ERR] Zipped import sanity check failed"
  exit 1
fi

# shellcheck disable=SC2035
echo "[INFO] Creating package.zip"
zip -r9q /local/package.zip * || (
  echo "[ERR] Failed ..."
  exit 1
)
