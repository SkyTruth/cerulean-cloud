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

pip install \
    --no-input \
    --disable-pip-version-check \
    --no-warn-script-location \
    --isolated \
    --no-cache-dir \
    --upgrade \
    --target /var/task \
    -r requirements.txt || (
  echo "[ERR] Failed to install Python packages"
  exit 1
)

echo "[INFO] Reduce package size: keep only bytecode (pyc) + app files"

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

# 2) Remove source .py files except our application entry and support files
find /var/task -type f -name '*.py' \
  ! -path '/var/task/handler.py' \
  ! -path '/var/task/auth.py' \
  -delete || (
  echo "[ERR] Failed to remove .py sources"
  exit 1
)

# 3) Remove test folders and large docs to trim size
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
python - <<'PY'
mods = [
    'fastapi', 'starlette', 'mangum',
    'titiler.core', 'rio_tiler', 'rio_tiler_pds',
    'jinja2', 'markupsafe', 'anyio', 'sniffio'
]
for m in mods:
    __import__(m)
print('Import sanity check: OK')
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
