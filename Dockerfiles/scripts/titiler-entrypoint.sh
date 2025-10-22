#!/bin/sh
set -euo pipefail

echo "[INFO] Preparing Cerulean Titiler installation package ..."

if [ ! -d /local ]; then
  echo "[ERR] Required mount point /local not found"
  exit 1
fi

cd /var/task

# Prefer prebuilt wheels and upgrade pip tooling for better resolver behavior
export PIP_PREFER_BINARY=1
export PIP_ONLY_BINARY=":all:"

# Ensure we install with the same interpreter as the Lambda runtime
python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || true
if ! python3 -m pip install \
    --no-input \
    --disable-pip-version-check \
    --no-warn-script-location \
    --isolated \
    --no-cache-dir \
    --upgrade \
    --target /var/task \
    --only-binary=:all: \
    -r requirements.txt; then
  echo "[ERR] Failed to install Python packages"
  exit 1
fi

echo "[INFO] Reduce package size and remove useless files"
# If no matches, xargs -r avoids invoking rm with no args
find . -type f -name '*.pyc' | while read -r f; do n=$(echo "$f" | sed 's/__pycache__\///' | sed 's/.cpython-[2-3][0-9]//'); cp "$f" "$n"; done || true
find . -type d -a -name '__pycache__' -print0 | xargs -0 -r rm -rf || true
find /var/task -type d -a -name 'tests' -print0 | xargs -0 -r rm -rf || true
rm -rf /var/task/numpy/doc/ || true
rm -rf /var/task/stack || true
find /var/task -type d -name "*.dist-info" -print0 | xargs -0 -r rm -rf || true

# shellcheck disable=SC2035
echo "[INFO] Creating package.zip"
zip -r9q /local/package.zip *
