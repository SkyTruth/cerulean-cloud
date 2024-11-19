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

echo "[INFO] Reduce package size and remove useless files"
find . -type f -name '*.pyc' | while read f; do n=$(echo "$f" | sed 's/__pycache__\///' | sed 's/.cpython-[2-3][0-9]//'); cp "$f" "$n"; done || (
  echo "[ERR] Failed ..."
  exit 1
)
find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf || (
  echo "[ERR] Failed ..."
  exit 1
)
find /var/task -type d -a -name 'tests' -print0 | xargs -0 rm -rf || (
  echo "[ERR] Failed ..."
  exit 1
)
rm -rdf /var/task/numpy/doc/ || (
  echo "[ERR] Failed ..."
  exit 1
)
rm -rdf /var/task/stack || (
  echo "[ERR] Failed ..."
  exit 1
)
find /var/task -type d -name "*.dist-info" -exec rm -r {} + || (
  echo "[ERR] Failed ..."
  exit 1
)

# shellcheck disable=SC2035
echo "[INFO] Creating package.zip"
zip -r9q /local/package.zip * || (
  echo "[ERR] Failed ..."
  exit 1
)
