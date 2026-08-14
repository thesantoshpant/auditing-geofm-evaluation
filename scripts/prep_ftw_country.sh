#!/usr/bin/env bash
# Download FTW labels and convert one country's instance masks to polygons.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/prep_ftw_country.sh <country>" >&2
  exit 2
fi
C="$1"
case "$C" in
  india|cambodia|vietnam|kenya|france|netherlands) ;;
  *) echo "Unsupported country: $C" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${PY:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PY=python
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    echo "No Python interpreter found. Set PY to the project interpreter." >&2
    exit 2
  fi
fi

UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
BASE_URL="https://data.source.coop/kerner-lab/fields-of-the-world-archive"
DOWNLOAD_DIR="${TMPDIR:-/tmp}/ftw_${C}_download"
ZIP_PATH="$DOWNLOAD_DIR/${C}.zip"
mkdir -p "$DOWNLOAD_DIR" data/ftw data/labels data/index
echo "### downloading ${C}.zip"
curl --fail --location --silent --show-error -A "$UA" -o "$ZIP_PATH" "$BASE_URL/${C}.zip"
echo "downloaded $(du -h "$ZIP_PATH" | cut -f1)"
echo "### unzip (excluding s2_images)"
unzip -q -o "$ZIP_PATH" -d "$DOWNLOAD_DIR" -x 's2_images/*'
cp "$DOWNLOAD_DIR/chips_${C}.parquet" data/ftw/
cp "$DOWNLOAD_DIR/data_config_${C}.json" data/ftw/
echo "### config window"
C="$C" "$PY" -c 'import json, os; c=json.load(open("data/ftw/data_config_%s.json" % os.environ["C"])); print("year", c.get("year_of_collection")); print("seasons", json.dumps(c["seasons"])[:200])'
echo "### extract ${C} polygons"
"$PY" scripts/ftw_to_polygons.py --country "$C" --ftw-dir "$DOWNLOAD_DIR" --out "data/labels/polygons_ftw_${C}.parquet"
echo "### ${C} PREP DONE"
