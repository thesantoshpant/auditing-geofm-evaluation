C="$1"
cd "$(dirname "$0")/.."
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
B="https://data.source.coop/kerner-lab/fields-of-the-world-archive"
mkdir -p /tmp/ftw_${C}_dl && cd /tmp/ftw_${C}_dl
echo "### downloading ${C}.zip"; curl -s -A "$UA" -o ${C}.zip "$B/${C}.zip" && echo "downloaded $(du -h ${C}.zip|cut -f1)"
echo "### unzip (excluding s2_images)"; unzip -q -o ${C}.zip -x 's2_images/*' && echo "ok"
echo "instance masks: $(ls label_masks/instance 2>/dev/null | wc -l)"
cd "$(dirname "$0")/.."
cp /tmp/ftw_${C}_dl/chips_${C}.parquet data/ftw/ && cp /tmp/ftw_${C}_dl/data_config_${C}.json data/ftw/
echo "### config window"; C="$C" ./.venv/bin/python -c "import os,json;c=json.load(open('data/ftw/data_config_%s.json'%os.environ['C']));print('year',c.get('year_of_collection'));print('seasons',json.dumps(c['seasons'])[:200])"
echo "### extract ${C} polygons"; ./.venv/bin/python scripts/ftw_to_polygons.py --country ${C} --ftw-dir /tmp/ftw_${C}_dl --out data/labels/polygons_ftw_${C}.parquet 2>&1 | tail -3
echo "### ${C} PREP DONE"
