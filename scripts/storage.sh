set -eu
cd "$(dirname "$0")/.."
mkdir -p logs
storage_location="${LEDGERLIGHT_STORAGE_PATH:-.}"
{ date -u +%FT%TZ; df -h / "$storage_location"; du -sh .; } | tee -a "logs/storage-$1.log"
python3 - "$storage_location" <<'PYTHON'
import shutil
import sys
for location in ('/', sys.argv[1]):
    if shutil.disk_usage(location).free < 20 * 1024**3:
        raise SystemExit(f'Storage guard: {location} below 20 GiB free')
PYTHON
