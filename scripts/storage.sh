set -eu
cd "$(dirname "$0")/.."
mkdir -p logs
{ date -u +%FT%TZ; df -h / .; du -sh .; } | tee -a "logs/storage-$1.log"
python3 - <<'PY'
import shutil
for location in ('/', '.'):
    if shutil.disk_usage(location).free < 20 * 1024**3:
        raise SystemExit(f'Storage guard: {location} below 20 GiB free')
PY
