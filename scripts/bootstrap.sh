set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
uv sync --frozen
mkdir -p .runtime logs .tmp data/postgres data/prometheus data/grafana
python - <<'PY'
from pathlib import Path
import secrets
p=Path('.env')
if not p.exists():
    p.write_text('POSTGRES_USER=ledgerlight\nPOSTGRES_PASSWORD='+secrets.token_hex(24)+'\nPOSTGRES_DB=ledgerlight\nPOSTGRES_PORT=5433\nGRAFANA_PASSWORD='+secrets.token_hex(16)+'\nLEDGERLIGHT_BACKEND=ort-cuda\n')
    p.chmod(0o600)
PY
docker compose stop llama retrieval gateway ui
docker compose up -d --build postgres mlflow
python - <<'PY'
import time
import httpx
deadline=time.monotonic()+180
while True:
    try:
        response=httpx.get('http://127.0.0.1:5000/health',timeout=2)
        response.raise_for_status()
        break
    except httpx.HTTPError:
        if time.monotonic()>=deadline:
            raise RuntimeError('MLflow did not become ready') from None
        time.sleep(1)
PY
python scripts/fetch_services.py llama qwen
LEDGERLIGHT_RESUME=1 dvc repro evaluate
if [ ! -f data/registry/versions.json ]; then python -m ledgerlight.registry; fi
ledgerlight start
