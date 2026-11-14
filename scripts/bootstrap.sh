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
bash scripts/train.sh
python scripts/run_stack.py start
