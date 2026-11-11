set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=""
ruff check src scripts tests
ruff format --check src scripts tests
pytest -q
python scripts/leakage_check.py --static
python -m ledgerlight.fixture ci
