set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/run_stack.py tracking
python -m ledgerlight.run_training
