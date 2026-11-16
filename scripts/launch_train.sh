#!/usr/bin/env bash
set -euo pipefail

launch_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [[ "${1:-}" == "--check-env" ]]; then
    cd "$launch_root"
    source scripts/env.sh
    command -v python
    python --version
    uv --version
    exit
fi

if [[ $# -ne 0 ]]; then
    printf 'Usage: bash scripts/launch_train.sh [--check-env]\n' >&2
    exit 2
fi

printf -v launch_command 'cd %q && source scripts/env.sh && mkdir -p logs && exec bash scripts/train.sh >> logs/train.log 2>&1' "$launch_root"

if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t '=ledgerlight-train' 2>/dev/null; then
        printf 'Session ledgerlight-train already exists; refusing to start another training run.\n' >&2
        exit 1
    fi
    printf -v launch_shell '/bin/bash -c %q' "$launch_command"
    tmux new-session -d -s ledgerlight-train "$launch_shell"
    printf 'Training started in tmux session ledgerlight-train.\n'
else
    nohup /bin/bash -c "$launch_command" >/dev/null 2>&1 </dev/null &
    printf 'Training started with nohup (PID %s).\n' "$!"
fi

printf '%s\n' 'Progress: tail -n 30 logs/train.log' 'DEV report: data/evaluation/dev_metrics.json'
