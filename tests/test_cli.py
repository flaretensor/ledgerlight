import os
import subprocess
import sys

import pytest

MODULES = [
    "data",
    "retrieval",
    "fixture",
    "training",
    "registry",
    "vector_store",
    "evaluation",
    "run_training",
    "cli",
    "final_evaluation",
    "posthoc",
]
SCRIPTS = [
    "brand_ui",
    "container_service",
    "ui_service",
    "forward_ui",
    "run_stack",
    "leakage_check",
    "checks/admission_locust",
    "checks/large_encoder",
    "checks/search_parity",
    "checks/streaming_browser",
]


@pytest.mark.parametrize(
    "command",
    [["-m", "ledgerlight." + name] for name in MODULES]
    + [["scripts/" + name + ".py"] for name in SCRIPTS],
)
def test_entrypoint_help(command):
    result = subprocess.run(
        [sys.executable, *command, "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_data_unknown_action():
    result = subprocess.run(
        [sys.executable, "-m", "ledgerlight.data", "bogus"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
