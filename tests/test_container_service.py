import importlib.util
import sys
from pathlib import Path

import pytest


def test_llama_thinking_budget_comes_from_serving_params(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "scripts/container_service.py"
    specification = importlib.util.spec_from_file_location("container_service_test", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    executable = tmp_path / ".tools/llama/llama-server"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(module, "path", lambda name: tmp_path / name)
    monkeypatch.setattr(module, "digest", lambda name: "synthetic")
    monkeypatch.setattr(
        module,
        "read_json",
        lambda name: {"qwen": {"name": "model.gguf", "digest": "sha256:synthetic"}},
    )
    monkeypatch.setattr(module, "params", lambda: {"serving": {"thinking_budget": 123}})
    monkeypatch.setattr(sys, "argv", [str(source), "llama"])
    calls = []
    monkeypatch.setattr(module, "serve_llama", lambda command: calls.append(command) or 0)
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0
    assert calls[0][calls[0].index("--reasoning-budget") + 1] == "123"
