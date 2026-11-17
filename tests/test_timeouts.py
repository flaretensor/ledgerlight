import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("script,function", [("site", "run"), ("run_stack", "main")])
def test_control_command_timeout_propagates(script, function, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts" / (script + ".py")
    specification = importlib.util.spec_from_file_location("timeout_" + script, source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    calls = []

    def expired(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(module.subprocess, "run", expired)
    if script == "run_stack":
        monkeypatch.setattr(module, "command", lambda: ["docker", "compose"])
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        getattr(module, function)(*[["docker", "info"]] if script == "site" else [])
    assert failure.value.timeout == 300
    assert calls == [300]
