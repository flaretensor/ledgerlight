import importlib.util
from pathlib import Path

import pytest


def test_unmeasured_gpu_layers_use_larger_neighbour_peak():
    specification = importlib.util.spec_from_file_location(
        "site_lifecycle", Path(__file__).parents[1] / "scripts/site.py"
    )
    lifecycle = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(lifecycle)
    settings = {
        "total_layers": 24,
        "vram_mib": {"torch": {"0": 3691, "12": 3538, "all": 3130}},
    }
    config = {"search": "torch", "triton": False, "gpu_layers": "6"}
    assert lifecycle.vram_requirement(config, settings) == (3691, True)
    config["gpu_layers"] = "18"
    assert lifecycle.vram_requirement(config, settings) == (3538, True)
    config["gpu_layers"] = "12"
    assert lifecycle.vram_requirement(config, settings) == (3538, False)


def test_public_mode_requires_nonempty_private_token_file(tmp_path, monkeypatch):
    specification = importlib.util.spec_from_file_location(
        "site_token", Path(__file__).parents[1] / "scripts/site.py"
    )
    lifecycle = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(lifecycle)
    token = tmp_path / "cloudflared/token"
    token.parent.mkdir(mode=0o700)
    monkeypatch.setattr(lifecycle, "TOKEN", token)
    with pytest.raises(ValueError, match="absent or empty"):
        lifecycle.public_token()
    token.write_text("")
    with pytest.raises(ValueError, match="absent or empty"):
        lifecycle.public_token()
    token.write_text("synthetic-test-token")
    token.chmod(0o644)
    with pytest.raises(ValueError, match="0700.*0600"):
        lifecycle.public_token()
    token.chmod(0o600)
    assert lifecycle.public_token().st_size > 0


@pytest.fixture
def lifecycle():
    specification = importlib.util.spec_from_file_location(
        "site_cpu", Path(__file__).parents[1] / "scripts/site.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_cpu_configuration_does_not_query_gpu(lifecycle, monkeypatch, tmp_path):
    from argparse import Namespace

    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(lifecycle.subprocess, "check_output", missing)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(lifecycle, "ACTIVE", tmp_path / "active.json")
    monkeypatch.setattr(lifecycle, "OVERRIDE", tmp_path / "active.compose.yaml")
    args = Namespace(search=None, triton=None, gpu_layers=None, cpu=True, public=False, force=False)
    config = lifecycle.configure(args)
    assert config["search"] == "openvino" and config["gpu_layers"] == "0"
    assert not config["triton"]


def test_missing_nvidia_smi_is_zero_processes_for_stop_and_status(
    lifecycle, monkeypatch, tmp_path, capsys
):
    from ledgerlight import common

    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(lifecycle.subprocess, "run", missing)
    monkeypatch.setattr(lifecycle, "container_ids", lambda *args: [])
    monkeypatch.setattr(lifecycle, "project_pids", lambda: set())
    monkeypatch.setattr(lifecycle, "inspect_containers", lambda: [])
    monkeypatch.setattr(lifecycle, "PORTS", [])
    monkeypatch.setattr(lifecycle, "ROOT", tmp_path)
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(lifecycle, "ACTIVE", tmp_path / "active.json")
    assert lifecycle.gpu_processes({123}) == []
    lifecycle.stop()
    lifecycle.status()
    text = capsys.readouterr().out
    assert text.count("PASS:") == 4
    assert '"stack_gpu": []' in text


@pytest.mark.parametrize(
    "layers,triton,search", [("-1", False, "ort"), ("bad", False, "ort"), ("all", True, "torch")]
)
def test_settings_validation_is_pure(lifecycle, layers, triton, search):
    import copy
    from argparse import Namespace

    settings = {"search": search, "triton": triton, "gpu_layers": layers, "llm_threads": 4}
    original = copy.deepcopy(settings)
    args = Namespace(search=None, triton=None, gpu_layers=None, cpu=False, public=False)
    with pytest.raises(ValueError):
        lifecycle.resolve_settings(settings, args, {})
    assert settings == original


def test_vram_policy_is_pure_and_cpu_does_not_need_capacity(lifecycle):
    gpu = {"search": "ort", "gpu_layers": "all"}
    with pytest.raises(ValueError, match="VRAM fit unverified or insufficient"):
        lifecycle.check_vram(gpu, 1000, 1149, False)
    lifecycle.check_vram(gpu, 1000, 1150, False)
    lifecycle.check_vram(gpu, None, 0, True)
    lifecycle.check_vram({"search": "openvino", "gpu_layers": "0"}, None, 0, False)


def test_cpu_stop_status_skip_nvidia_smi(lifecycle, monkeypatch, tmp_path, capsys):
    import json

    active = tmp_path / "active.json"
    active.write_text(json.dumps({"search": "openvino", "gpu_layers": "0"}))
    monkeypatch.setattr(lifecycle, "ACTIVE", active)
    monkeypatch.setattr(lifecycle, "ROOT", tmp_path)
    monkeypatch.setattr(lifecycle, "inspect_containers", lambda: [])
    monkeypatch.setattr(lifecycle, "container_ids", lambda *args: [])
    monkeypatch.setattr(lifecycle, "project_pids", lambda: {123})
    monkeypatch.setattr(lifecycle, "PORTS", [])
    monkeypatch.setattr(lifecycle.subprocess, "run", lambda *args, **kwargs: pytest.fail())
    lifecycle.stop()
    lifecycle.status()
    assert capsys.readouterr().out.count("PASS:") == 4
