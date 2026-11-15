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
