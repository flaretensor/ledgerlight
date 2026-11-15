import importlib.util
from pathlib import Path


def test_default_selection_uses_independent_triton_speed_band_and_vram_tie(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    specification = importlib.util.spec_from_file_location(
        "measured_defaults", scripts / "measure_stage2.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    engines = ["torch", "ort", "openvino", "triton-ort", "triton-openvino"]
    document = {
        "parity": {e: {"passed": True} for e in engines},
        "search_latency": {
            e: {"queries": 118, "search_seconds": {"p95": p}}
            for e, p in zip(engines, [1, 0.95, 5, 1.02, 0.1], strict=True)
        },
        "matrix": {
            e + "/all": {
                "engine": e,
                "layers": "all",
                "status": "DONE",
                "completed": 8,
                "errors": {},
                "peak_gpu_mib": v,
            }
            for e, v in zip(engines, [3100, 2350, 1450, 2424, 1450], strict=True)
        },
    }
    saved = {}
    monkeypatch.setattr(module, "read_json", lambda p: document if "summary" in p else {})
    monkeypatch.setattr(module, "save", lambda p, v: None)
    monkeypatch.setattr(module, "summary", lambda s, n, v: saved.update(v))
    monkeypatch.setattr(module, "mark_step", lambda *a, **k: None)
    module.choose_defaults()
    assert saved["selected"] == "triton-ort"
    assert "triton-openvino" not in saved["eligible"]
    assert saved["p95_limit"] == 1.1 * 0.95
    document["search_latency"]["ort"]["queries"] = 117
    module.choose_defaults()
    assert saved["selected"] == "triton-ort"
    document["matrix"]["triton-ort/all"]["completed"] = 7
    module.choose_defaults()
    assert saved["selected"] == "torch"
