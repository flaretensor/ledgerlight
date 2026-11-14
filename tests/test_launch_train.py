import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def workspace(tmp_path):
    root = tmp_path / "project with spaces"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parents[1] / "scripts/launch_train.sh", root / "scripts/launch_train.sh"
    )
    binaries = root / ".venv/bin"
    binaries.mkdir(parents=True)
    for name, target in [("python", sys.executable), ("bash", "/bin/bash")]:
        (binaries / name).symlink_to(target)
    (root / "scripts/env.sh").write_text('export PATH="' + str(binaries) + ':$PATH"\n')
    (root / "scripts/train.sh").write_text(
        "python - <<'PY'\nimport json, os, sys\n"
        "print(json.dumps({'mode': os.environ.get('LEDGERLIGHT_RUN_MODE'), "
        "'experiment': os.environ.get('LEDGERLIGHT_EXPERIMENT'), "
        "'suffix': os.environ.get('LEDGERLIGHT_STUDY_SUFFIX'), "
        "'python': sys.executable, 'cwd': os.getcwd()}), flush=True)\nPY\n"
    )
    tools = tmp_path / "bin"
    tools.mkdir()
    for name in ["dirname", "mkdir", "nohup"]:
        (tools / name).symlink_to(shutil.which(name))
    env = {
        **os.environ,
        "PATH": str(tools),
        "LEDGERLIGHT_RUN_MODE": "pilot",
        "LEDGERLIGHT_EXPERIMENT": "stale",
        "LEDGERLIGHT_STUDY_SUFFIX": "-stale",
    }
    return root, tools, env


def output(root):
    target = root / "logs/train.log"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if target.exists() and target.read_text().strip():
            return json.loads(target.read_text())
        time.sleep(0.02)
    raise AssertionError("Detached command did not produce its environment record")


def assert_environment(record, root):
    assert record["mode"] == "canonical"
    assert record["experiment"] is None and record["suffix"] is None
    assert record["python"] == str(root / ".venv/bin/python")
    assert record["cwd"] == str(root)


def test_tmux_command_initializes_a_stale_server_environment(tmp_path):
    root, tools, env = workspace(tmp_path)
    tmux = tools / "tmux"
    tmux.write_text(
        f"#!{sys.executable}\nimport os, subprocess, sys\n"
        "if sys.argv[1] == 'has-session': sys.exit(1)\n"
        "assert sys.argv[1:5] == ['new-session', '-d', '-s', 'ledgerlight-train']\n"
        "env = {**os.environ, 'PATH':'/usr/bin:/bin', 'LEDGERLIGHT_RUN_MODE':'pilot', "
        "'LEDGERLIGHT_EXPERIMENT':'server', 'LEDGERLIGHT_STUDY_SUFFIX':'-server'}\n"
        "sys.exit(subprocess.call(['/bin/bash', '-c', sys.argv[5]], env=env))\n"
    )
    tmux.chmod(0o755)
    subprocess.run(["/bin/bash", str(root / "scripts/launch_train.sh")], env=env, check=True)
    assert_environment(output(root), root)


def test_launcher_refuses_an_existing_named_session(tmp_path):
    root, tools, env = workspace(tmp_path)
    tmux = tools / "tmux"
    tmux.write_text("#!/bin/bash\n[[ $1 == has-session ]]\n")
    tmux.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(root / "scripts/launch_train.sh")],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "already exists" in result.stderr
    assert not (root / "logs").exists()


def test_nohup_initializes_the_same_environment(tmp_path):
    root, _, env = workspace(tmp_path)
    subprocess.run(["/bin/bash", str(root / "scripts/launch_train.sh")], env=env, check=True)
    assert_environment(output(root), root)
