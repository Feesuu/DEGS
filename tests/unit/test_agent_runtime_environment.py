from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from degs_skill2bench import runtime as skill2bench_runtime
from spreadsheet_agent.tools import bash as sandbox_bash


def test_spreadsheet_sandbox_mounts_venv_and_base_interpreter(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox_bash.shutil, "which", lambda _name: "/usr/bin/bwrap")
    monkeypatch.setattr(sandbox_bash.sys, "prefix", "/workspace/.venv")
    monkeypatch.setattr(sandbox_bash.sys, "exec_prefix", "/workspace/.venv")
    monkeypatch.setattr(sandbox_bash.sys, "base_prefix", "/opt/python")
    monkeypatch.setattr(sandbox_bash.sys, "base_exec_prefix", "/opt/python")

    command = sandbox_bash._bubblewrap_command(str(tmp_path), "true")
    triples = set(zip(command, command[1:], command[2:]))
    assert ("--ro-bind", "/workspace/.venv", "/workspace/.venv") in triples
    assert ("--ro-bind", "/opt/python", "/opt/python") in triples
    environment = sandbox_bash._minimal_environment(str(tmp_path))
    assert environment["PATH"].startswith("/workspace/.venv/bin:")
    assert environment["VIRTUAL_ENV"] == "/workspace/.venv"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in environment


def test_skill2bench_worker_exposes_project_environment(monkeypatch, tmp_path):
    captured = {}

    def run(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(skill2bench_runtime.subprocess, "run", run)
    assert skill2bench_runtime._worker_request(
        "aggregate", {}, baseline_root=tmp_path
    ) == {}
    environment = captured["env"]
    prefix = Path(skill2bench_runtime.sys.prefix).resolve()
    assert environment["PATH"].split(os.pathsep)[0] == str(prefix / "bin")
    assert environment["VIRTUAL_ENV"] == str(prefix)
    assert environment["PYTHONNOUSERSITE"] == "1"
