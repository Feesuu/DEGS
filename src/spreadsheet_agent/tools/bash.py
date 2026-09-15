"""
Bash command execution tool.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

# Add parent src to path for react_agent imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from react_agent import tool


_SYSTEM_ROOTS = tuple(Path(value) for value in ("/usr", "/bin", "/lib", "/lib64", "/etc"))


def _python_runtime_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for value in (sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix):
        root = Path(value).resolve()
        if root in roots or any(root == system or root.is_relative_to(system) for system in _SYSTEM_ROOTS):
            continue
        roots.append(root)
    return tuple(roots)


def _minimal_environment(working_dir: str) -> dict[str, str]:
    temporary = Path(working_dir) / ".tmp"
    temporary.mkdir(mode=0o700, exist_ok=True)
    environment = {
        "HOME": working_dir,
        "TMPDIR": str(temporary),
        "PATH": f"{Path(sys.prefix) / 'bin'}:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "VIRTUAL_ENV": str(Path(sys.prefix).resolve()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    return environment


def _bubblewrap_command(working_dir: str, command: str) -> list[str]:
    executable = shutil.which("bwrap")
    if not executable:
        raise RuntimeError("bubblewrap (`bwrap`) is required for benchmark execution")
    prefix = Path(sys.prefix).resolve()
    runtime_roots = _python_runtime_roots()
    arguments = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-net",
        "--unshare-pid",
        "--clearenv",
    ]
    for root in _SYSTEM_ROOTS:
        if root.exists():
            arguments.extend(("--ro-bind", str(root), str(root)))
    created = set()
    for runtime_root in runtime_roots:
        for parent in reversed(runtime_root.parents):
            value = str(parent)
            if value == "/" or value in created or parent in _SYSTEM_ROOTS:
                continue
            arguments.extend(("--dir", value))
            created.add(value)
    resolved_working_dir = Path(working_dir).resolve()
    for runtime_root in runtime_roots:
        arguments.extend(("--ro-bind", str(runtime_root), str(runtime_root)))
    arguments.extend(
        (
            "--bind",
            str(resolved_working_dir),
            "/workspace",
        )
    )
    retrieved_skills = resolved_working_dir / "retrieved_skills"
    if retrieved_skills.exists() or retrieved_skills.is_symlink():
        if retrieved_skills.is_symlink() or not retrieved_skills.is_dir():
            raise ValueError("retrieved Skills must be a regular directory")
        arguments.extend(
            ("--ro-bind", str(retrieved_skills), "/workspace/retrieved_skills")
        )
    arguments.extend(
        (
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            "/workspace",
            "--setenv",
            "HOME",
            "/workspace",
            "--setenv",
            "TMPDIR",
            "/workspace/.tmp",
            "--setenv",
            "PATH",
            f"{prefix / 'bin'}:/usr/bin:/bin",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "VIRTUAL_ENV",
            str(prefix),
            "--setenv",
            "LANG",
            "C.UTF-8",
            "--setenv",
            "LC_ALL",
            "C.UTF-8",
            "/bin/bash",
            "-c",
            command,
        )
    )
    return arguments


def create_bash_tool(
    working_dir: str,
    timeout: int = 120,
    *,
    sandbox_mode: str = "required",
):
    """
    Create a bash execution tool for running commands.

    Args:
        working_dir: Directory where commands will be executed
        timeout: Command timeout in seconds
        sandbox_mode: ``required`` uses bubblewrap; ``off`` is for isolated tests only
    """
    if sandbox_mode not in {"required", "off"}:
        raise ValueError("sandbox_mode must be 'required' or 'off'")
    if sandbox_mode == "required" and not shutil.which("bwrap"):
        raise RuntimeError("bubblewrap (`bwrap`) is required for benchmark execution")
    environment = _minimal_environment(working_dir)

    @tool(name="bash")
    def bash(command: str) -> str:
        """
        Execute a bash command in the working directory.
        Use this to run Python scripts, install packages, navigate files,
        or perform any shell operations.

        Args:
            command: The bash command to execute
        """
        try:
            invocation = (
                _bubblewrap_command(working_dir, command)
                if sandbox_mode == "required"
                else ["/bin/bash", "-c", command]
            )
            result = subprocess.run(
                invocation,
                shell=False,
                cwd=None if sandbox_mode == "required" else working_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=environment,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}" if output else result.stderr
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
                if "SyntaxError" in output:
                    if "python -c" in command:
                        output += (
                            "\n[Recovery hint] This `python -c` command has invalid Python syntax. "
                            "Do not retry the same one-line command. Write a multi-line `solution.py` "
                            "with a heredoc, then run `python solution.py`."
                        )
                    elif "solution.py" in output:
                        output += (
                            "\n[Recovery hint] `solution.py` is not valid Python. Do not retry the same file. "
                            "Simplify the code, avoid fragile nested quote formula strings, and write final "
                            "computed values directly when that satisfies the target cells."
                        )
            return output.strip() if output.strip() else "[Command completed with no output]"
        except subprocess.TimeoutExpired:
            return f"[ERROR] Command timed out after {timeout} seconds"
        except Exception as e:
            return f"[ERROR] Failed to execute command: {e}"

    return bash
