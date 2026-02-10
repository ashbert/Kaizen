"""
Modal container runner for the Kaizen py_to_go demo.

Runs the demo on a Modal container (no GPU needed). The container has
Python 3.11 + Go + git. LLM calls go to the existing vLLM endpoint.

Usage:
    modal run modal_demo_runner.py
"""

import modal
import sys
from pathlib import Path

VLLM_ENDPOINT = (
    "https://ashwin-chaugule--kaizen-vllm-inference-v1-chat-completions.modal.run"
)

app = modal.App("kaizen-demo-runner")

_PROJECT_ROOT = str(Path(__file__).resolve().parent)

# Image: Python 3.11 + Go + git + httpx + project source
demo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "ca-certificates")
    .run_commands(
        "wget -q https://go.dev/dl/go1.22.5.linux-amd64.tar.gz",
        "tar -C /usr/local -xzf go1.22.5.linux-amd64.tar.gz",
        "rm go1.22.5.linux-amd64.tar.gz",
    )
    .env({"PATH": "/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
    .pip_install("httpx>=0.27.0")
    .add_local_dir(
        _PROJECT_ROOT,
        remote_path="/mnt/kaizen",
        ignore=[".git", "__pycache__", ".venv", ".agentfs", "node_modules", ".mypy_cache"],
    )
)


@app.function(
    image=demo_image,
    timeout=2400,  # 40 minutes
)
def run_demo():
    import subprocess, os, shutil

    writable = "/tmp/kaizen"
    if os.path.exists(writable):
        shutil.rmtree(writable)
    shutil.copytree("/mnt/kaizen", writable)

    # Remove stale session if it got copied
    session_file = os.path.join(writable, "demo", "py_to_go", "py_to_go.kaizen")
    if os.path.exists(session_file):
        os.remove(session_file)

    env = os.environ.copy()
    env["KAIZEN_MODEL_URL"] = VLLM_ENDPOINT
    env["PYTHONPATH"] = f"{writable}/src:{writable}"
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(
        [sys.executable, "-u", f"{writable}/demo/py_to_go/run_demo.py"],
        env=env,
        cwd=writable,
    )
    return result.returncode


@app.local_entrypoint()
def main():
    code = run_demo.remote()
    print(f"\nDemo exited with code: {code}")
    raise SystemExit(code)
