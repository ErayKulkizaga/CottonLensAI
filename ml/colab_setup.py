"""Stdlib-only notebook helpers. The notebook kernel never imports training packages."""

import os
import subprocess
import sys
from pathlib import Path

PYTHON_VERSION = "3.12.11"
UV_VERSION = "0.12.0"
REPOSITORY = "https://github.com/ErayKulkizaga/CottonLensAI.git"
PROJECT = Path("/content/CottonLensAI")
ENVIRONMENT = Path("/content/cottonlens-py312")
INFERENCE = Path("/content/cottonlens-inference")


def process_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "MLFLOW_TRACKING_URI", "MLFLOW_ALLOW_FILE_STORE", "TF_USE_LEGACY_KERAS"):
        env.pop(key, None)
    env.update({"MPLBACKEND": "Agg", "KERAS_BACKEND": "tensorflow", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"})
    libraries = sorted(ENVIRONMENT.glob("lib/python3.12/site-packages/nvidia/*/lib"))
    drivers = [Path("/usr/lib64-nvidia"), Path("/usr/local/nvidia/lib64")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(str(path) for path in [*libraries, *drivers] if path.is_dir())
    binaries = sorted(ENVIRONMENT.glob("lib/python3.12/site-packages/nvidia/*/bin"))
    env["PATH"] = os.pathsep.join([*(str(path) for path in binaries), env.get("PATH", "")])
    return env


def run(command: list[str], *, env=None, cwd=PROJECT) -> None:
    command = [str(part) for part in command]
    print("$ " + " ".join(command), flush=True)
    with subprocess.Popen(command, cwd=cwd, env=env or process_env(), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
        assert child.stdout is not None
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    if code:
        raise RuntimeError(f"Stage failed (exit {code}). Full stdout/stderr is printed above: {' '.join(command)}")


def setup() -> Path:
    if sys.platform != "linux" or not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Run setup in Google Colab after mounting Drive")
    tools_root = Path("/content/cottonlens-tools")
    # --target changes only this tool directory, never Colab's site-packages.
    # Avoid system `venv`: Colab images need not ship Python's ensurepip module.
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade",
         "--target", str(tools_root), f"uv=={UV_VERSION}"])
    uv = str(tools_root / "bin/uv")
    env = process_env()
    env["UV_PYTHON_INSTALL_DIR"] = "/content/cottonlens-python"
    env["UV_PROJECT_ENVIRONMENT"] = str(ENVIRONMENT)
    run([uv, "python", "install", PYTHON_VERSION], env=env)
    run([uv, "sync", "--project", "ml", "--locked", "--no-dev", "--extra", "cuda",
         "--python", PYTHON_VERSION, "--managed-python"], env=env)
    python = ENVIRONMENT / "bin/python"
    run([uv, "pip", "check", "--python", str(python)], env=env)
    run([uv, "venv", "--allow-existing", "--python", PYTHON_VERSION, "--managed-python", str(INFERENCE)], env=env)
    # Use the same lock for the inference-only dependency closure; no TF/Keras/MLflow.
    requirements = Path("/content/cottonlens-inference.txt")
    run([uv, "export", "--project", "ml", "--locked", "--no-dev", "--no-emit-project",
         "--no-hashes", "--output-file", str(requirements)], env=env)
    run([uv, "pip", "install", "--python", str(INFERENCE / "bin/python"),
         "--constraint", str(requirements), "numpy==2.1.3", "onnxruntime==1.22.1", "xgboost==3.0.2"], env=env)
    return python


if __name__ == "__main__":
    print(setup())
