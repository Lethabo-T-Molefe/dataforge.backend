"""Start JupyterLab for DataForge.

    .venv/Scripts/python run.py        (Windows)
    .venv/bin/python run.py            (Linux, macOS)

Loads .env from this folder, installs the DataForge settings overrides into this environment's JupyterLab,
and starts the server on http://127.0.0.1:8888/jupyter/.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load_env() -> None:
    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


CONFIG_DIR = HERE / "config"


def install_theme() -> None:
    """Install the DataForge look.

    JupyterLab reads default settings from <prefix>/share/jupyter/lab/settings/overrides.json, and serves
    custom CSS only from <JUPYTER_CONFIG_DIR>/custom, so both are copied into place on every start.
    """
    target = Path(sys.prefix) / "share" / "jupyter" / "lab" / "settings"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "dataforge_jupyter" / "overrides.json", target / "overrides.json")
    shutil.copytree(HERE / "dataforge_jupyter" / "custom", CONFIG_DIR / "custom", dirs_exist_ok=True)
    os.environ["JUPYTER_CONFIG_DIR"] = str(CONFIG_DIR)


def main() -> None:
    load_env()
    for name in ("SSO_SHARED_SECRET", "JUPYTER_API_TOKEN"):
        if not os.environ.get(name):
            raise SystemExit(f"Set {name} in integrations/jupyter/.env (see .env.example).")
    install_theme()
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH", "")]))
    command = [sys.executable, "-m", "jupyterlab", f"--config={HERE / 'jupyter_server_config.py'}"]
    raise SystemExit(subprocess.call(command, env=os.environ, cwd=HERE))


if __name__ == "__main__":
    main()
