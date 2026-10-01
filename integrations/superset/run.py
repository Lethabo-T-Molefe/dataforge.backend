"""Start Superset with DataForge's configuration on port 8088 (development server).

    .venv/Scripts/python run.py

In production run Superset under gunicorn on Linux with the same environment:
    gunicorn -w 4 -k gevent -b 0.0.0.0:8088 "superset.app:create_app()"
"""

import os
import subprocess

from bootstrap import load_env, superset_exe

if __name__ == "__main__":
    load_env()
    port = os.environ.get("SUPERSET_PORT", "8088")
    host = os.environ.get("SUPERSET_HOST", "127.0.0.1")
    raise SystemExit(subprocess.call([superset_exe(), "run", "-h", host, "-p", port, "--with-threads"], env=os.environ))
