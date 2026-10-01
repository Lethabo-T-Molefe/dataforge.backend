"""JupyterLab configuration for DataForge notebooks.

Members open JupyterLab from DataForge (single sign-on), work in members/<id>/ and shared/, and read data
only through the DataForge API with their own token. The UI uses the DataForge design system.
Settings come from the environment (see .env.example in this folder).
"""

import os
from pathlib import Path

HERE = Path(__file__).parent
PACKAGE = HERE / "dataforge_jupyter"
WORKSPACE = Path(os.environ.get("JUPYTER_WORKSPACE", HERE / "workspace")).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
(WORKSPACE / "members").mkdir(exist_ok=True)
(WORKSPACE / "shared").mkdir(exist_ok=True)

c = get_config()  # type: ignore[name-defined]  # noqa: F821

_frontend = [
    o.strip()
    for o in os.environ.get("DATAFORGE_FRONTEND_ORIGINS", "http://localhost:8080").split(",")
    if o.strip()
]

# Served under /jupyter so a reverse proxy can put it next to the DataForge frontend and API.
c.ServerApp.base_url = "/jupyter/"
c.ServerApp.ip = os.environ.get("JUPYTER_HOST", "127.0.0.1")
c.ServerApp.port = int(os.environ.get("JUPYTER_PORT", "8888"))
c.ServerApp.open_browser = False
c.ServerApp.root_dir = str(WORKSPACE)
c.ServerApp.allow_remote_access = os.environ.get("JUPYTER_ALLOW_REMOTE", "false") == "true"
c.ServerApp.terminals_enabled = False
c.ServerApp.allow_origin_pat = "|".join(o.replace(".", r"\.") for o in _frontend)
c.ServerApp.tornado_settings = {
    "headers": {
        # Only DataForge may frame JupyterLab.
        "Content-Security-Policy": "frame-ancestors 'self' " + " ".join(_frontend),
    },
}

# DataForge single sign-on, per-member access and per-member kernels.
c.ServerApp.identity_provider_class = "dataforge_jupyter.auth.DataForgeIdentityProvider"
c.ServerApp.authorizer_class = "dataforge_jupyter.auth.DataForgeAuthorizer"
c.ServerApp.contents_manager_class = "dataforge_jupyter.auth.MemberContentsManager"
c.ServerApp.kernel_manager_class = "dataforge_jupyter.auth.MemberKernelManager"
c.IdentityProvider.token = ""  # no shared login token: members arrive through DataForge
c.IdentityProvider.cookie_options = {"samesite": "Lax", "httponly": True}
c.MappingKernelManager.cull_idle_timeout = 3600
c.MappingKernelManager.cull_interval = 300
c.MappingKernelManager.cull_connected = False

# The DataForge look: custom CSS, brand mark and favicon.
c.LabApp.custom_css = True
# LabApp fixes its name as a class attribute (not a configurable trait); the page title and header use it.
from jupyterlab.labapp import LabApp  # noqa: E402

LabApp.app_name = "DataForge Notebooks"
# Custom CSS is served from <JUPYTER_CONFIG_DIR>/custom; run.py installs dataforge_jupyter/custom there.
c.ServerApp.extra_static_paths = [str(PACKAGE / "static")]
c.LabApp.news_url = None
c.LabApp.check_for_updates_class = "jupyterlab.NeverCheckForUpdate"
c.LabApp.collaborative = False
