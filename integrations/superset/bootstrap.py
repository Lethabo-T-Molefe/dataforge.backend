"""Prepare Superset for DataForge. Idempotent, safe to run on every deploy.

    .venv/Scripts/python bootstrap.py      (Windows)
    .venv/bin/python bootstrap.py          (Linux, macOS)

Steps: migrate Superset's metadata database, create the default roles and permissions, create the service
account DataForge's backend uses for its REST calls, and create the DataForge Viewer and Guest roles.
"""

import os
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
    os.environ.setdefault("SUPERSET_CONFIG_PATH", str(HERE / "superset_config.py"))
    os.environ.setdefault("SUPERSET_HOME", str(HERE / "superset_home"))
    os.environ.setdefault("FLASK_APP", "superset")


def superset_exe() -> str:
    return str(Path(sys.executable).with_name("superset.exe" if os.name == "nt" else "superset"))


def superset(*args: str) -> None:
    shown = ["••••" if i > 0 and args[i - 1] == "--password" else a for i, a in enumerate(args)]
    print("+ superset", *shown, flush=True)
    subprocess.run([superset_exe(), *args], check=True, env=os.environ)


def ensure_roles() -> None:
    from superset.app import create_app

    app = create_app()
    with app.app_context():
        from superset import db, security_manager

        gamma = security_manager.find_role("Gamma")
        access = security_manager.find_permission_view_menu("all_datasource_access", "all_datasource_access")
        for name in ("DataForge Viewer", "DataForge Guest"):
            role = security_manager.find_role(name) or security_manager.add_role(name)
            perms = set(role.permissions)
            if name == "DataForge Guest" and gamma is not None:
                # Guests (embedded dashboards) read what Gamma reads; nothing is editable.
                perms |= {p for p in gamma.permissions if p.permission.name.startswith("can_read")}
            if access is not None:
                perms.add(access)
            role.permissions = list(perms)
        for dataforge_role in ("admin", "steward", "practitioner", "curator", "analyst", "executive"):
            if security_manager.find_role(f"df_{dataforge_role}") is None:
                security_manager.add_role(f"df_{dataforge_role}")
        db.session.commit()
        print("roles ready", flush=True)


def main() -> None:
    load_env()
    superset("db", "upgrade")
    superset("init")
    user = os.environ["SUPERSET_ADMIN_USER"]
    superset(
        "fab", "create-admin",
        "--username", user,
        "--firstname", "DataForge",
        "--lastname", "Service",
        "--email", f"{user}@dataforge.local",
        "--password", os.environ["SUPERSET_ADMIN_PASSWORD"],
    )  # fmt: skip
    ensure_roles()


if __name__ == "__main__":
    main()
