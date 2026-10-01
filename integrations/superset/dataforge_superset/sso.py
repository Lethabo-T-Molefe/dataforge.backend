"""Sign in to Superset from DataForge, and serve DataForge's brand assets.

DataForge issues a short-lived token (two minutes, single use) signed with SSO_SHARED_SECRET. This view
checks it, creates or updates the matching Superset user with roles mapped from the DataForge role, signs
the user in and redirects inside Superset. Members never get a separate Superset password.
"""

import os
import secrets
import time
from pathlib import Path

import jwt
from flask import Blueprint, abort, redirect, request, send_from_directory
from flask_login import login_user

STATIC_DIR = Path(__file__).parent / "static"

blueprint = Blueprint("dataforge", __name__)

# DataForge role → Superset roles. "df_<role>" carries the row-level security rules DataForge syncs.
ROLE_MAP: dict[str, list[str]] = {
    "admin": ["Admin"],
    "steward": ["Alpha", "sql_lab"],
    "practitioner": ["Alpha", "sql_lab"],
    "curator": ["Alpha", "sql_lab"],
    "analyst": ["Alpha"],
    "executive": ["Gamma", "DataForge Viewer"],
}

_used: dict[str, float] = {}


def _remember(jti: str, expires: float) -> bool:
    """Single use: reject a token id seen before. Old entries are dropped as they expire."""
    now = time.time()
    for key in [k for k, v in _used.items() if v < now]:
        _used.pop(key, None)
    if jti in _used:
        return False
    _used[jti] = expires
    return True


def _safe_next(target: str | None) -> str:
    if not target or not target.startswith("/") or target.startswith("//"):
        return "/superset/welcome/"
    return target


@blueprint.route("/dataforge-static/<path:name>")
def static_asset(name: str):  # type: ignore[no-untyped-def]
    return send_from_directory(STATIC_DIR, name, max_age=86_400)


@blueprint.route("/dataforge/sso")
def sso():  # type: ignore[no-untyped-def]
    from superset import security_manager

    token = request.args.get("token", "")
    secret = os.environ.get("SSO_SHARED_SECRET", "")
    if not token or not secret:
        abort(401)
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            audience="superset",
            issuer=os.environ.get("DATAFORGE_JWT_ISSUER", "dataforge"),
            options={"require": ["exp", "sub", "aud", "jti"]},
        )
    except jwt.PyJWTError:
        abort(401)
    if not _remember(str(claims["jti"]), float(claims["exp"])):
        abort(401)
    role = str(claims.get("role", ""))
    if role not in ROLE_MAP:
        abort(403)

    username = f"df_{claims['sub']}"
    names = str(claims.get("name") or username).split(" ", 1)
    first, last = names[0], names[1] if len(names) > 1 else "-"
    roles = []
    for name in [*ROLE_MAP[role], f"df_{role}"]:
        found = security_manager.find_role(name)
        if found is None:
            found = security_manager.add_role(name)
        roles.append(found)

    user = security_manager.find_user(username=username)
    if user is None:
        user = security_manager.add_user(
            username=username,
            first_name=first,
            last_name=last,
            email=str(claims.get("email") or f"{username}@dataforge.local"),
            role=roles,
            # Unusable random password: these accounts only ever sign in through DataForge.
            password=secrets.token_urlsafe(48),
        )
        if not user:
            abort(500)
    else:
        user.first_name, user.last_name = first, last
        user.roles = roles
        user.active = True
        security_manager.update_user(user)
    login_user(user, remember=False)
    return redirect(_safe_next(request.args.get("next")))
