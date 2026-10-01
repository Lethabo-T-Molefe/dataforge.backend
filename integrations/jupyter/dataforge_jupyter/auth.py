"""DataForge single sign-on, per-member access rules and per-member kernels for JupyterLab.

Sign-in: DataForge issues a two-minute, single-use token (audience "jupyter") signed with SSO_SHARED_SECRET.
It carries the member's id, name, role and a notebook token for the DataForge API. The identity provider
checks it, remembers the member in memory and sets Jupyter's login cookie.

Access: members work in members/<their id>/ and shared/. Other members' folders, and terminals, are refused.

Kernels: a kernel starts with the starting member's notebook token in DATAFORGE_TOKEN, and none of the
server's own secrets. Notebooks read data only through the DataForge API, so the member's permissions,
masking and row policies apply to every query. Kernels are tied to the member who started them.

This is a shared single-server deployment for a trusted pilot. Kernels run as the same operating-system
user; for hard isolation between members run JupyterHub with a container spawner (see README).
"""

import contextvars
import os
import posixpath
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

import jwt
from jupyter_server.auth.authorizer import Authorizer
from jupyter_server.auth.identity import IdentityProvider, User
from jupyter_server.services.contents.largefilemanager import AsyncLargeFileManager
from jupyter_server.services.kernels.kernelmanager import AsyncMappingKernelManager

SERVICE_USER = "dataforge-service"
CODE_ROLES = {"admin", "steward", "practitioner", "curator"}
MEMBERS_DIR = "members"
SHARED_DIR = "shared"
_SECRET_ENV = {"SSO_SHARED_SECRET", "JUPYTER_API_TOKEN", "JUPYTER_TOKEN"}


@dataclass
class Member:
    username: str
    name: str
    role: str
    notebook_token: str
    expires_at: float


MEMBERS: dict[str, Member] = {}
KERNEL_OWNERS: dict[str, str] = {}
_USED_JTIS: dict[str, float] = {}
current_username: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "dataforge_username", default=None
)


def _initials(name: str) -> str:
    return "".join(part[0].upper() for part in name.split()[:2] if part) or "DF"


def home_of(username: str) -> str:
    return f"{MEMBERS_DIR}/{username}"


class DataForgeIdentityProvider(IdentityProvider):  # type: ignore[misc]
    """Accept DataForge's SSO token once, then Jupyter's own login cookie."""

    def _decode(self, token: str) -> dict[str, Any] | None:
        secret = os.environ.get("SSO_SHARED_SECRET", "")
        if not secret:
            return None
        try:
            claims = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="jupyter",
                issuer=os.environ.get("DATAFORGE_JWT_ISSUER", "dataforge"),
                options={"require": ["exp", "sub", "aud", "jti"]},
            )
        except jwt.PyJWTError:
            return None
        now = time.time()
        for key in [k for k, v in _USED_JTIS.items() if v < now]:
            _USED_JTIS.pop(key, None)
        if claims["jti"] in _USED_JTIS:
            return None
        _USED_JTIS[str(claims["jti"])] = float(claims["exp"])
        return dict(claims)

    async def get_user_token(self, handler: Any) -> User | None:
        sso = handler.get_argument("dataforge_sso", None)
        if sso:
            claims = self._decode(sso)
            if claims is None or claims.get("role") not in CODE_ROLES:
                return None
            username = f"df_{claims['sub']}"
            name = str(claims.get("name") or username)
            MEMBERS[username] = Member(
                username=username,
                name=name,
                role=str(claims["role"]),
                notebook_token=str(claims.get("nbt") or ""),
                expires_at=float(claims.get("nbt_exp") or time.time() + 8 * 3600),
            )
            prepare_home(handler.settings.get("contents_manager"), username)
            return self._set(User(username=username, name=name, display_name=name, initials=_initials(name)))
        # The DataForge backend itself calls the REST API with the service token.
        service_token = os.environ.get("JUPYTER_API_TOKEN", "")
        header = handler.request.headers.get("Authorization", "")
        if service_token and header == f"token {service_token}":
            return self._set(
                User(username=SERVICE_USER, name="DataForge", display_name="DataForge", initials="DF")
            )
        return None

    def get_user_cookie(self, handler: Any) -> User | None:
        user = super().get_user_cookie(handler)
        if user is None:
            return None
        member = MEMBERS.get(user.username)
        # After a server restart, or when the notebook token has expired, sign in from DataForge again.
        if member is None or member.expires_at < time.time():
            return None
        return self._set(
            User(
                username=member.username,
                name=member.name,
                display_name=member.name,
                initials=_initials(member.name),
            )
        )

    def _set(self, user: User) -> User:
        current_username.set(user.username)
        return user


def prepare_home(contents: Any, username: str) -> None:
    """Create the member's folder and the shared folder on first sign-in."""
    if contents is None:
        return
    root = getattr(contents, "root_dir", None)
    if not root:
        return
    for folder in (os.path.join(root, MEMBERS_DIR, username), os.path.join(root, SHARED_DIR)):
        os.makedirs(folder, exist_ok=True)


def _api_path(handler: Any) -> str | None:
    """The contents path a request is about, relative to the server root."""
    base = handler.settings.get("base_url", "/").rstrip("/")
    path = unquote(handler.request.path)
    if base and path.startswith(base):
        path = path[len(base) :]
    for prefix in ("/api/contents/", "/files/", "/nbconvert/", "/api/contents"):
        if path.startswith(prefix):
            rest = path[len(prefix) :]
            if prefix == "/nbconvert/":
                rest = rest.partition("/")[2]
            return posixpath.normpath(rest.split("/checkpoints")[0].strip("/")) if rest.strip("/") else ""
    if (
        path.startswith("/api/sessions")
        and handler.request.method in ("POST", "PATCH")
        and handler.request.body
    ):
        import json

        try:
            body = json.loads(handler.request.body)
        except ValueError:
            return None
        target = body.get("path") or (body.get("notebook") or {}).get("path") or ""
        return posixpath.normpath(str(target).strip("/")) if target else ""
    return None


def path_allowed(username: str, path: str, action: str) -> bool:
    if path in ("", "."):
        return action == "read"
    if path.startswith(".."):
        return False
    parts = path.split("/")
    if parts[0] == SHARED_DIR:
        return True
    if parts[0] == MEMBERS_DIR:
        if len(parts) == 1:
            return action == "read"
        return parts[1] == username
    # Anything outside members/ and shared/ is read-only (for example a README placed by an admin).
    return action == "read"


class DataForgeAuthorizer(Authorizer):  # type: ignore[misc]
    def is_authorized(self, handler: Any, user: User, action: str, resource: str) -> bool:
        if user.username == SERVICE_USER:
            return True
        member = MEMBERS.get(user.username)
        if member is None:
            return False
        if resource == "terminals":
            # A shell would reach the server's files and environment directly.
            return False
        if member.role == "admin" and resource in ("contents", "files", "nbconvert"):
            return True
        if resource in ("contents", "files", "nbconvert", "sessions"):
            path = _api_path(handler)
            if path is not None and not path_allowed(user.username, path, action):
                return False
        if resource == "kernels":
            parts = handler.request.path.rstrip("/").split("/")
            if "kernels" in parts:
                index = parts.index("kernels")
                kernel_id = parts[index + 1] if index + 1 < len(parts) else None
                if kernel_id and KERNEL_OWNERS.get(kernel_id) not in (None, user.username):
                    return False
        return True


class MemberContentsManager(AsyncLargeFileManager):  # type: ignore[misc]
    """Lists only the current member's folder under members/."""

    async def get(
        self,
        path: str,
        content: bool = True,
        type: str | None = None,
        format: str | None = None,
        require_hash: bool = False,
    ) -> dict[str, Any]:  # noqa: A002
        model: dict[str, Any] = await super().get(
            path, content=content, type=type, format=format, require_hash=require_hash
        )
        username = current_username.get()
        if (
            username
            and username != SERVICE_USER
            and path.strip("/") == MEMBERS_DIR
            and isinstance(model.get("content"), list)
        ):
            model["content"] = [item for item in model["content"] if item.get("name") == username]
        return model


class MemberKernelManager(AsyncMappingKernelManager):  # type: ignore[misc]
    """Starts each kernel with its member's DataForge notebook token and without the server's secrets."""

    async def _async_start_kernel(
        self, *, kernel_id: str | None = None, path: str | None = None, **kwargs: Any
    ) -> str:
        username = current_username.get()
        member = MEMBERS.get(username or "")
        if kernel_id is None or kernel_id not in self:
            env = {k: v for k, v in os.environ.items() if k not in _SECRET_ENV}
            env["DATAFORGE_API_URL"] = os.environ.get("DATAFORGE_API_URL", "http://localhost:8000")
            env["DATAFORGE_TOKEN"] = member.notebook_token if member else ""
            env["DATAFORGE_MEMBER"] = member.name if member else ""
            kwargs["env"] = env
        started: str = await super()._async_start_kernel(kernel_id=kernel_id, path=path, **kwargs)
        KERNEL_OWNERS.setdefault(started, username or SERVICE_USER)
        return started

    start_kernel = _async_start_kernel
