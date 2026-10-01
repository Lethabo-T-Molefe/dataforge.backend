"""Encryption for secrets stored at rest (connector credentials, MFA seeds)."""

import base64
import hashlib
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from .config import settings
from .errors import AppError


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    material = settings.SECRETS_KEY or settings.JWT_SECRET
    key = base64.urlsafe_b64encode(hashlib.sha256(material.encode("utf-8")).digest())
    return Fernet(key)


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise AppError("A stored secret could not be decrypted. Was SECRETS_KEY changed?") from exc
