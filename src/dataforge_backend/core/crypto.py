"""Encryption for secrets stored at rest (connector credentials, MFA seeds) and for files kept off-site."""

import base64
import hashlib
import os
from functools import lru_cache

from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import settings
from .errors import AppError

_FILE_MARKER = b"DFE1"


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    material = settings.SECRETS_KEY or settings.JWT_SECRET
    key = base64.urlsafe_b64encode(hashlib.sha256(material.encode("utf-8")).digest())
    return Fernet(key)


def _key() -> bytes:
    return hashlib.sha256((settings.SECRETS_KEY or settings.JWT_SECRET).encode("utf-8")).digest()


def encrypt_bytes(data: bytes) -> bytes:
    """AES-256-GCM for files kept outside the platform. The output is the marker, a nonce and the ciphertext."""
    nonce = os.urandom(12)
    return _FILE_MARKER + nonce + AESGCM(_key()).encrypt(nonce, data, None)


def decrypt_bytes(blob: bytes) -> bytes:
    """Reverse encrypt_bytes. Content without the marker was stored unencrypted and is returned as it is."""
    if not blob.startswith(_FILE_MARKER):
        return blob
    start = len(_FILE_MARKER)
    try:
        return AESGCM(_key()).decrypt(blob[start : start + 12], blob[start + 12 :], None)
    except InvalidTag as exc:
        raise AppError("A stored file could not be decrypted. Was SECRETS_KEY changed?") from exc


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise AppError("A stored secret could not be decrypted. Was SECRETS_KEY changed?") from exc
