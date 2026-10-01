from datetime import timedelta

import pyotp
import pytest

from dataforge_backend.core.crypto import decrypt_secret, encrypt_secret
from dataforge_backend.core.errors import Unauthorized
from dataforge_backend.core.security import (
    create_token,
    decode_token,
    hash_password,
    new_totp_secret,
    verify_password,
    verify_totp,
)


def test_passwords_are_hashed_with_argon2() -> None:
    hashed = hash_password("Correct-horse-9")
    assert hashed.startswith("$argon2id$")
    assert verify_password("Correct-horse-9", hashed)
    assert not verify_password("wrong", hashed)


def test_tokens_are_typed_and_expire() -> None:
    token = create_token("u1", "access", expires_in=timedelta(minutes=5), role="analyst")
    claims = decode_token(token, "access")
    assert claims.subject == "u1" and claims.role == "analyst"
    with pytest.raises(Unauthorized):
        decode_token(token, "mfa")
    assert decode_token(token, ("access", "notebook")).token_type == "access"
    expired = create_token("u1", "access", expires_in=timedelta(seconds=-5))
    with pytest.raises(Unauthorized):
        decode_token(expired, "access")


def test_totp_and_secret_encryption() -> None:
    secret = new_totp_secret()
    assert verify_totp(secret, pyotp.TOTP(secret).now())
    assert not verify_totp(secret, "000000") or verify_totp(secret, "000000") is False
    assert decrypt_secret(encrypt_secret("s3cret")) == "s3cret"
