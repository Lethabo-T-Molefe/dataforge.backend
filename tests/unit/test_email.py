"""Email delivery through Resend, with a fake HTTP client: what is sent, and what never is."""

from typing import Any

import pytest

from catalogue_progress import marks
from dataforge_backend.core.config import settings
from dataforge_backend.core.errors import Unavailable
from dataforge_backend.infrastructure import email


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.text = '{"id": "test"}'


class _Client:
    calls: list[dict[str, Any]] = []
    status = 200

    def __init__(self, **_: Any) -> None:
        pass

    async def __aenter__(self) -> "_Client":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def request(self, method: str, url: str, **request: Any) -> _Response:
        _Client.calls.append({"url": url, "method": method, **request})
        return _Response(_Client.status)

    async def post(self, url: str, **request: Any) -> _Response:
        return await self.request("POST", url, **request)


@pytest.fixture
def resend(monkeypatch: pytest.MonkeyPatch) -> type[_Client]:
    _Client.calls, _Client.status = [], 200
    monkeypatch.setattr(settings, "RESEND_API_KEY", "re_test_key")
    monkeypatch.setattr(settings, "SMTP_FROM", "DataForge <no-reply@chopa.dev>")
    monkeypatch.setattr(email.httpx, "AsyncClient", _Client)
    return _Client


@marks("WK-12", "X-08")
async def test_resend_carries_the_message(resend: type[_Client]) -> None:
    await email.send_email("member@gmail.com", "Verify your DataForge email", "Hello", reply_to="a@b.org")
    call = resend.calls[0]
    assert call["url"] == "https://api.resend.com/emails"
    assert call["headers"] == {"Authorization": "Bearer re_test_key"}
    assert call["json"] == {
        "from": "DataForge <no-reply@chopa.dev>",
        "to": ["member@gmail.com"],
        "subject": "Verify your DataForge email",
        "text": "Hello",
        "reply_to": "a@b.org",
    }


@marks("WK-09")
async def test_addresses_that_cannot_receive_mail_are_skipped(resend: type[_Client]) -> None:
    await email.send_email("thandi@dataforge.local", "Alert", "Hello")
    assert resend.calls == []


@marks("WK-10")
async def test_a_refused_send_is_reported(resend: type[_Client]) -> None:
    resend.status = 403
    with pytest.raises(Unavailable):
        await email.send_email("member@gmail.com", "Alert", "Hello")
