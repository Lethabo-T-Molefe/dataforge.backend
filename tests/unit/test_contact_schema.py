from pydantic import ValidationError
import pytest

from dataforge_backend.presentation.api.schemas import ContactIn


def test_contact_rejects_header_injection() -> None:
    with pytest.raises(ValidationError):
        ContactIn(name="Amina\nBcc: evil@x.com", email="amina@org.example", message="Hello there friend")
    with pytest.raises(ValidationError):
        ContactIn(name="Amina", email="amina@org.example", organisation="Org\r\nX-Inject: 1", message="Hello there friend")
    ok = ContactIn(name="Amina Patel", email="amina@org.example", organisation="DataForge", message="Hello there friend")
    assert ok.name == "Amina Patel"
