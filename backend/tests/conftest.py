import pytest

from app import llm


@pytest.fixture(autouse=True)
def no_language_model(monkeypatch):
    """Tests check the written rules, so no hosted or local model is called."""
    monkeypatch.setattr(llm, "available", lambda: "")
