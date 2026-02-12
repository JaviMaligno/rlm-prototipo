import os

import pytest

from rlm.config import _normalize_endpoint, load_azure_config


def test_normalize_endpoint_trims_openai_suffix():
    endpoint = "https://example.openai.azure.com/openai/deployments/gpt-5"
    assert _normalize_endpoint(endpoint) == "https://example.openai.azure.com"


def test_normalize_endpoint_trailing_slash():
    endpoint = "https://example.openai.azure.com/"
    assert _normalize_endpoint(endpoint) == "https://example.openai.azure.com"


def test_load_azure_config_missing_env(monkeypatch):
    for key in list(os.environ.keys()):
        if key.startswith("AZURE_OPENAI_"):
            monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError):
        load_azure_config()


def test_load_azure_config_ok(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-5")
    monkeypatch.setenv("AZURE_OPENAI_MODEL_NAME", "gpt-5")
    monkeypatch.setenv("AZURE_OPENAI_SUBMODEL_NAME", "gpt-5-mini")

    cfg = load_azure_config()
    assert cfg.endpoint == "https://x.openai.azure.com"
    assert cfg.submodel_name == "gpt-5-mini"
