import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AzureConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment_name: str
    model_name: str
    submodel_name: str


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return endpoint
    # Azure SDK expects the base resource URL. Trim any /openai/... suffix.
    marker = "/openai/"
    if marker in endpoint:
        endpoint = endpoint.split(marker, 1)[0]
    return endpoint.rstrip("/")


def load_azure_config() -> AzureConfig:
    endpoint = _normalize_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", deployment_name)
    submodel_name = os.getenv("AZURE_OPENAI_SUBMODEL_NAME", model_name)

    missing = []
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not api_version:
        missing.append("AZURE_OPENAI_API_VERSION")
    if not deployment_name:
        missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    return AzureConfig(
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        deployment_name=deployment_name,
        model_name=model_name or deployment_name,
        submodel_name=submodel_name or model_name or deployment_name,
    )
