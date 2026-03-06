"""
config.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_api_settings() -> tuple[str, str, str]:
    """Return (api_key, base_url, model_name) from environment variables.

    Raises:
        EnvironmentError: If any required setting is missing.
    """
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "LLM_API_KEY is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )

    base_url = os.environ.get("LLM_BASE_URL", "")
    if not base_url:
        raise EnvironmentError(
            "LLM_BASE_URL is not set. "
            "Please specify it in your .env file."
        )

    model_name = os.environ.get("LLM_MODEL", "")
    if not model_name:
        raise EnvironmentError(
            "LLM_MODEL is not set. "
            "Specify the model identifier in your .env file."
        )

    return api_key, base_url, model_name


def get_cot_sc_settings() -> tuple[int, float]:
    """Return (samples, temperature) for the CoT-SC sampler.

    Returns:
        A tuple of (num_samples, temperature).

    Raises:
        EnvironmentError: If COT_SC_SAMPLES or COT_SC_TEMP are missing or invalid.
    """
    raw_samples = os.environ.get("COT_SC_SAMPLES", "")
    if not raw_samples:
        raise EnvironmentError("COT_SC_SAMPLES is not set in .env")

    raw_temp = os.environ.get("COT_SC_TEMP", "")
    if not raw_temp:
        raise EnvironmentError("COT_SC_TEMP is not set in .env")

    try:
        samples = int(raw_samples)
    except ValueError:
        raise EnvironmentError(f"COT_SC_SAMPLES must be an integer, got '{raw_samples}'")

    try:
        temperature = float(raw_temp)
    except ValueError:
        raise EnvironmentError(f"COT_SC_TEMP must be a float, got '{raw_temp}'")

    return samples, temperature
