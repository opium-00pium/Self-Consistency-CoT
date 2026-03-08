"""
sampler.py
==========
Parallel-sampling helper for CoT-SC.

Runs the same prompt through an LLM N times at a higher temperature to
obtain diverse reasoning paths, then returns all responses for voting.
"""

from langchain_openai import ChatOpenAI


def parallel_sample(
    base_llm,
    messages: list,
    n_samples: int,
    temperature: float,
) -> list:
    """Sample N independent responses from an LLM for the same prompt.

    A new LLM instance is created with the requested *temperature* so
    that diversity is injected only during sampling — the caller's base
    LLM settings remain unchanged.

    Args:
        base_llm: A LangChain ChatOpenAI instance from which model name,
            API key and base URL are copied.
        messages: The conversation messages to send for each path.
        n_samples: Number of independent paths to sample (must be >= 1).
        temperature: Sampling temperature; higher = more varied outputs.

    Returns:
        A list of *n_samples* LLM response objects.
    """
    # Build a high-temperature copy of the LLM for diverse sampling.
    # Only pass base_url when it is explicitly configured; str(None)
    # would become "None" and produce an invalid request URL.
    sampling_llm_kwargs = {
        "model": base_llm.model_name,
        "api_key": base_llm.openai_api_key.get_secret_value(),
        "temperature": temperature,
    }
    openai_api_base = getattr(base_llm, "openai_api_base", None)
    if openai_api_base is not None:
        sampling_llm_kwargs["base_url"] = str(openai_api_base)

    sampling_llm = ChatOpenAI(**sampling_llm_kwargs)

    # Propagate any tool bindings the base LLM already has
    if hasattr(base_llm, "kwargs") and "tools" in base_llm.kwargs:
        sampling_llm = sampling_llm.bind_tools(base_llm.kwargs["tools"])

    # Run all paths in a single batched call
    batch_inputs = [messages] * n_samples
    return sampling_llm.batch(batch_inputs)

