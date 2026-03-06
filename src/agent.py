"""
agent.py
========
Top-level CoT-SC entry point.

``run_cot_sc`` is the public-facing function that wires together the
three sub-modules:  sampler → fingerprint (via voting) → result.

It mirrors the decision logic of the original agent step:
  - samples == 1  →  plain single invoke (no overhead)
  - samples  > 1  →  parallel sample + majority vote
"""

from .core.sampler import parallel_sample
from .core.voting import aggregate_votes


def run_cot_sc(
    llm,
    messages: list,
    *,
    samples: int = 3,
    temperature: float = 0.7,
    log: bool = True,
):

    if samples <= 1:
        return llm.invoke(messages)

    if log:
        print(f"[CoT-SC] sampling {samples} paths (temp={temperature}) …")

    candidates = parallel_sample(llm, messages, samples, temperature)
    return aggregate_votes(candidates, log=log)
