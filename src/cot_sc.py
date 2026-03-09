"""
cot_sc.py
=========
Top-level CoT-SC entry point.

run_cot_sc wires together sampler and voting:
- samples == 1 -> plain single invoke
- samples > 1 -> parallel sample + majority vote
"""

from .core.sampler import parallel_sample
from .core.voting import aggregate_votes
from .utils.console import print_box


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
        print_box(f"[CoT-SC] sampling {samples} paths (temp={temperature}) ...")

    candidates = parallel_sample(llm, messages, samples, temperature)
    return aggregate_votes(candidates, log=log)
