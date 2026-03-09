"""
voting.py
=========
Majority-vote aggregator for a list of LLM candidate responses.

Given N responses sampled independently, ``aggregate_votes`` counts how
many paths produced each unique fingerprint and returns the response that
represents the most-agreed-upon reasoning path.

Tie-breaking rules
------------------
1. If a strict majority exists (> N/2 votes), that winner is returned.
2. If no strict majority exists AND at least one plain-text response is
   present, the text response is returned as the conservative fallback.
3. Otherwise the most-voted response is returned regardless.
"""

from collections import Counter

from ..utils.console import print_box
from ..utils.fingerprint import compute_fingerprint


def aggregate_votes(candidates: list, log: bool = True):
    """Pick the consensus response from a pool of independently sampled
    LLM outputs.

    Args:
        candidates: A list of LLM response objects (AIMessage or similar).
        log: If True, print a short vote summary to stdout.

    Returns:
        The winning response, or ``None`` if *candidates* is empty.
    """
    if not candidates:
        return None

    # --- Step 1: assign each response to a fingerprint bucket ----------
    keys = []
    clusters = {}  # fingerprint -> best representative response

    for resp in candidates:
        key = compute_fingerprint(resp)
        keys.append(key)

        # Within each bucket keep the response with the most content
        prev = clusters.get(key)
        if prev is None or len(str(resp.content)) > len(str(prev.content)):
            clusters[key] = resp

    # --- Step 2: count votes -------------------------------------------
    tally = Counter(keys)
    top_key, top_votes = tally.most_common(1)[0]
    total = len(candidates)

    if log:
        breakdown = "  |  ".join(f"{k[:40]!r}: {v}" for k, v in tally.most_common())
        print_box(
            [
                f"[CoT-SC vote] {top_votes}/{total} -> {top_key[:50]!r}",
                f"breakdown: {breakdown}",
            ]
        )

    # --- Step 3: fallback when no strict majority ----------------------
    strict_majority = total // 2 + 1
    if top_votes < strict_majority:
        if "text" in clusters:
            if log:
                print_box("[CoT-SC vote] no clear majority - using plain-text fallback")
            return clusters["text"]

    return clusters[top_key]
