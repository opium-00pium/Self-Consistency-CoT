"""
fingerprint.py
==============
Converts an LLM response into a stable, comparable string key.

A "fingerprint" collapses the infinite variety of LLM outputs into a
small set of equivalence classes so that majority voting can count which
reasoning path won.

Two response types are handled:
  - Tool / function calls  →  fingerprint encodes tool name + sorted args.
  - Plain text replies     →  all map to a single shared bucket "text".
"""

import json


def _canonicalize(value):
    """Recursively sort dict keys and list elements so that two
    semantically identical structures always produce the same JSON string.

    Args:
        value: Any Python object (dict, list, scalar).

    Returns:
        A new object with all dicts key-sorted and lists preserved.
    """
    if isinstance(value, dict):
        return {k: _canonicalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def compute_fingerprint(response) -> str:
    """Produce a stable string key for a single LLM response.

    Args:
        response: An LLM response object. Expected to optionally carry a
            ``.tool_calls`` attribute (list of dicts with ``"name"`` and
            ``"args"`` keys).

    Returns:
        A string such as ``"tool::calculator::{...}"`` for tool responses
        or ``"text"`` for plain conversational replies.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        call = tool_calls[0]
        canonical_args = _canonicalize(call.get("args", {}))
        args_str = json.dumps(canonical_args, sort_keys=True, ensure_ascii=False)
        return f"tool::{call['name']}::{args_str}"
    return "text"
