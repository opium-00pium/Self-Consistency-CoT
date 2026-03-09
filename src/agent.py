"""
agent.py
========
LangGraph wrapper agent for running CoT-SC with a shared AGENT.md system prompt.

Direct run mode:
    python agent.py
"""

import sys
import time
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# Support both package import (from src.agent import ...) and
# direct script execution from the src directory (python agent.py).
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.cot_sc import run_cot_sc
    from src.utils.config import get_api_settings, get_cot_sc_settings
    from src.utils.console import print_box
else:
    from .cot_sc import run_cot_sc
    from .utils.config import get_api_settings, get_cot_sc_settings
    from .utils.console import print_box


class CotScAgentState(TypedDict, total=False):
    query: str
    messages: list
    response: object


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stream_print(prefix: str, text: str, delay: float = 0.01) -> None:
    """Print text progressively to mimic streaming output in terminal."""
    print(prefix, end="", flush=True)
    for ch in text:
        print(ch, end="", flush=True)
        if delay > 0:
            time.sleep(delay)
    print("\n")


def load_system_prompt(system_prompt_path: str | Path | None = None) -> str:
    """Load the AGENT.md system prompt.

    Args:
        system_prompt_path: Optional custom prompt file path.
            Defaults to <repo_root>/AGENT.md.
    """
    prompt_path = Path(system_prompt_path) if system_prompt_path else _repo_root() / "AGENT.md"
    return prompt_path.read_text(encoding="utf-8").strip()


def build_default_llm() -> ChatOpenAI:
    """Build a ChatOpenAI instance from project env settings."""
    api_key, base_url, model_name = get_api_settings()
    return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)


class CotScLangGraphAgent:
    """Minimal LangGraph agent that runs one CoT-SC call per query."""

    def __init__(
        self,
        llm: ChatOpenAI,
        *,
        samples: int,
        temperature: float,
        log: bool = True,
        system_prompt: str | None = None,
        system_prompt_path: str | Path | None = None,
    ) -> None:
        self.llm = llm
        self.samples = samples
        self.temperature = temperature
        self.log = log
        self.system_prompt = system_prompt or load_system_prompt(system_prompt_path)
        self._app = self._build_graph()

    def _prepare_messages(self, state: CotScAgentState) -> CotScAgentState:
        query = state["query"]
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query),
        ]
        return {"messages": messages}

    def _run_cotsc(self, state: CotScAgentState) -> CotScAgentState:
        response = run_cot_sc(
            self.llm,
            state["messages"],
            samples=self.samples,
            temperature=self.temperature,
            log=self.log,
        )
        return {"response": response}

    def _build_graph(self):
        graph = StateGraph(CotScAgentState)
        graph.add_node("prepare_messages", self._prepare_messages)
        graph.add_node("run_cotsc", self._run_cotsc)

        graph.add_edge(START, "prepare_messages")
        graph.add_edge("prepare_messages", "run_cotsc")
        graph.add_edge("run_cotsc", END)
        return graph.compile()

    def invoke(self, query: str):
        """Return the raw LLM response object from CoT-SC."""
        result = self._app.invoke({"query": query})
        return result.get("response")

    def invoke_text(self, query: str) -> str:
        """Return response content as plain text."""
        response = self.invoke(query)
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content
        return str(content)


def create_cotsc_langgraph_agent(
    *,
    log: bool = True,
    system_prompt_path: str | Path | None = None,
) -> CotScLangGraphAgent:
    """Factory using env-based LLM config and env-based CoT-SC settings."""
    llm = build_default_llm()
    samples, temperature = get_cot_sc_settings()
    return CotScLangGraphAgent(
        llm,
        samples=samples,
        temperature=temperature,
        log=log,
        system_prompt_path=system_prompt_path,
    )


def main() -> None:
    samples, temperature = get_cot_sc_settings()
    agent = create_cotsc_langgraph_agent(log=True)

    print_box(
        f"CoT-SC agent ready (samples={samples}, temperature={temperature}). "
        "Type 'quit' or 'exit' to stop."
    )

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q"}:
            break

        try:
            answer = agent.invoke_text(user_text)
            _stream_print("agent> ", answer)
        except Exception as exc:
            print(f"[error] {exc}\n")


if __name__ == "__main__":
    main()
