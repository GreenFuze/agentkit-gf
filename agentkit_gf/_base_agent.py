# agentkit_gf/_base_agent.py
from __future__ import annotations

import json
from typing import Any, Generic, List, Optional, Sequence, TypeVar

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

OutputT = TypeVar("OutputT")


class _UnlimitedDeps:
    """Custom deps class with unlimited usage limits."""
    def __init__(self):
        self.usage_limits = UsageLimits(request_limit=None)


class _LimitedDeps:
    """Custom deps class with configurable usage limits."""
    def __init__(self, request_limit: Optional[int] = None):
        self.usage_limits = UsageLimits(request_limit=request_limit)


class _BaseAgent(Agent[_UnlimitedDeps, OutputT], Generic[OutputT]):
    """
    Base agent that adds a persistent conversation transcript and forwards
    constructor parameters to pydantic_ai.Agent using correct typing.

    The transcript is a simple sequence of text blocks:
      - "USER:\\n<text>"
      - "ASSISTANT:\\n<text>"
      - "ASSISTANT JSON <label>:\\n<json>"
    """

    def __init__(
        self,
        model: str,
        *,
        system_prompt: Optional[str],
        output_type: Any,
        model_settings: Optional[ModelSettings] = None,
        tools: Optional[Sequence[Any]] = None,
        toolsets: Optional[Sequence[Any]] = None,
        builtin_tools: Optional[Sequence[Any]] = None,
        usage_limit: Optional[int] = None,
    ) -> None:
        if not model:
            raise ValueError("model must be provided")

        self._history: List[str] = []

        # Always pass sequences (never None) to satisfy Pyright typing
        tools_seq: Sequence[Any] = tuple(tools) if tools else ()
        toolsets_seq: Sequence[Any] = tuple(toolsets) if toolsets else ()
        builtin_tools_seq: Sequence[Any] = tuple(builtin_tools) if builtin_tools else ()

        # Store usage_limit and initialize request counter
        self._usage_limit = usage_limit
        self._request_count = 0

        # Always use unlimited deps since we'll handle limits ourselves
        deps_type = _UnlimitedDeps

        super().__init__(
            model,
            system_prompt=system_prompt if system_prompt is not None else "",
            tools=tools_seq,
            toolsets=toolsets_seq,
            builtin_tools=builtin_tools_seq,
            output_type=output_type,
            model_settings=model_settings,
            deps_type=deps_type,
        )

    # -------- Usage limit checking --------
    
    def _check_usage_limit(self) -> None:
        """Check if usage limit has been exceeded."""
        if self._usage_limit is not None:
            if self._request_count >= self._usage_limit:
                raise Exception(f"The next request would exceed the request_limit of {self._usage_limit}")
            self._request_count += 1

    # -------- Conversation history API (public) --------

    def reset_history(self) -> None:
        """Clear the conversation transcript."""
        self._history.clear()

    def export_history_blocks(self) -> Sequence[str]:
        """Return a snapshot of the transcript (immutable)."""
        return tuple(self._history)

    def export_history_text(self) -> str:
        """Return the transcript as a single prompt string."""
        return self._compose_history()

    # -------- Protected helpers for subclasses --------

    def _history_add_user(self, text: str) -> None:
        if not text or not isinstance(text, str):
            raise ValueError("user text must be a non-empty string")
        self._history.append(f"USER:\n{text}")

    def _history_add_assistant_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise ValueError("assistant text must be str")
        self._history.append(f"ASSISTANT:\n{text}")

    def _history_add_assistant_json(self, label: str, json_text: str) -> None:
        if not label or not isinstance(label, str):
            raise ValueError("label must be a non-empty string")

        try:
            parsed = json.loads(json_text)
        except Exception as e:
            raise ValueError(f"json_text must be valid JSON: {e}") from e

        stable = json.dumps(parsed, ensure_ascii=False)
        self._history.append(f"ASSISTANT JSON {label}:\n{stable}")

    def _compose_history(self) -> str:
        return "\n\n".join(self._history)
