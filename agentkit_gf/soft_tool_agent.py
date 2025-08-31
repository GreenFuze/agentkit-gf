# agentkit_gf/soft_tool_agent.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field, NonNegativeInt
from pydantic_ai import PromptedOutput
from pydantic_ai.settings import ModelSettings

from ._base_agent import _BaseAgent


# ============================ Envelope & payload schemas ============================

class OpenResponse(BaseModel):
    kind: Literal["OPEN_RESPONSE"] = "OPEN_RESPONSE"
    text: str
    confidence: Optional[int] = None  # 0..100 if provided


class ToolCall(BaseModel):
    kind: Literal["TOOL_CALL"] = "TOOL_CALL"
    tool: str
    args_json: str
    reason: str

    def model_post_init(self, _ctx: Any) -> None:
        if not self.tool or not isinstance(self.tool, str):
            raise ValueError("tool must be a non-empty string")
        try:
            obj = json.loads(self.args_json)
        except Exception as e:
            raise ValueError(f"args_json must be valid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError("args_json must encode a JSON object")
        if not self.reason or len(self.reason.strip()) < 8:
            raise ValueError("reason must describe the missing fact or purpose (>= 8 chars)")


class ToolResult(BaseModel):
    kind: Literal["TOOL_RESULT"] = "TOOL_RESULT"
    tool: str
    args_json: str
    result_json: str
    success: bool
    note: Optional[str] = None

    def model_post_init(self, _ctx: Any) -> None:
        if not self.tool or not isinstance(self.tool, str):
            raise ValueError("tool must be a non-empty string")
        for name in ("args_json", "result_json"):
            val = getattr(self, name)
            try:
                json.loads(val)
            except Exception as e:
                raise ValueError(f"{name} must be valid JSON text: {e}") from e


Payload = Union[OpenResponse, ToolCall, ToolResult]


class Envelope(BaseModel):
    message: Payload = Field(
        ...,
        description="One of OPEN_RESPONSE | TOOL_CALL | TOOL_RESULT.",
    )


# ============================ Run results ============================

class ExecutedTool(BaseModel):
    tool: str
    args_json: str
    result_json: str
    success: bool
    error_message: Optional[str] = None


class SoftRunResult(BaseModel):
    final_text: str
    steps: Sequence[ExecutedTool] = ()
    total_steps: NonNegativeInt


# ============================ SoftToolAgent (true soft loop, schema-driven) ============================

class SoftToolAgent(_BaseAgent[Envelope]):
    """
    True soft-tool agent with a single JSON-Schema envelope for output.
    Maintains an internal transcript between runs.
    """

    def __init__(
        self,
        *,
        model: str,
        system_prompt: Optional[str] = None,
        allow_llm_tool_result: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model must be provided (e.g., 'openai:gpt-5-nano')")

        self._allow_llm_tool_result = bool(allow_llm_tool_result)

        policy = (
            "You are a reasoning-first assistant.\n"
            "Always return exactly ONE JSON object matching the provided schema (Envelope).\n\n"
            "Envelope.message kinds:\n"
            "- OPEN_RESPONSE { text, confidence? }\n"
            "- TOOL_CALL { tool, args_json, reason }\n"
            "- TOOL_RESULT { tool, args_json, result_json, success, note? }\n\n"
            "Policy:\n"
            "1) Prefer OPEN_RESPONSE when you can answer from context.\n"
            "2) Emit TOOL_CALL only if you can name the missing fact and the precise operation.\n"
            "3) Never fabricate TOOL_RESULT; the host provides it after executing a tool.\n"
            "4) Be concise and specific."
        )

        spec: PromptedOutput[Envelope] = PromptedOutput(
            Envelope,
            name="SoftToolEnvelope",
            description="Return exactly one Envelope JSON object; no extra text.",
        )

        super().__init__(
            model,
            system_prompt=system_prompt or policy,
            output_type=spec,
            model_settings=None,  # Don't set parallel_tool_calls when no tools are provided
        )

    def run_soft_sync(
        self,
        prompt: str,
        tool_registry: Mapping[str, Callable[..., Any]],
        *,
        max_steps: int = 4,
    ) -> SoftRunResult:
        """
        Run a soft-tool interaction synchronously and persist conversation.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(tool_registry, Mapping):
            raise ValueError("tool_registry must be a Mapping[str, Callable]")

        # Persist user message
        self._history_add_user(prompt)

        executed: list[ExecutedTool] = []

        for _ in range(max_steps):
            res = super().run_sync(self._compose_history())
            env = res.output
            msg = env.message

            # Persist the assistant decision envelope
            self._history_add_assistant_json("ENVELOPE", env.model_dump_json(exclude_none=True))

            if isinstance(msg, OpenResponse):
                self._history_add_assistant_text(msg.text)
                return SoftRunResult(final_text=msg.text, steps=executed, total_steps=len(executed))

            if isinstance(msg, ToolResult):
                if not self._allow_llm_tool_result:
                    raise ValueError(
                        "LLM emitted TOOL_RESULT, but allow_llm_tool_result=False. "
                        "Models must not fabricate tool results."
                    )
                continue

            if isinstance(msg, ToolCall):
                tool_name, args = self._parse_tool_call(msg)

                tool_fn = tool_registry.get(tool_name)
                if tool_fn is None:
                    raise ValueError(f"Unknown tool requested: {tool_name!r}. Available: {sorted(tool_registry.keys())}")
                if not callable(tool_fn):
                    raise TypeError(f"Registered tool '{tool_name}' is not callable")

                success, result_json, error_message = self._execute_tool(tool_fn, args)

                executed.append(
                    ExecutedTool(
                        tool=tool_name,
                        args_json=json.dumps(args, ensure_ascii=False),
                        result_json=result_json,
                        success=success,
                        error_message=error_message,
                    )
                )

                tool_result_env = Envelope(
                    message=ToolResult(
                        tool=tool_name,
                        args_json=json.dumps(args, ensure_ascii=False),
                        result_json=result_json,
                        success=success,
                        note=error_message,
                    )
                )
                self._history_add_assistant_json("ENVELOPE", tool_result_env.model_dump_json(exclude_none=True))
                continue

            raise ValueError(f"Unsupported envelope payload: {type(msg).__name__}")

        raise TimeoutError(f"max_steps={max_steps} reached without an OPEN_RESPONSE; steps: {len(executed)}")

    # -------- Private helpers --------

    @staticmethod
    def _parse_tool_call(decision: ToolCall) -> Tuple[str, Mapping[str, Any]]:
        try:
            args_obj = json.loads(decision.args_json)
        except Exception as e:
            raise ValueError(f"Invalid args_json for tool '{decision.tool}': {e}") from e
        if not isinstance(args_obj, dict):
            raise ValueError(f"args_json for tool '{decision.tool}' must encode a JSON object")
        return decision.tool, args_obj

    @staticmethod
    def _execute_tool(tool_fn: Callable[..., Any], args: Mapping[str, Any]) -> Tuple[bool, str, Optional[str]]:
        try:
            result = tool_fn(**args)
        except Exception as e:
            return False, json.dumps({"error": str(e)}, ensure_ascii=False), str(e)

        try:
            if isinstance(result, str):
                return True, json.dumps({"text": result}, ensure_ascii=False), None
            if isinstance(result, (int, float, bool)) or result is None:
                return True, json.dumps({"value": result}, ensure_ascii=False), None
            if isinstance(result, dict):
                return True, json.dumps(result, ensure_ascii=False), None
            if isinstance(result, BaseModel):
                return True, result.model_dump_json(), None
            return True, json.dumps({"repr": repr(result)}, ensure_ascii=False), None
        except Exception as e:
            return True, json.dumps({"string": str(result)}, ensure_ascii=False), f"non-JSON result normalized: {e}"
