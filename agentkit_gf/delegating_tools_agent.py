# agentkit_gf/delegating_tools_agent.py
from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Iterable, Optional, Sequence

from pydantic_ai import RunContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.builtin_tools import CodeExecutionTool, UrlContextTool, WebSearchTool
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import AbstractToolset, CombinedToolset, FunctionToolset
from pydantic_ai.tools import Tool

from ._base_agent import _BaseAgent
from .tools.builtin_tools_matrix import (
    BuiltinTool,
    provider_key_from_model,
    validate_builtin_tools,
)

# ---------- helpers: turn callables / objects into a Toolset ----------

def _build_function_toolset_from_callables(
    funcs: Iterable[Callable[..., Any]],
    *,
    prefix: Optional[str] = None,
) -> FunctionToolset:
    ts = FunctionToolset()

    for f in funcs:
        if not callable(f):
            raise TypeError(f"Non-callable provided in tool_sources: {f!r}")

        name = f.__name__
        if not name:
            raise ValueError("Tool function must have a name")

        if prefix:
            name = f"{prefix}_{name}"

        ts.add_tool(Tool(function=f, name=name))

    return ts


def _build_function_toolset_from_object(
    obj: Any,
    *,
    prefix: Optional[str] = None,
) -> FunctionToolset:
    funcs: list[Callable[..., Any]] = []

    for attr in dir(obj):
        if attr.startswith("_"):
            continue

        member = getattr(obj, attr)
        if callable(member):
            try:
                inspect.signature(member)
            except Exception as e:
                raise TypeError(f"Public attribute '{attr}' on {obj!r} is not a valid callable: {e}") from e
            funcs.append(member)

    if not funcs:
        raise ValueError(f"No public methods found to expose as tools on {obj!r}")

    return _build_function_toolset_from_callables(funcs, prefix=prefix)


def _build_combined_toolset(
    sources: Sequence[Callable[..., Any] | Any | AbstractToolset],
    *,
    class_prefix: Optional[str] = None,
) -> AbstractToolset:
    toolsets: list[AbstractToolset] = []
    pending_funcs: list[Callable[..., Any]] = []

    for s in sources:
        if isinstance(s, AbstractToolset):
            toolsets.append(s)
        elif callable(s):
            pending_funcs.append(s)
        else:
            toolsets.append(_build_function_toolset_from_object(s, prefix=class_prefix))

    if pending_funcs:
        toolsets.append(_build_function_toolset_from_callables(pending_funcs))

    if not toolsets:
        return FunctionToolset()

    return toolsets[0] if len(toolsets) == 1 else CombinedToolset(toolsets)


# ---------- internal executor agent ----------

class _ToolExecutorAgent(_BaseAgent[str]):
    """
    Internal agent that owns real tools and provider built-ins.
    We keep a local history for debugging/inspection; it is not sent to the main agent.
    """

    def __init__(
        self,
        *,
        model: str,
        toolset: AbstractToolset,
        builtin_enums: Sequence[BuiltinTool],
        system_prompt: Optional[str] = None,
        model_settings: Optional[ModelSettings] = None,
        usage_limit: Optional[int] = None,
    ):
        if builtin_enums is None:
            raise ValueError("builtin_enums must be provided (use [] if no builtins)")

        provider_key = provider_key_from_model(model)
        validated = validate_builtin_tools(provider_key, builtin_enums)

        builtin_tools_instances: list[Any] = []
        for t in validated:
            if t is BuiltinTool.WEB_SEARCH:
                builtin_tools_instances.append(WebSearchTool(max_uses=1, search_context_size="medium"))
            elif t is BuiltinTool.CODE_EXECUTION:
                builtin_tools_instances.append(CodeExecutionTool())
            elif t is BuiltinTool.URL_CONTEXT:
                builtin_tools_instances.append(UrlContextTool())
            else:
                raise ValueError(f"Unhandled BuiltinTool enum: {t!r}")

        # Merge user model_settings with default settings
        default_settings = ModelSettings(parallel_tool_calls=False)
        if model_settings:
            # Merge user settings with defaults
            merged_settings = ModelSettings(default_settings)
            merged_settings.update(model_settings)
        else:
            merged_settings = default_settings

        super().__init__(
            model,
            system_prompt=system_prompt
            or "Ops agent. Execute exactly one requested operation using your tools and return concise results.",
            tools=None,
            toolsets=[toolset],
            builtin_tools=builtin_tools_instances or None,
            output_type=str,  # native string output
            model_settings=merged_settings,
            usage_limit=usage_limit,
        )

    # Note: Usage limits are handled by pydantic_ai directly, no custom overrides needed


# ---------- public delegating agent ----------

class DelegatingToolsAgent(_BaseAgent[str]):
    """
    Main agent with a single gateway tool (delegate_ops). All real tools live on the internal executor.
    Maintains an internal transcript between runs and logs delegated tool calls/results.
    """

    def __init__(
        self,
        *,
        model: str,
        builtin_enums: Sequence[BuiltinTool],
        tool_sources: Sequence[Callable[..., Any] | Any | AbstractToolset] = (),
        class_prefix: Optional[str] = None,
        system_prompt: Optional[str] = None,
        ops_system_prompt: Optional[str] = None,
        model_settings: Optional[ModelSettings] = None,
        real_time_log_user: bool = False,
        real_time_log_agent: bool = False,
        usage_limit: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        if builtin_enums is None:
            raise ValueError("builtin_enums must be provided (use [] if no builtins)")

        # Store logging parameters
        self._real_time_log_user = real_time_log_user
        self._real_time_log_agent = real_time_log_agent

        toolset = _build_combined_toolset(tool_sources, class_prefix=class_prefix)

        # Merge user model_settings with default settings
        default_settings = ModelSettings(parallel_tool_calls=False)
        if model_settings:
            # Merge user settings with defaults
            merged_settings = ModelSettings(default_settings)
            merged_settings.update(model_settings)
        else:
            merged_settings = default_settings
        
        # Handle temperature parameter
        if temperature is not None:
            merged_settings["temperature"] = temperature

        self._ops = _ToolExecutorAgent(
            model=model,
            toolset=toolset,
            builtin_enums=builtin_enums,
            system_prompt=ops_system_prompt,
            model_settings=merged_settings,
            usage_limit=usage_limit,
        )

        super().__init__(
            model,
            system_prompt=system_prompt
            or ("Use delegate_ops to access files and execute commands when needed. "
                "Provide a brief justification for each tool use."),
            tools=[self.delegate_ops],
            toolsets=None,
            builtin_tools=None,
            output_type=str,
            model_settings=merged_settings,
            usage_limit=usage_limit,
        )

    # ---- Override run_sync to manage history automatically ----
    def run_sync(self, prompt: str, *args: Any, **kwargs: Any):  # type: ignore[override]
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        # Real-time user input logging
        if self._real_time_log_user:
            print("[USER] USER INPUT:")
            print(prompt)
            print("=" * 80)

        self._history_add_user(prompt)
        composed = self._compose_history()

        res = super().run_sync(composed, *args, **kwargs)

        # Real-time agent response logging
        if self._real_time_log_agent:
            print("[AGENT] AGENT RESPONSE:")
            print(res.output)
            print("=" * 80)

        self._history_add_assistant_text(res.output)
        return res

    # ---- Override run to manage history automatically ----
    async def run(self, prompt: str, *args: Any, **kwargs: Any):  # type: ignore[override]
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        # Real-time user input logging
        if self._real_time_log_user:
            print("[USER] USER INPUT:")
            print(prompt)
            print("=" * 80)

        self._history_add_user(prompt)
        composed = self._compose_history()

        res = await super().run(composed, *args, **kwargs)

        # Real-time agent response logging
        if self._real_time_log_agent:
            print("[AGENT] AGENT RESPONSE:")
            print(res.output)
            print("=" * 80)

        self._history_add_assistant_text(res.output)
        return res

    # ---- Gateway tool (logs TOOL_CALL/TOOL_RESULT into history) ----
    async def delegate_ops(
        self,
        ctx: RunContext[None],
        tool: str,
        args_json: str,
        why: str,
    ) -> str:
        """
        Gateway to the ops agent.

        Args:
          tool: exact tool name on the executor (e.g., 'read_text', 'web_search', 'MyCls_method')
          args_json: JSON object of arguments for that tool.
          why: concrete justification; must include 'because'
        """
        if not tool or not isinstance(tool, str):
            raise ModelRetry("Invalid 'tool' (expected non-empty string).")

        if len(why.split()) < 3:
            raise ModelRetry(
                "Provide a brief justification for using this tool."
            )

        try:
            # Sanitize and validate JSON before parsing
            sanitized_json = self._sanitize_tool_args(args_json)
            args_obj = json.loads(sanitized_json) if sanitized_json else {}
            if not isinstance(args_obj, dict):
                raise ValueError("args_json must encode a JSON object")
        except json.JSONDecodeError as e:
            # Try JSON recovery before giving up
            try:
                recovered_json = self._recover_json(args_json)
                args_obj = json.loads(recovered_json)
                if not isinstance(args_obj, dict):
                    raise ValueError("Recovered JSON is not an object")
            except:
                # Enhanced JSON error handling for better debugging
                error_msg = f"JSON parsing error in tool '{tool}': {e}. "
                error_msg += f"Problematic JSON (first 200 chars): {args_json[:200] if args_json else 'None'}"
                error_msg += f"\nAttempted recovery but failed. Please provide valid JSON."
                raise ModelRetry(error_msg)
        except Exception as e:
            raise ModelRetry(f"args_json must be a JSON object for tool '{tool}'. Error: {e}")

        # Log TOOL_CALL into history
        self._history_add_assistant_json(
            "TOOL_CALL",
            json.dumps({"tool": tool, "args": args_obj, "why": why}, ensure_ascii=False),
        )

        prompt = (
            "Execute exactly ONE tool call and return only its result.\n"
            f"- TOOL: {tool}\n"
            f"- ARGS: {json.dumps(args_obj, ensure_ascii=False)}\n"
            f"- WHY: {why}\n"
            "Do not call multiple tools. Keep output concise."
        )

        try:
            r = await self._ops.run(prompt, usage=None)  # Don't pass main agent's usage context
        except Exception as e:
            # Handle path discovery errors - give LLM another chance
            if "not a directory" in str(e) or "not a file" in str(e):
                error_msg = f"PATH DISCOVERY ERROR: {e}\n\n"
                error_msg += "The path you tried to access is incorrect. Please:\n"
                error_msg += "1. Use 'list_dir' to see what directories exist first\n"
                error_msg += "2. Use 'file_exists' to check if a path is a file or directory\n"
                error_msg += "3. Use 'is_directory' to verify if a path is a directory before listing it\n"
                error_msg += "4. Use 'is_file' to verify if a path is a file before reading it\n\n"
                error_msg += "Try again with the correct path."
                
                # Log the error and retry instruction
                self._history_add_assistant_json(
                    "TOOL_ERROR",
                    json.dumps({"tool": tool, "args": args_obj, "error": str(e), "retry_instruction": error_msg}, ensure_ascii=False),
                )
                
                return error_msg
            elif "file not found" in str(e) and "CMakeLists.txt" in str(e):
                # Handle missing CMakeLists.txt files gracefully
                error_msg = f"CMakeLists.txt not found in this directory.\n\n"
                error_msg += "This is normal - not all directories have CMakeLists.txt files.\n"
                error_msg += "Continue exploring the directory structure using 'list_dir' to see what files are available.\n"
                error_msg += "You can still analyze the source files and components without CMakeLists.txt."
                
                # Log the error and continue instruction
                self._history_add_assistant_json(
                    "TOOL_ERROR",
                    json.dumps({"tool": tool, "args": args_obj, "error": str(e), "continue_instruction": error_msg}, ensure_ascii=False),
                )
                
                return error_msg
            else:
                # Re-raise other errors
                raise

        # Log TOOL_RESULT into history
        self._history_add_assistant_json(
            "TOOL_RESULT",
            json.dumps({"tool": tool, "args": args_obj, "result": r.output}, ensure_ascii=False),
        )

        return r.output
    
    def _sanitize_tool_args(self, args_json: str) -> str:
        """Sanitize tool arguments to prevent JSON parsing errors."""
        if not args_json:
            return "{}"
        
        # Remove common problematic characters that cause JSON parsing issues
        sanitized = args_json.strip()
        
        # Fix common JSON issues
        # 1. Remove trailing commas
        sanitized = sanitized.replace(',}', '}').replace(',]', ']')
        
        # 2. Ensure proper string escaping for paths
        # This is a basic fix - more sophisticated escaping could be added
        
        # 3. Truncate extremely long arguments to prevent context overflow
        if len(sanitized) > 1000:
            # Try to find a good truncation point
            truncate_at = sanitized.rfind(',', 0, 1000)
            if truncate_at > 500:
                sanitized = sanitized[:truncate_at] + '}'
            else:
                sanitized = sanitized[:1000] + '}'
        
        return sanitized
    
    def _recover_json(self, json_str: str) -> str:
        """Attempt to recover malformed JSON using common fixes."""
        if not json_str:
            return "{}"
        
        recovered = json_str.strip()
        
        # Fix common JSON issues
        # 1. Remove trailing commas
        recovered = recovered.replace(',}', '}').replace(',]', ']')
        
        # 2. Fix unescaped quotes in strings
        # This is a basic fix - more sophisticated escaping could be added
        
        # 3. Ensure proper object/array structure
        if not recovered.startswith('{') and not recovered.startswith('['):
            # Try to find the first complete object
            start_idx = recovered.find('{')
            if start_idx != -1:
                recovered = recovered[start_idx:]
        
        # 4. Try to balance braces
        open_braces = recovered.count('{')
        close_braces = recovered.count('}')
        if open_braces > close_braces:
            recovered += '}' * (open_braces - close_braces)
        elif close_braces > open_braces:
            # Remove extra closing braces from the end
            extra = close_braces - open_braces
            for _ in range(extra):
                if recovered.endswith('}'):
                    recovered = recovered[:-1]
        
        return recovered
