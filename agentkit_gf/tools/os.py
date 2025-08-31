# tools/os.py
from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from pydantic import BaseModel, NonNegativeInt


class ProcessResult(BaseModel):
    """
    Strongly-typed result for process execution.
    Convert to JSON with `.model_dump()` at the tool boundary.
    """
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: NonNegativeInt
    timed_out: bool = False
    cwd: Optional[str] = None


def _as_text(val: Optional[Union[str, bytes]]) -> str:
    """
    Normalize provider/stdlib values (str | bytes | None) into str.
    This avoids Pyright issues when concatenating/formatting messages.
    """
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return val


@dataclass(frozen=True)
class _ProcessPolicy:
    """
    Execution policy and helpers:
      - root_cwd: all executions must occur at or below this directory (if set)
      - allowed_basenames: restrict set of executable basenames (if set)
      - max_output_bytes: clip stdout/stderr to bounded size
    """
    root_cwd: Optional[Path]
    allowed_basenames: Optional[set[str]]
    max_output_bytes: int

    def check_cwd(self, cwd: Optional[str]) -> Path:
        """
        Resolve and validate the working directory against the root.
        Fail fast on escape attempts.
        """
        if cwd is None:
            return self.root_cwd if self.root_cwd is not None else Path.cwd().resolve()

        p = Path(cwd).resolve(strict=False)

        if self.root_cwd is None:
            return p

        try:
            _ = p.resolve().relative_to(self.root_cwd.resolve())
        except Exception as e:
            raise PermissionError(f"cwd {p} is outside allowed root {self.root_cwd}") from e

        return p

    def ensure_allowed(self, argv0: str) -> None:
        """
        Enforce the allowed-basenames policy on the executable basename.
        """
        if self.allowed_basenames is None:
            return

        base = Path(argv0).name
        if base not in self.allowed_basenames:
            raise PermissionError(
                f"command '{base}' is not in allowed set: {sorted(self.allowed_basenames)}"
            )

    def clip(self, s: str) -> str:
        """
        Clip UTF-8 text to max_output_bytes, preserving encoding.
        """
        b = s.encode("utf-8", errors="replace")
        if len(b) <= self.max_output_bytes:
            return s

        return b[: self.max_output_bytes].decode("utf-8", errors="replace") + "\n[...truncated...]"


class ProcessTools:
    """
    OS/process toolset with optional execution policy.

    Public methods are intended to be exposed as LLM tools.
    They return Pydantic models for strong typing; call `.model_dump()` at the boundary.
    """

    def __init__(
        self,
        *,
        root_cwd: Optional[str] = None,
        allowed_basenames: Optional[Sequence[str]] = None,
        max_output_bytes: int = 32_768,
    ) -> None:
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be > 0")

        self._policy = _ProcessPolicy(
            root_cwd=Path(root_cwd).resolve() if root_cwd else None,
            allowed_basenames=set(allowed_basenames) if allowed_basenames else None,
            max_output_bytes=max_output_bytes,
        )

    # ----- Public Tool Methods -------------------------------------------------

    def run_process(
        self,
        argv: Sequence[str],
        timeout_sec: int = 10,
        cwd: Optional[str] = None,
    ) -> dict:
        """
        Run a process without a shell (recommended). `argv` is the full argument vector.

        Returns:
            dict representation of ProcessResult
        """
        if not argv or not isinstance(argv, (list, tuple)):
            raise ValueError("argv must be a non-empty list/tuple of arguments")

        if any(not isinstance(a, str) or a == "" for a in argv):
            raise ValueError("all argv elements must be non-empty strings")

        self._policy.ensure_allowed(argv[0])

        resolved_cwd = self._policy.check_cwd(cwd)

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                list(argv),
                shell=False,
                cwd=str(resolved_cwd),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )

            duration_ms = int((time.perf_counter() - start) * 1000)

            return ProcessResult(
                exit_code=proc.returncode,
                stdout=self._policy.clip(proc.stdout),
                stderr=self._policy.clip(proc.stderr),
                duration_ms=duration_ms,
                timed_out=False,
                cwd=str(resolved_cwd),
            ).model_dump()

        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.perf_counter() - start) * 1000)

            out = _as_text(e.stdout)
            err = _as_text(e.stderr) + f"\n[timeout after {timeout_sec}s]"

            return ProcessResult(
                exit_code=-1,
                stdout=self._policy.clip(out),
                stderr=self._policy.clip(err),
                duration_ms=duration_ms,
                timed_out=True,
                cwd=str(resolved_cwd),
            ).model_dump()

    def run_shell(
        self,
        command: str,
        timeout_sec: int = 10,
        cwd: Optional[str] = None,
    ) -> dict:
        """
        Run a shell command (more flexible but riskier).
        Prefer run_process(argv) when possible.

        Returns:
            dict representation of ProcessResult
        """
        if not command or not isinstance(command, str):
            raise ValueError("command must be a non-empty string")

        # Best-effort basename extraction for allowlist checks
        argv0 = ""
        try:
            parts = shlex.split(command, posix=True)
            if parts:
                argv0 = parts[0]
        except ValueError:
            # If the command cannot be parsed, let subprocess handle it.
            # We skip allowlist enforcement in this edge case.
            argv0 = ""

        if argv0:
            self._policy.ensure_allowed(argv0)

        resolved_cwd = self._policy.check_cwd(cwd)

        start = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(resolved_cwd),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )

            duration_ms = int((time.perf_counter() - start) * 1000)

            return ProcessResult(
                exit_code=proc.returncode,
                stdout=self._policy.clip(proc.stdout),
                stderr=self._policy.clip(proc.stderr),
                duration_ms=duration_ms,
                timed_out=False,
                cwd=str(resolved_cwd),
            ).model_dump()

        except subprocess.TimeoutExpired as e:
            duration_ms = int((time.perf_counter() - start) * 1000)

            out = _as_text(e.stdout)
            err = _as_text(e.stderr) + f"\n[timeout after {timeout_sec}s]"

            return ProcessResult(
                exit_code=-1,
                stdout=self._policy.clip(out),
                stderr=self._policy.clip(err),
                duration_ms=duration_ms,
                timed_out=True,
                cwd=str(resolved_cwd),
            ).model_dump()
