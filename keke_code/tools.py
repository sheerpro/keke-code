"""Sandboxed tools that the agent can call.

核心原理：Claude Code 这类产品不是让模型直接“拥有电脑”，而是把电脑能力封装成
一组可审计的工具。模型只负责决定下一步调用哪个工具；程序负责校验路径、执行动作、
截断输出并把结果反馈给模型。
"""

from __future__ import annotations

import dataclasses
import os
import shlex
import subprocess
from pathlib import Path
from typing import Callable, Optional, Union


MAX_TOOL_OUTPUT = 12_000


class ToolError(RuntimeError):
    """Raised when a tool call is invalid or unsafe."""


@dataclasses.dataclass(frozen=True)
class ToolResult:
    """Normalized result returned to the agent after every tool call."""

    ok: bool
    output: str


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """Description passed to the model so it knows how to call a tool."""

    name: str
    description: str
    parameters: dict[str, str]


class Workspace:
    """Restricts file access to one project root."""

    def __init__(self, root: Union[Path, str]) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise ToolError(f"workspace does not exist: {self.root}")

    def resolve(self, path: Optional[str] = None) -> Path:
        candidate = self.root if not path else (self.root / path).expanduser().resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolError(f"path escapes workspace: {path}") from exc
        return candidate


class ToolRegistry:
    """Maps JSON tool calls from the LLM to Python functions."""

    def __init__(self, workspace: Workspace, allow_shell: bool = False) -> None:
        self.workspace = workspace
        self.allow_shell = allow_shell
        self._tools: dict[str, Callable[..., ToolResult]] = {
            "pwd": self.pwd,
            "list_files": self.list_files,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "run_shell": self.run_shell,
        }

    @property
    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("pwd", "Return the current workspace root.", {}),
            ToolSpec("list_files", "List files under a relative directory.", {"path": "relative directory, default ."}),
            ToolSpec("read_file", "Read a UTF-8 text file from the workspace.", {"path": "relative file path"}),
            ToolSpec(
                "write_file",
                "Create or overwrite a UTF-8 text file inside the workspace.",
                {"path": "relative file path", "content": "full new file content"},
            ),
            ToolSpec(
                "run_shell",
                "Run a shell command in the workspace when --allow-shell is enabled.",
                {"command": "command line", "timeout": "seconds, optional"},
            ),
        ]

    def call(self, name: str, args: Optional[dict[str, object]] = None) -> ToolResult:
        if name not in self._tools:
            return ToolResult(False, f"unknown tool: {name}")
        try:
            return self._tools[name](**(args or {}))
        except TypeError as exc:
            return ToolResult(False, f"invalid arguments for {name}: {exc}")
        except (OSError, UnicodeError, ToolError, subprocess.SubprocessError) as exc:
            return ToolResult(False, str(exc))

    def pwd(self) -> ToolResult:
        return ToolResult(True, str(self.workspace.root))

    def list_files(self, path: str = ".") -> ToolResult:
        root = self.workspace.resolve(path)
        if not root.exists():
            return ToolResult(False, f"not found: {path}")
        if root.is_file():
            return ToolResult(True, str(root.relative_to(self.workspace.root)))

        entries: list[str] = []
        for child in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name in {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}:
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.relative_to(self.workspace.root)}{suffix}")
        return ToolResult(True, "\n".join(entries) or "(empty)")

    def read_file(self, path: str) -> ToolResult:
        file_path = self.workspace.resolve(path)
        if not file_path.is_file():
            return ToolResult(False, f"not a file: {path}")
        content = file_path.read_text(encoding="utf-8")
        return ToolResult(True, _truncate(content))

    def write_file(self, path: str, content: str) -> ToolResult:
        file_path = self.workspace.resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return ToolResult(True, f"wrote {file_path.relative_to(self.workspace.root)} ({len(content)} bytes)")

    def run_shell(self, command: str, timeout: int = 30) -> ToolResult:
        if not self.allow_shell:
            return ToolResult(False, "shell is disabled; restart with --allow-shell to enable it")
        if not command.strip():
            return ToolResult(False, "empty command")
        completed = subprocess.run(
            command,
            cwd=self.workspace.root,
            shell=True,
            text=True,
            capture_output=True,
            timeout=max(1, min(int(timeout), 120)),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        output = completed.stdout
        if completed.stderr:
            output = f"{output}\n[stderr]\n{completed.stderr}".strip()
        header = f"exit_code={completed.returncode}\ncommand={shlex.quote(command)}\n"
        return ToolResult(completed.returncode == 0, header + _truncate(output))

    def render_specs(self) -> str:
        lines = []
        for spec in self.specs:
            params = ", ".join(f"{k}: {v}" for k, v in spec.parameters.items()) or "no parameters"
            lines.append(f"- {spec.name}: {spec.description} Parameters: {params}")
        return "\n".join(lines)


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
