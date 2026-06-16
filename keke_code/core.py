"""Agent loop for keke_code.

核心原理：Claude Code 的心脏是 Agent Loop：
1. 收集用户目标、仓库上下文、可用工具说明。
2. 让模型输出结构化动作 JSON。
3. 程序执行工具并把观察结果追加回上下文。
4. 循环直到模型给出 final answer 或达到步数上限。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from typing import Optional

from .llm import ChatMessage, LLMError, OpenAICompatibleClient
from .tools import ToolRegistry, ToolResult


SYSTEM_PROMPT = """You are keke_code, a Claude Code-style terminal coding agent.
You help users understand, edit, test, and explain code in the current workspace.
You are not Claude Code itself; you are an educational reimplementation named keke_code.

Core behavior:
- Be an autonomous senior pair-programmer.
- Before answering questions about files or code, inspect the workspace with tools.
- Prefer small, reviewable edits: read_file -> edit_file/apply_patch -> git_diff -> final.
- Never invent file contents, command outputs, model names, or repository state.
- Explain important architecture principles in Chinese when useful.

Return exactly one JSON object per turn, with one of these shapes:
{"tool":"tool_name","args":{...},"thought":"short reason"}
{"final":"answer to user"}

For identity/model questions, answer with the configured provider/model if visible from context.
When writing files, prefer edit_file or apply_patch. Keep final answers concise and in Chinese.
"""


@dataclass
class AgentStep:
    tool: str
    args: dict[str, object]
    thought: str
    result: ToolResult


@dataclass
class AgentResponse:
    answer: str
    steps: list[AgentStep]


class KekeAgent:
    """Coordinates model reasoning and deterministic tool execution."""

    def __init__(self, tools: ToolRegistry, llm: Optional[OpenAICompatibleClient] = None, max_steps: int = 8) -> None:
        self.tools = tools
        self.llm = llm or OpenAICompatibleClient()
        self.max_steps = max_steps

    def run(self, user_input: str, use_llm: bool = True) -> AgentResponse:
        if not use_llm or not self.llm.configured:
            return self._run_local(user_input)

        messages = [
            ChatMessage("system", self._build_system_prompt()),
            ChatMessage("user", user_input),
        ]
        steps: list[AgentStep] = []
        for _ in range(self.max_steps):
            raw = self.llm.chat(messages)
            action = _parse_json_object(raw)
            if "final" in action:
                return AgentResponse(str(action["final"]), steps)
            tool = str(action.get("tool", ""))
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            thought = str(action.get("thought", ""))
            result = self.tools.call(tool, args)
            steps.append(AgentStep(tool=tool, args=args, thought=thought, result=result))
            messages.append(ChatMessage("assistant", json.dumps(action, ensure_ascii=False)))
            messages.append(
                ChatMessage(
                    "user",
                    json.dumps({"observation": {"ok": result.ok, "output": result.output}}, ensure_ascii=False),
                )
            )
        return AgentResponse("达到最大工具调用步数，请把目标拆小后重试。", steps)

    def _build_system_prompt(self) -> str:
        context = (
            f"Workspace: {self.tools.workspace.root}\n"
            f"Provider base_url: {self.llm.base_url}\n"
            f"Configured model: {self.llm.model}\n"
        )
        return SYSTEM_PROMPT + "\n\nRuntime context:\n" + context + "\nAvailable tools:\n" + self.tools.render_specs()

    def _run_local(self, user_input: str) -> AgentResponse:
        """Deterministic fallback used for learning, demos, and tests without an API key."""

        text = user_input.strip()
        lowered = text.lower()

        if lowered in {"model", "/model", "模型", "当前模型"}:
            answer = f"base_url={self.llm.base_url}\nmodel={self.llm.model}\nconfigured={self.llm.configured}"
            return AgentResponse(answer, [])

        if lowered in {"pwd", "where am i", "当前目录"}:
            result = self.tools.call("pwd")
            return AgentResponse(result.output, [AgentStep("pwd", {}, "查看工作区根目录", result)])

        if lowered in {"ls", "list", "list files", "列文件", "列出文件"}:
            result = self.tools.call("list_files", {"path": "."})
            return AgentResponse(result.output, [AgentStep("list_files", {"path": "."}, "列出工作区文件", result)])

        if lowered in {"tree", "目录树"}:
            result = self.tools.call("tree", {"path": "."})
            return AgentResponse(result.output, [AgentStep("tree", {"path": "."}, "查看目录树", result)])

        if lowered in {"status", "git status", "状态"}:
            result = self.tools.call("git_status")
            return AgentResponse(result.output, [AgentStep("git_status", {}, "查看 Git 状态", result)])

        if lowered in {"diff", "git diff"}:
            result = self.tools.call("git_diff")
            return AgentResponse(result.output, [AgentStep("git_diff", {}, "查看 Git diff", result)])

        read_match = re.match(r"^(?:read|cat|打开|读取)\s+(.+)$", text, re.IGNORECASE)
        if read_match:
            path = read_match.group(1).strip()
            result = self.tools.call("read_file", {"path": path})
            return AgentResponse(result.output, [AgentStep("read_file", {"path": path}, "读取文件内容", result)])

        run_match = re.match(r"^(?:run|执行)\s+(.+)$", text, re.IGNORECASE)
        if run_match:
            command = run_match.group(1).strip()
            result = self.tools.call("run_shell", {"command": command})
            return AgentResponse(result.output, [AgentStep("run_shell", {"command": command}, "执行命令", result)])

        grep_match = re.match(r"^(?:grep|搜索)\s+(.+)$", text, re.IGNORECASE)
        if grep_match:
            pattern = grep_match.group(1).strip()
            result = self.tools.call("grep", {"pattern": pattern})
            return AgentResponse(result.output, [AgentStep("grep", {"pattern": pattern}, "搜索文件内容", result)])

        help_text = (
            "当前未配置大模型 API，因此运行在本地教学模式。\n"
            "可用指令：pwd、ls、tree、read <path>、grep <text>、status、diff、run <command>。\n"
            "如需完整 Agent 推理，请设置 KEKE_CODE_API_KEY，可选设置 KEKE_CODE_BASE_URL 和 KEKE_CODE_MODEL。"
        )
        return AgentResponse(help_text, [])


def _parse_json_object(text: str) -> dict[str, object]:
    """Extract the first JSON object from a model response."""

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise LLMError(f"model did not return JSON: {text}")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise LLMError(f"model JSON must be an object: {data}")
    return data
