"""Command line interface for keke_code."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Optional

from . import __version__
from .core import AgentResponse, KekeAgent
from .llm import LLMError, OpenAICompatibleClient
from .tools import ToolRegistry, Workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keke_code",
        description="A learning-first Claude Code-style terminal coding agent.",
    )
    parser.add_argument("prompt", nargs="*", help="Task to run once. Omit it to start interactive chat.")
    parser.add_argument("--workspace", default=".", help="Workspace root. Defaults to current directory.")
    parser.add_argument("--model", default=None, help="OpenAI-compatible model name. Also reads KEKE_CODE_MODEL.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL. Also reads KEKE_CODE_BASE_URL.")
    parser.add_argument("--api-key", default=None, help="API key. Prefer KEKE_CODE_API_KEY or OPENAI_API_KEY.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum tool calls per prompt.")
    parser.add_argument("--allow-shell", action="store_true", help="Allow the agent to execute shell commands.")
    parser.add_argument("--no-llm", action="store_true", help="Use deterministic local teaching mode even if an API key exists.")
    parser.add_argument("--show-steps", action="store_true", help="Print tool calls and observations.")
    parser.add_argument("--version", action="version", version=f"keke_code {__version__}")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Workspace(Path(args.workspace))
    tools = ToolRegistry(workspace, allow_shell=args.allow_shell)
    llm = OpenAICompatibleClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
    agent = KekeAgent(tools=tools, llm=llm, max_steps=args.max_steps)

    if args.prompt:
        prompt = " ".join(args.prompt)
        return _run_once(agent, prompt, use_llm=not args.no_llm, show_steps=args.show_steps)
    return _run_repl(agent, use_llm=not args.no_llm, show_steps=args.show_steps)


def _run_once(agent: KekeAgent, prompt: str, use_llm: bool, show_steps: bool) -> int:
    try:
        response = agent.run(prompt, use_llm=use_llm)
    except LLMError as exc:
        print(f"模型调用失败：{exc}", file=sys.stderr)
        print("提示：可使用 --no-llm 进入本地教学模式，或配置 KEKE_CODE_API_KEY。", file=sys.stderr)
        return 2
    _print_response(response, show_steps=show_steps)
    return 0


def _run_repl(agent: KekeAgent, use_llm: bool, show_steps: bool) -> int:
    print("keke_code 已启动。输入 /exit 退出，/help 查看本地指令。")
    while True:
        try:
            prompt = input("keke_code> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt in {"/exit", "exit", "quit"}:
            return 0
        if prompt == "/help":
            print("指令：直接输入任务；或在 --no-llm 模式使用 pwd、ls、read <path>、run <command>。")
            continue
        exit_code = _run_once(agent, prompt, use_llm=use_llm, show_steps=show_steps)
        if exit_code != 0:
            return exit_code


def _print_response(response: AgentResponse, show_steps: bool) -> None:
    if show_steps and response.steps:
        for index, step in enumerate(response.steps, start=1):
            status = "ok" if step.result.ok else "error"
            print(f"[step {index}] {step.tool} {step.args} -> {status}")
            print(step.result.output)
    print(response.answer)


if __name__ == "__main__":
    raise SystemExit(main())
