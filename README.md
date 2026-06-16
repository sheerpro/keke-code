# keke-code
本仓库用于学习大模型架构相关知识，从0构建一个agent。

`keke_code` 是一个教学版 Claude Code：它把“模型推理”和“本地工具执行”拆开，让你能从代码里学习 CLI Agent 的核心结构。

## 快速开始

```bash
python -m pip install -e .
keke_code --no-llm "ls"
keke_code --no-llm "read README.md"
```

如需接入真实大模型，配置 OpenAI 兼容接口：

```bash
cat > .keke_code_config.json <<'JSON'
{
  "api_key": "your-api-key",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o-mini"
}
JSON
keke_code "帮我阅读 README 并总结项目"
```

`.keke_code_config.json` 已被 `.gitignore` 忽略，适合放本机私有 API Key；命令行参数和环境变量仍然可用，并且优先级更高。

## 架构学习路线

- `keke_code/cli.py`：命令行入口，负责参数解析、单次任务和交互式 REPL。
- `keke_code/core.py`：Agent Loop，负责“模型输出动作 → 执行工具 → 回填观察结果”的循环。
- `keke_code/tools.py`：工具系统，负责限制工作区、读写文件、列目录和可选 Shell 执行。
- `keke_code/llm.py`：模型适配层，负责调用 OpenAI 兼容的 `chat/completions` API。
