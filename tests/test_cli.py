import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from keke_code.cli import _run_repl
from keke_code.core import KekeAgent
from keke_code.llm import OpenAICompatibleClient
from keke_code.tools import ToolRegistry, Workspace


class CliReplTest(unittest.TestCase):
    def test_model_command_is_handled_locally(self) -> None:
        agent = KekeAgent(
            ToolRegistry(Workspace(".")),
            llm=OpenAICompatibleClient(api_key="test", base_url="https://example.com/v1", model="demo-model"),
        )
        output = io.StringIO()

        with patch("builtins.input", side_effect=["/model", "/exit"]), redirect_stdout(output):
            exit_code = _run_repl(agent, use_llm=True, show_steps=False)

        self.assertEqual(exit_code, 0)
        self.assertIn("base_url=https://example.com/v1", output.getvalue())
        self.assertIn("model=demo-model", output.getvalue())


if __name__ == "__main__":
    unittest.main()
