import tempfile
import unittest
from pathlib import Path

from keke_code.core import KekeAgent
from keke_code.tools import ToolRegistry, Workspace


class KekeAgentTest(unittest.TestCase):
    def test_local_mode_lists_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("demo", encoding="utf-8")
            agent = KekeAgent(ToolRegistry(Workspace(root)))

            response = agent.run("ls", use_llm=False)

            self.assertIn("README.md", response.answer)
            self.assertEqual(response.steps[0].tool, "list_files")


    def test_local_mode_reads_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("demo", encoding="utf-8")
            agent = KekeAgent(ToolRegistry(Workspace(root)))

            response = agent.run("read README.md", use_llm=False)

            self.assertEqual(response.answer, "demo")


if __name__ == "__main__":
    unittest.main()
