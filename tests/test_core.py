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

    def test_local_mode_reports_model(self) -> None:
        agent = KekeAgent(ToolRegistry(Workspace(".")))

        response = agent.run("model", use_llm=False)

        self.assertIn("model=", response.answer)

    def test_local_mode_grep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("hello keke", encoding="utf-8")
            agent = KekeAgent(ToolRegistry(Workspace(root)))

            response = agent.run("grep keke", use_llm=False)

            self.assertIn("README.md:1", response.answer)


if __name__ == "__main__":
    unittest.main()
