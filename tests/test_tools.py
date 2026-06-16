import tempfile
import unittest
from pathlib import Path

from keke_code.tools import ToolRegistry, Workspace


class ToolRegistryTest(unittest.TestCase):
    def test_workspace_blocks_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(Path(directory))
            tools = ToolRegistry(workspace)

            result = tools.call("read_file", {"path": "../secret.txt"})

            self.assertFalse(result.ok)
            self.assertIn("escapes workspace", result.output)


    def test_write_read_and_list_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolRegistry(Workspace(Path(directory)))

            write = tools.call("write_file", {"path": "notes/hello.txt", "content": "hello"})
            read = tools.call("read_file", {"path": "notes/hello.txt"})
            listing = tools.call("list_files", {"path": "notes"})

            self.assertTrue(write.ok)
            self.assertEqual(read.output, "hello")
            self.assertIn("notes/hello.txt", listing.output)

    def test_grep_tree_and_edit_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tools = ToolRegistry(Workspace(Path(directory)))
            tools.call("write_file", {"path": "src/app.py", "content": "print('hello')\n"})

            grep = tools.call("grep", {"pattern": "hello"})
            tree = tools.call("tree", {"path": "."})
            edit = tools.call("edit_file", {"path": "src/app.py", "old": "hello", "new": "keke"})
            read = tools.call("read_file", {"path": "src/app.py"})

            self.assertIn("src/app.py:1", grep.output)
            self.assertIn("src/", tree.output)
            self.assertTrue(edit.ok)
            self.assertIn("keke", read.output)


if __name__ == "__main__":
    unittest.main()
