import tempfile
import unittest
from pathlib import Path

from keke_code.llm import load_local_config


class LocalConfigTest(unittest.TestCase):
    def test_load_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / ".keke_code_config.json"
            config.write_text('{"api_key":"secret","base_url":"https://example.com/v1","model":"demo"}', encoding="utf-8")

            data = load_local_config(config)

            self.assertEqual(data["api_key"], "secret")
            self.assertEqual(data["base_url"], "https://example.com/v1")
            self.assertEqual(data["model"], "demo")


if __name__ == "__main__":
    unittest.main()
