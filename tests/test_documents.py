"""Guards for the published prompts and README."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepseek_prompt import load_system_prompt


class DocumentTests(unittest.TestCase):
    def test_files_are_utf8_and_newline_terminated(self) -> None:
        names = [
            "README.md",
            "DEEPSEEK-ENHANCE.md",
            "DEEPSEEK-API-SYSTEM-PROMPT-FINAL.md",
            "DEEPSEEK-API-SYSTEM-PROMPT-FINAL-USAGE.md",
            "deepseek_prompt.py",
        ]
        for name in names:
            path = ROOT / name
            data = path.read_bytes()
            data.decode("utf-8")
            self.assertFalse(data.startswith(b"\xef\xbb\xbf"), name)
            self.assertTrue(data.endswith(b"\n"), name)
            self.assertNotIn("â€".encode(), data, name)
            self.assertNotIn(b"\xef\xbf\xbd", data, name)

    def test_short_prompt_keeps_its_behavior_rules(self) -> None:
        text = load_system_prompt("final")
        for sentence in (
            "Prioritize correctness, clarity, and efficient execution.",
            "Match effort to task complexity.",
            "Do not invent current facts, API behavior, product limits, runtime results, or tool outputs.",
            "If JSON is requested, return valid JSON only.",
            "Optimize for useful output per token.",
        ):
            self.assertIn(sentence, text)

    def test_long_prompt_matches_the_current_api(self) -> None:
        text = load_system_prompt("enhance")
        required = [
            "deepseek-flash",
            "DeepSeek-V4.1-Flash-1017",
            "DeepSeek-V4-Pro-0813",
            "2026-07-24",
            "https://api.deepseek.com/beta",
            "https://api.deepseek.com/v1/tools/search",
            "https://api.deepseek.com/v1/tools/fetch",
            "prompt_cache_hit_tokens",
            "additionalProperties",
            "0.95",
            "$0.022",
            "01:00–04:00",
            "| xhigh | high |",
            "HTTP 400",
        ]
        for snippet in required:
            self.assertIn(snippet, text, snippet)
        self.assertNotIn("$0.435", text)
        self.assertNotIn("$0.87", text)
        self.assertNotIn("antml:cite", text)
        self.assertNotIn("dangerously-skip-permissions", text)
        self.assertNotIn("5-6x", text)
        self.assertNotIn("mapped to `high`; `xhigh` is mapped to `max`", text)

    def test_readme_describes_the_loader_and_not_the_old_claude_flag(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("deepseek_prompt.py", readme)
        self.assertIn("DEEPSEEK_API_KEY", readme)
        self.assertIn("python -m unittest discover -s tests -v", readme)
        self.assertNotIn("dangerously-skip-permissions", readme)
        self.assertNotIn("<your-api-key>", readme)
        self.assertNotIn("antml:cite", readme)


if __name__ == "__main__":
    unittest.main()
