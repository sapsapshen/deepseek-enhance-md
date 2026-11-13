"""Offline tests for deepseek_prompt.py."""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepseek_prompt import (
    PromptError,
    append_assistant_turn,
    build_chat_kwargs,
    build_messages,
    client_settings_from_env,
    collect_stream_text,
    effective_reasoning_effort,
    explain_status,
    load_system_prompt,
    main,
    recommend_model,
)


def temporary_directory(test: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp(prefix="deepseek-prompt-"))
    test.addCleanup(shutil.rmtree, path, True)
    return path


class LoadPromptTests(unittest.TestCase):
    def test_loads_both_published_prompts(self) -> None:
        final = load_system_prompt("final")
        enhance = load_system_prompt(" enhance ")
        self.assertIn("Do not invent current facts", final)
        self.assertIn("deepseek-flash", enhance)
        self.assertGreater(len(enhance), len(final))

    def test_unknown_prompt_names_the_choices(self) -> None:
        with self.assertRaises(PromptError) as caught:
            load_system_prompt("both")
        self.assertIn("final", str(caught.exception))
        self.assertIn("enhance", str(caught.exception))

    def test_missing_root_and_empty_file(self) -> None:
        missing = Path(__file__).resolve().parent / "does-not-exist"
        with self.assertRaises(PromptError):
            load_system_prompt(root=missing)

        empty_root = temporary_directory(self)
        with self.assertRaises(PromptError) as caught:
            load_system_prompt("final", root=empty_root)
        self.assertIn("not found", str(caught.exception))

    def test_rejects_secret_shaped_text(self) -> None:
        root = temporary_directory(self)
        (root / "DEEPSEEK-API-SYSTEM-PROMPT-FINAL.md").write_text(
            "You are helpful. sk-" + ("a" * 20),
            encoding="utf-8",
        )
        with self.assertRaises(PromptError) as caught:
            load_system_prompt("final", root=root)
        self.assertIn("API key", str(caught.exception))
        self.assertNotIn("a" * 20, str(caught.exception))

    def test_rejects_invalid_utf8(self) -> None:
        root = temporary_directory(self)
        (root / "DEEPSEEK-API-SYSTEM-PROMPT-FINAL.md").write_bytes(b"\xff\xfe ok")
        with self.assertRaises(PromptError) as caught:
            load_system_prompt("final", root=root)
        self.assertIn("UTF-8", str(caught.exception))


class RequestTests(unittest.TestCase):
    def test_messages_append_extra_rules_once(self) -> None:
        messages = build_messages("  Explain caching.  ", extra_system="Use short answers.")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "Explain caching.")
        self.assertIn("Use short answers.", messages[0]["content"])
        self.assertEqual(messages[0]["content"].count("Use short answers."), 1)

    def test_blank_user_text_is_rejected(self) -> None:
        with self.assertRaises(PromptError):
            build_messages("   ")

    def test_model_recommendation_covers_vision(self) -> None:
        self.assertEqual(
            recommend_model(complex_reasoning=False, needs_vision=False),
            "deepseek-flash",
        )
        self.assertEqual(
            recommend_model(complex_reasoning=True, needs_vision=False),
            "deepseek-v4-pro",
        )
        self.assertEqual(
            recommend_model(complex_reasoning=False, needs_vision=True),
            "deepseek-flash",
        )
        with self.assertRaises(PromptError) as caught:
            recommend_model(complex_reasoning=True, needs_vision=True)
        self.assertIn("deepseek-v4-pro", str(caught.exception))

    def test_effort_mapping_matches_the_service(self) -> None:
        expected = {
            "minimal": "low",
            "low": "low",
            "medium": "high",
            "high": "high",
            "xhigh": "high",
            "max": "max",
            "ultra": "max",
        }
        for requested, actual in expected.items():
            self.assertEqual(effective_reasoning_effort(f" {requested.upper()} "), actual)
        with self.assertRaises(PromptError):
            effective_reasoning_effort("extreme")

    def test_chat_kwargs_omit_the_key_and_disable_sampling_passthrough(self) -> None:
        kwargs = build_chat_kwargs(
            "Return json matching {\"ok\": true}",
            model="deepseek-v4-pro",
            thinking=False,
            response_format={"type": "json_object"},
        )
        self.assertEqual(kwargs["model"], "deepseek-v4-pro")
        self.assertNotIn("api_key", kwargs)
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertEqual(kwargs["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_legacy_alias_is_accepted_and_retired_names_are_not(self) -> None:
        kwargs = build_chat_kwargs("Hello", model="deepseek-v4-flash")
        self.assertEqual(kwargs["model"], "deepseek-v4-flash")
        with self.assertRaises(PromptError) as caught:
            build_chat_kwargs("Hello", model="deepseek-chat")
        self.assertIn("2026-07-24", str(caught.exception))

    def test_json_mode_requires_the_word_json(self) -> None:
        root = temporary_directory(self)
        (root / "DEEPSEEK-API-SYSTEM-PROMPT-FINAL.md").write_text(
            "Answer plainly.",
            encoding="utf-8",
        )
        with self.assertRaises(PromptError) as caught:
            build_chat_kwargs(
                "Give me the answer",
                root=root,
                response_format={"type": "json_object"},
            )
        self.assertIn("json", str(caught.exception).lower())

    def test_strict_tool_requires_beta_and_a_closed_schema(self) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": False,
                },
            },
        }
        with self.assertRaises(PromptError) as caught:
            build_chat_kwargs("Weather in Hangzhou", tools=[tool], beta=False)
        self.assertIn("beta", str(caught.exception))

        kwargs = build_chat_kwargs("Weather in Hangzhou", tools=[tool], beta=True)
        self.assertEqual(kwargs["tools"][0]["function"]["name"], "get_weather")

        broken = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        }
        with self.assertRaises(PromptError) as caught:
            build_chat_kwargs("Weather", tools=[broken], beta=True)
        self.assertIn("required", str(caught.exception))

    def test_stream_fields_are_independent(self) -> None:
        reasoning, content = collect_stream_text(
            [
                {"reasoning_content": "think", "content": None},
                {"reasoning_content": None, "content": "hi"},
                {"content": "!"},
                {},
            ]
        )
        self.assertEqual(reasoning, "think")
        self.assertEqual(content, "hi!")
        with self.assertRaises(PromptError):
            collect_stream_text([{"content": 1}])

    def test_tool_history_requires_reasoning_content(self) -> None:
        messages: list[dict[str, object]] = [{"role": "user", "content": "Hi"}]
        with self.assertRaises(PromptError) as caught:
            append_assistant_turn(
                messages,
                {"content": "done", "tool_calls": None},
                tools_in_request=True,
            )
        self.assertIn("400", str(caught.exception))
        self.assertEqual(len(messages), 1)

        append_assistant_turn(
            messages,
            {"content": "", "reasoning_content": "need a tool", "tool_calls": [{"id": "call_1"}]},
            tools_in_request=True,
        )
        self.assertEqual(messages[-1]["reasoning_content"], "need a tool")
        self.assertEqual(messages[-1]["tool_calls"], [{"id": "call_1"}])

    def test_status_explanations_do_not_invent_codes(self) -> None:
        self.assertIn("balance", explain_status(402).lower())
        with self.assertRaises(PromptError):
            explain_status(418)
        with self.assertRaises(PromptError):
            explain_status(True)  # type: ignore[arg-type]


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = {
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY"),
            "DEEPSEEK_BASE_URL": os.environ.get("DEEPSEEK_BASE_URL"),
        }

    def tearDown(self) -> None:
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_missing_key_message_has_no_secret(self) -> None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        with self.assertRaises(PromptError) as caught:
            client_settings_from_env()
        self.assertIn("DEEPSEEK_API_KEY", str(caught.exception))
        self.assertNotIn("sk-", str(caught.exception))

    def test_base_url_selection(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "test-key-not-a-real-secret"
        os.environ.pop("DEEPSEEK_BASE_URL", None)
        self.assertEqual(
            client_settings_from_env()["base_url"],
            "https://api.deepseek.com",
        )
        self.assertEqual(
            client_settings_from_env(beta=True)["base_url"],
            "https://api.deepseek.com/beta",
        )
        os.environ["DEEPSEEK_BASE_URL"] = "https://example.test/deepseek"
        self.assertEqual(
            client_settings_from_env(beta=True)["base_url"],
            "https://example.test/deepseek",
        )
        os.environ["DEEPSEEK_API_KEY"] = "   "
        with self.assertRaises(PromptError):
            client_settings_from_env()

    def test_cli_check_and_failure(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = main(["--check"])
        self.assertEqual(status, 0)
        self.assertIn("final:", stdout.getvalue())
        self.assertIn("enhance:", stdout.getvalue())
        self.assertNotIn("sk-", stdout.getvalue())

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            status = main(["missing"])
        self.assertEqual(status, 1)
        self.assertIn("Unknown prompt", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
