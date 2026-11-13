"""Load this repository's DeepSeek system prompts and build chat requests.

The functions here only check local files and arguments. They do not call the
DeepSeek API and they do not print API keys.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PROMPT_FILES: dict[str, str] = {
    "final": "DEEPSEEK-API-SYSTEM-PROMPT-FINAL.md",
    "enhance": "DEEPSEEK-ENHANCE.md",
}

OPENAI_BASE_URL = "https://api.deepseek.com"
BETA_BASE_URL = "https://api.deepseek.com/beta"

CURRENT_MODELS = frozenset({"deepseek-flash", "deepseek-v4-pro"})
LEGACY_FLASH_ALIASES = frozenset({"deepseek-v4-flash", "deepseek-v4-flash-vision-exp"})
KNOWN_MODELS = CURRENT_MODELS | LEGACY_FLASH_ALIASES

# Requested effort values accepted by the API, and the effort the service
# actually runs. See https://api-docs.deepseek.com/guides/thinking_mode
EFFORT_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "high",
    "max": "max",
    "ultra": "max",
}

API_ERRORS: dict[int, str] = {
    400: "Invalid request body. A tools request that omits reasoning_content from an earlier assistant turn also returns 400.",
    401: "Authentication failed. Check DEEPSEEK_API_KEY. Do not paste the key into chat.",
    402: "Insufficient balance. Top up the DeepSeek account before retrying.",
    422: "Invalid parameters. Adjust the request to match the error message.",
    429: "Rate limit or concurrency limit reached. Slow down and retry.",
    500: "DeepSeek server error. Retry after a short wait.",
    503: "DeepSeek is overloaded. Retry after a short wait.",
}

_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9]{16,}\b")
_MAX_SCHEMA_DEPTH = 20


class PromptError(ValueError):
    """A local prompt file or request argument cannot be used."""


def load_system_prompt(name: str = "final", *, root: Path | None = None) -> str:
    """Read one system prompt as UTF-8 text.

    `name` is ``final`` (short default) or ``enhance`` (API reference prompt).
    """

    key = name.strip().lower()
    try:
        filename = PROMPT_FILES[key]
    except KeyError:
        known = ", ".join(sorted(PROMPT_FILES))
        raise PromptError(f"Unknown prompt {name!r}. Expected one of: {known}.") from None

    base = _root_dir(root)
    path = base / filename
    if not path.is_file():
        raise PromptError(f"Prompt file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PromptError(f"Prompt file is not valid UTF-8: {path}") from exc

    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    if not text.strip():
        raise PromptError(f"Prompt file is empty: {path}")
    _reject_secret(text, path)
    return text


def build_messages(
    user_content: str,
    *,
    prompt: str = "final",
    extra_system: str | None = None,
    root: Path | None = None,
) -> list[dict[str, str]]:
    """Build a system-plus-user message list.

    App-specific rules in `extra_system` are appended to the selected prompt.
    Pass either ``final`` or ``enhance``, not both.
    """

    if not isinstance(user_content, str) or not user_content.strip():
        raise PromptError("user_content must be a non-empty string.")

    system = load_system_prompt(prompt, root=root).rstrip()
    if extra_system is not None:
        extra = extra_system.strip()
        if not extra:
            raise PromptError("extra_system must be a non-empty string when provided.")
        _reject_secret(extra, "extra_system")
        system = f"{system}\n\n{extra}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content.strip()},
    ]


def recommend_model(*, complex_reasoning: bool, needs_vision: bool) -> str:
    """Choose a current model id for this request.

    ``deepseek-v4-pro`` has no vision support. A request that needs both an
    image and the Pro model has to be split by the caller.
    """

    if not isinstance(complex_reasoning, bool) or not isinstance(needs_vision, bool):
        raise PromptError("complex_reasoning and needs_vision must be booleans.")
    if needs_vision and complex_reasoning:
        raise PromptError(
            "deepseek-v4-pro does not accept images. Use deepseek-flash for vision, "
            "or remove the image and use deepseek-v4-pro for text-only reasoning."
        )
    if needs_vision:
        return "deepseek-flash"
    if complex_reasoning:
        return "deepseek-v4-pro"
    return "deepseek-flash"


def effective_reasoning_effort(requested: str) -> str:
    """Return the effort DeepSeek actually runs for a requested value."""

    if not isinstance(requested, str):
        raise PromptError("reasoning_effort must be a string.")
    key = requested.strip().lower()
    try:
        return EFFORT_MAP[key]
    except KeyError:
        known = ", ".join(EFFORT_MAP)
        raise PromptError(
            f"Unknown reasoning_effort {requested!r}. Expected one of: {known}."
        ) from None


def explain_status(code: int) -> str:
    """Return a short local explanation for a documented DeepSeek HTTP status."""

    if not isinstance(code, int) or isinstance(code, bool):
        raise PromptError("HTTP status code must be an integer.")
    try:
        return API_ERRORS[code]
    except KeyError:
        raise PromptError(
            f"No local explanation for HTTP {code}. "
            "See https://api-docs.deepseek.com/quick_start/error_codes"
        ) from None


def client_settings_from_env(*, beta: bool = False) -> dict[str, str]:
    """Read client settings from the environment.

    Uses ``DEEPSEEK_API_KEY``. ``DEEPSEEK_BASE_URL``, when set, replaces the
    default OpenAI-compatible base URL (or the beta URL when `beta` is true).
    The returned key is for the SDK only; callers must not log it.
    """

    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise PromptError(
            "DEEPSEEK_API_KEY is not set. Export it in the environment. "
            "Do not put it in source, command-line arguments, or a prompt file."
        )

    default_base = BETA_BASE_URL if beta else OPENAI_BASE_URL
    base = os.environ.get("DEEPSEEK_BASE_URL", default_base).strip()
    if not base:
        raise PromptError("DEEPSEEK_BASE_URL is set but empty.")
    return {"api_key": key, "base_url": base}


def build_chat_kwargs(
    user_content: str,
    *,
    model: str | None = None,
    prompt: str = "final",
    extra_system: str | None = None,
    thinking: bool = True,
    reasoning_effort: str = "high",
    response_format: Mapping[str, object] | None = None,
    tools: Sequence[Mapping[str, object]] | None = None,
    beta: bool = False,
    root: Path | None = None,
) -> dict[str, object]:
    """Build keyword arguments for ``chat.completions.create``.

    The result never includes an API key. Pass the requested `reasoning_effort`
    through; call :func:`effective_reasoning_effort` to see how the service
    maps compatibility aliases. Strict tools require `beta=True` and the beta
    base URL.
    """

    if not isinstance(thinking, bool):
        raise PromptError("thinking must be a boolean.")

    selected = "deepseek-flash" if model is None else _normalize_model(model)
    if not isinstance(reasoning_effort, str):
        raise PromptError("reasoning_effort must be a string.")
    effort = reasoning_effort.strip().lower()
    effective_reasoning_effort(effort)

    validated_format = _validate_response_format(response_format)
    validated_tools = _validate_tools(tools)
    if validated_tools and any(_tool_is_strict(tool) for tool in validated_tools) and not beta:
        raise PromptError(
            "strict tools require the beta base URL. "
            "Pass beta=True and use client_settings_from_env(beta=True)."
        )

    messages = build_messages(
        user_content,
        prompt=prompt,
        extra_system=extra_system,
        root=root,
    )
    if validated_format is not None and validated_format["type"] == "json_object":
        combined = "\n".join(str(message["content"]) for message in messages)
        if "json" not in combined.lower():
            raise PromptError(
                "JSON output requires the word 'json' in the system or user message, "
                "plus an example of the target shape. Otherwise the API can return spaces."
            )

    kwargs: dict[str, object] = {
        "model": selected,
        "messages": messages,
        "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
    }
    if thinking:
        kwargs["reasoning_effort"] = effort
    if validated_format is not None:
        kwargs["response_format"] = validated_format
    if validated_tools is not None:
        kwargs["tools"] = validated_tools
    return kwargs


def collect_stream_text(deltas: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    """Join streamed ``reasoning_content`` and ``content`` without dropping either.

    A chunk may contain one field, both, or neither. ``None`` is ignored.
    This does not reconstruct ``tool_calls``; append the SDK message object
    for tool turns.
    """

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for index, delta in enumerate(deltas):
        if not isinstance(delta, Mapping):
            raise PromptError(f"Stream delta {index} must be a mapping.")
        reasoning_parts.append(_stream_field(delta, "reasoning_content", index))
        content_parts.append(_stream_field(delta, "content", index))
    return "".join(reasoning_parts), "".join(content_parts)


def append_assistant_turn(
    messages: list[dict[str, object]],
    message: Mapping[str, object],
    *,
    tools_in_request: bool,
) -> None:
    """Append an assistant turn in the shape the next request must send back.

    When the next request includes ``tools``, every earlier assistant turn
    needs ``reasoning_content``, including turns that did not call a tool.
    """

    if not isinstance(messages, list):
        raise PromptError("messages must be a list.")
    if not isinstance(message, Mapping):
        raise PromptError("assistant message must be a mapping.")

    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise PromptError("assistant content must be a string or null.")

    entry: dict[str, object] = {"role": "assistant", "content": content}
    reasoning = message.get("reasoning_content")
    if tools_in_request:
        if not isinstance(reasoning, str):
            raise PromptError(
                "A request that includes tools must send reasoning_content back "
                "on every assistant turn, including turns with no tool call. "
                "Omitting it makes the API return HTTP 400."
            )
        entry["reasoning_content"] = reasoning
    elif isinstance(reasoning, str) and reasoning:
        entry["reasoning_content"] = reasoning

    tool_calls = message.get("tool_calls")
    if tool_calls:
        try:
            entry["tool_calls"] = copy.deepcopy(tool_calls)
        except TypeError as exc:
            raise PromptError("tool_calls cannot be copied.") from exc
    messages.append(entry)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the prompt files, or print one of them."""

    parser = argparse.ArgumentParser(
        description="Load a DeepSeek system prompt from this repository."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="final",
        help="Prompt name: final (default) or enhance. Ignored with --check.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate both prompt files and print their sizes.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.check:
            for prompt_name in sorted(PROMPT_FILES):
                text = load_system_prompt(prompt_name)
                print(f"{prompt_name}: {len(text)} characters")
            return 0
        text = load_system_prompt(args.prompt)
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    except PromptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _root_dir(root: Path | None) -> Path:
    base = (root or ROOT).resolve()
    if not base.is_dir():
        raise PromptError(f"Prompt root is not a directory: {base}")
    return base


def _reject_secret(text: str, label: object) -> None:
    if _SECRET_RE.search(text):
        raise PromptError(
            f"{label} looks like it contains an API key. Remove it before loading."
        )


def _normalize_model(model: str) -> str:
    if not isinstance(model, str):
        raise PromptError("model must be a string.")
    name = model.strip()
    if name in KNOWN_MODELS:
        return name
    raise PromptError(
        f"Unknown model {model!r}. Use deepseek-flash or deepseek-v4-pro. "
        "deepseek-chat and deepseek-reasoner were scheduled to be discontinued "
        "on 2026-07-24; do not use them in new calls."
    )


def _validate_response_format(
    response_format: Mapping[str, object] | None,
) -> dict[str, str] | None:
    if response_format is None:
        return None
    if not isinstance(response_format, Mapping):
        raise PromptError("response_format must be a mapping.")
    kind = response_format.get("type")
    if not isinstance(kind, str) or kind not in {"text", "json_object"}:
        raise PromptError("response_format.type must be 'text' or 'json_object'.")
    return {"type": kind}


def _validate_tools(
    tools: Sequence[Mapping[str, object]] | None,
) -> list[Mapping[str, object]] | None:
    if tools is None:
        return None
    if isinstance(tools, str) or not isinstance(tools, Sequence) or not tools:
        raise PromptError("tools must be a non-empty list.")

    validated: list[Mapping[str, object]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise PromptError(f"tools[{index}] must be a mapping with type 'function'.")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise PromptError(f"tools[{index}] is missing a function object.")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise PromptError(f"tools[{index}] needs a non-empty function name.")
        try:
            copied = copy.deepcopy(dict(tool))
        except TypeError as exc:
            raise PromptError(f"tools[{index}] cannot be copied.") from exc
        if function.get("strict") is True:
            parameters = function.get("parameters")
            if not isinstance(parameters, Mapping):
                raise PromptError(f"Strict tool {name!r} needs a parameters object.")
            _check_strict_schema(parameters, f"Strict tool {name!r}", 0)
        validated.append(copied)
    return validated


def _tool_is_strict(tool: Mapping[str, object]) -> bool:
    function = tool.get("function")
    return isinstance(function, Mapping) and function.get("strict") is True


def _check_strict_schema(schema: Mapping[str, object], where: str, depth: int) -> None:
    if depth > _MAX_SCHEMA_DEPTH:
        raise PromptError(f"{where} is nested too deeply to validate.")
    if "$ref" in schema:
        return

    if "anyOf" in schema:
        branches = schema.get("anyOf")
        if not isinstance(branches, list):
            raise PromptError(f"{where} anyOf must be a list.")
        for index, branch in enumerate(branches):
            if not isinstance(branch, Mapping):
                raise PromptError(f"{where}.anyOf[{index}] must be a mapping.")
            _check_strict_schema(branch, f"{where}.anyOf[{index}]", depth + 1)

    schema_type = schema.get("type")
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            _check_strict_schema(items, f"{where}.items", depth + 1)
        return

    if schema_type != "object" and "properties" not in schema:
        return

    if schema.get("additionalProperties") is not False:
        raise PromptError(f"{where} must set additionalProperties to false.")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise PromptError(f"{where} properties must be a mapping.")
    required = schema.get("required")
    if not isinstance(required, list):
        raise PromptError(f"{where} must list every property in required.")
    try:
        missing = set(properties) - set(required)
    except TypeError as exc:
        raise PromptError(f"{where} property names must be plain strings.") from exc
    if missing:
        raise PromptError(f"{where} must list every property in required.")
    for prop_name, child in properties.items():
        if isinstance(child, Mapping):
            _check_strict_schema(child, f"{where}.{prop_name}", depth + 1)


def _stream_field(delta: Mapping[str, object], field: str, index: int) -> str:
    value = delta.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PromptError(f"Stream delta {index} field {field} must be a string when present.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
