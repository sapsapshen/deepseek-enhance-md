# DeepSeek Enhance MD

System prompts and a small loader for the DeepSeek API. The prompts steer chat and coding answers. The loader checks the files and builds a chat request without calling the API or printing a key.

This repository does not ship a model and does not change the DeepSeek service. It documents how to call the current API as of 2026-11-13. Prices and limits change; confirm them at <https://api-docs.deepseek.com> before quoting a bill.

## Which prompt to send

Send one system prompt, not both.

| File | Role |
|---|---|
| `DEEPSEEK-API-SYSTEM-PROMPT-FINAL.md` | Default. Short rules for chat and coding. See [usage notes](DEEPSEEK-API-SYSTEM-PROMPT-FINAL-USAGE.md). |
| `DEEPSEEK-ENHANCE.md` | Same kind of assistant, plus a checked snapshot of thinking mode, tools, caching, search, and errors. Use it when the assistant must answer questions about the DeepSeek API. |

`deepseek_prompt.build_messages(..., prompt="final")` loads the short file. `prompt="enhance"` loads the long file. `extra_system` appends a few application rules to the selected file.

## Quick start

Export the key in your shell. Do not commit it or pass it as a command-line argument.

```bash
export DEEPSEEK_API_KEY="your key from the DeepSeek platform"
```

```python
from openai import OpenAI

from deepseek_prompt import build_chat_kwargs, client_settings_from_env

client = OpenAI(**client_settings_from_env())
response = client.chat.completions.create(
    **build_chat_kwargs(
        "Your task here",
        prompt="final",
        model="deepseek-flash",
        thinking=True,
        reasoning_effort="high",
    )
)
print(response.choices[0].message.content)
```

The OpenAI SDK base URL is `https://api.deepseek.com`. Do not add `/v1` yourself. Beta features (FIM, chat prefix completion, and strict tools) use `client_settings_from_env(beta=True)`, which selects `https://api.deepseek.com/beta` unless `DEEPSEEK_BASE_URL` is set. The Anthropic-compatible base URL is `https://api.deepseek.com/anthropic`.

Check both prompt files without printing them:

```bash
python deepseek_prompt.py --check
```

`deepseek_prompt.py` depends only on the Python standard library. CI runs it on Python 3.12. Install `openai` yourself when you want to make a request:

```bash
python -m pip install openai
```

If `DEEPSEEK_API_KEY` is missing, `client_settings_from_env()` raises `PromptError` and the message does not include a key.

## Models

| Model id | Use | Context | Max output | Vision | Concurrency |
|---|---|---|---|---|---|
| `deepseek-flash` | Fast, lower cost, images. Served by DeepSeek-V4.1-Flash-1017. | 1M | 384K | Yes | 2500 |
| `deepseek-v4-pro` | Hard text reasoning and agent work. Version DeepSeek-V4-Pro-0813. | 1M | 384K | No | 500 |

`deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` still route to Flash and are billed as Flash. Prefer `deepseek-flash` in new code.

`deepseek-chat` and `deepseek-reasoner` were scheduled to be discontinued on 2026-07-24. Do not use them in new calls. Choose a current model id and set `thinking` yourself. Thinking is on by default, and the default effort is `high`.

`recommend_model(complex_reasoning=..., needs_vision=...)` encodes the vision constraint: Pro cannot accept an image, so a request that needs both Pro and an image raises `PromptError`.

## API behavior the long prompt used to get wrong

These are the rules the loader and `DEEPSEEK-ENHANCE.md` now follow:

- Requested effort `low` stays low. `medium` maps to `high`. `xhigh` maps to `high`, not `max`. `max` and `ultra` map to `max`.
- Send `reasoning_content` back on every assistant turn when the request includes `tools`, even if that turn did not call a tool. Otherwise the API returns HTTP 400. Without `tools`, the API ignores `reasoning_content`.
- `presence_penalty` and `frequency_penalty` no longer affect chat completions. `temperature` has no effect in thinking mode. `top_p` is model-specific.
- Search and Fetch are separate HTTP APIs. The model must not claim it searched, and it must not emit another provider's citation tags.
- JSON mode needs `response_format={"type": "json_object"}`, the word `json` in the prompt, and an example shape. Unparseable output can come back as spaces.
- Strict tools need the beta base URL, `strict: true`, every property listed in `required`, and `additionalProperties: false`.

Peak prices apply at 01:00-04:00 and 06:00-10:00 UTC on weekdays, excluding Chinese public holidays. All other hours are off-peak, at half the peak price. The dated table is in `DEEPSEEK-ENHANCE.md`.

## Claude Code

Point Claude Code at DeepSeek with environment variables from the official guide: <https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code>. Set `ANTHROPIC_BASE_URL` to `https://api.deepseek.com/anthropic` and `ANTHROPIC_AUTH_TOKEN` to the same key. Names starting with `claude-opus` are billed as `deepseek-v4-pro`; other `claude*` names are billed as `deepseek-flash`. Set `CLAUDE_CODE_EFFORT_LEVEL` yourself. This repository does not enable permission bypasses.

## Tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests are offline. They do not call DeepSeek and they do not need an API key.

## License

MIT. See [LICENSE](LICENSE).
