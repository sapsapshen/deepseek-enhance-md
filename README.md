# DeepSeek Enhance MD

A pure capability-enhancement system prompt for DeepSeek V4 models (V4-Pro / V4-Flash), adapted and inspired by the Claude Fable 5 system prompt architecture.

## What This Is

`DEEPSEEK-ENHANCE.md` is a system-level prompt that unlocks the full potential of DeepSeek V4 models by providing structured guidance on:

- **Thinking Mode** — Chain-of-thought reasoning with controllable effort levels (high/max), reasoning_content lifecycle management, and multi-turn tool-call reasoning loops
- **Model Selection** — Decision guide for choosing between deepseek-v4-pro (flagship reasoning) and deepseek-v4-flash (fast/cost-efficient)
- **Code Enhancement** — FIM completion, JSON structured output, tool calls, chat prefix completion, and coding best practices
- **Context Caching** — Automatic disk cache mechanics, cache-hit patterns, and prefix reuse strategies to minimize costs
- **Search & Tools** — When to search, how to scale tool calls, citation formatting, and image search guidelines
- **Chinese Native Optimization** — Bilingual proficiency, typesetting conventions, code/technical writing language choice, and domain terminology

## Background

This document draws structural inspiration from Anthropic's Claude Fable 5 system prompt — the internal prompt that powers Claude's advanced reasoning, tool use, and behavioral consistency. The content has been completely rewritten for DeepSeek's model architecture, API conventions, and capabilities, with all capability-limiting instructions removed.

## Usage

Load `DEEPSEEK-ENHANCE.md` as the system prompt (or prepend it to your system prompt) when using DeepSeek V4 models via the API:

### OpenAI Format
```python
from openai import OpenAI

with open("DEEPSEEK-ENHANCE.md", "r", encoding="utf-8") as f:
    system_prompt = f.read()

client = OpenAI(
    api_key="<your-api-key>",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Your task here"}
    ],
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)
```

### Anthropic Format
```python
base_url = "https://api.deepseek.com/anthropic"
```

### With Claude Code
```bash
claude --dangerously-skip-permissions --system-prompt-file DEEPSEEK-ENHANCE.md
```

## Models Supported

| Model | Best For | Context | Max Output |
|---|---|---|---|
| `deepseek-v4-pro` | Complex reasoning, agent workflows, code | 1M | 384K |
| `deepseek-v4-flash` | High-concurrency, simple Q&A, low latency | 1M | 384K |

## License

MIT
