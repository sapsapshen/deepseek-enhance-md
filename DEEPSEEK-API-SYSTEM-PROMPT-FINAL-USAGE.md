# DeepSeek API Final Prompt Usage

This is the clean publishable version intended for direct use in the `system` message of the DeepSeek API.

Why this version is better:

- short enough for repeated API use
- broad enough to cover chat and coding
- avoids product-specific claims and brittle runtime assumptions
- reduces overthinking on simple tasks
- keeps strong anti-hallucination guidance

Recommended use:

- use this as the base `system` prompt
- add only a small number of app-specific rules on top
- do not concatenate it with `DEEPSEEK-ENHANCE.md`

Load it from the repository root with:

```python
from openai import OpenAI

from deepseek_prompt import build_chat_kwargs, client_settings_from_env

client = OpenAI(**client_settings_from_env())
response = client.chat.completions.create(
    **build_chat_kwargs("Your task here", prompt="final")
)
```

`deepseek_prompt.py` uses only the Python standard library. Install `openai` separately when you call the API. The key comes from `DEEPSEEK_API_KEY` and is not written into the request helper's logs.

Best fit:

- mixed chat + coding assistants
- Chinese-first technical assistants
- low-noise API integrations where cost and stability matter
