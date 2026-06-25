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
- do not stack it with the old long enhance file

Best fit:

- mixed chat + coding assistants
- Chinese-first technical assistants
- low-noise API integrations where cost and stability matter
