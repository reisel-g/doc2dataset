# OpenAI QA Recipe

1. Install deps: `pip install requests three-dcf-py` (or `maturin develop` in this repo).
2. Export `OPENAI_API_KEY`.
3. Run `python main.py path/to/report.pdf`.
4. The script prints 3DCF metrics first and the LLM answer last so you can pipe the output.

Feel free to tweak `QUESTION`, `MODEL`, or the `encode_to_context` budget.
