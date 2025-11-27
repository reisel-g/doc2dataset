# 3DCF Recipes

This folder mirrors the standalone examples repo described in the plan. Each recipe is intentionally self-contained so you can copy it into your own project.

## Contents

- `openai-qa/` – minimal script that runs `encode_to_context`, sends the prompt to OpenAI with savings breakdown, and prints the answer last.
- `langchain-rag/` – drop-in `ThreeDCFDocumentLoader` and `ThreeDCFCompressor` classes for LangChain pipelines.
- `http-service-client/` – Python client that hits the `/encode` HTTP microservice and prints the returned metrics.
- `numguard-demo/` – quick detector that highlights numeric mismatches by re-running the guard check.

Every recipe ships with a `README.md` describing setup and the exact commands to run.
