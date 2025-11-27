# `three_dcf_py`

[Pyo3](https://pyo3.rs/) bindings for the 3DCF encoder/decoder/stats helpers.
Build/install with [maturin](https://www.maturin.rs/) or directly with cargo:

```bash
maturin develop -m crates/ffi-py/Cargo.toml
# or
cargo build -p three_dcf_py
```

### Usage

```python
from three_dcf_py import encode, decode_text, stats

encode("input.pdf", "doc.3dcf", preset="reports", budget=256,
       json_out="doc.json", text_out="doc.txt")
text = decode_text("doc.3dcf")
summary = stats("doc.3dcf", tokenizer="cl100k_base")
print(summary.tokens_raw, summary.tokens_3dcf)
```

Tokenizer names mirror the CLI (`cl100k_base`, `gpt2`, `o200k`, `anthropic`).
Passing a filesystem path loads a custom tokenizer JSON.
