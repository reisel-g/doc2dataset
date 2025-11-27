# NumGuard demo

Use this helper to highlight when numbers change between source and encoded output.

```
python main.py ./datasets/sample.pdf
```

The script wraps `cargo run -p three_dcf_cli -- encode ... --strict-numguard` and surfaces the warning block. If you edit the PDF to change a number, NumGuard will trip and the script will print the mismatch summary.
