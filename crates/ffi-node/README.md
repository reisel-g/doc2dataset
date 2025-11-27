# `three_dcf_node`

Minimal [napi-rs](https://napi.rs/) bindings that expose the encoder, decoder, and
stats helpers to JavaScript runtimes. Build it with `npm run build` or directly
with cargo:

```bash
npm install # installs napi + builds the .node artifact
# or
cargo build -p three_dcf_node
```

### API surface

```ts
import { encodeFile, decodeText, stats } from 'three_dcf_node';

await encodeFile('input.pdf', 'doc.3dcf', 'reports', 256, 'doc.json', 'doc.txt');
const raw = await decodeText('doc.3dcf', undefined);
const metrics = await stats('doc.3dcf', 'cl100k_base');
```

Tokenizer names mirror the CLI (`cl100k_base`, `gpt2`, `o200k`, `anthropic`).
Passing a filesystem path loads a custom tokenizer JSON.
