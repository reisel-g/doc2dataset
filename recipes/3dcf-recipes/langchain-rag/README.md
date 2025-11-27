# LangChain helpers

Drop `loader.py` into your project and wire it up as follows:

```python
from langchain.schema import Document
from loader import LoaderConfig, ThreeDCFDocumentLoader, ThreeDCFCompressor

loader = ThreeDCFDocumentLoader(LoaderConfig(path="/tmp/report.pdf", budget=256))
docs: list[Document] = loader.load()
compressor = ThreeDCFCompressor(keep=128)
trimmed = compressor.transform_documents(docs)
```

`trimmed` now contains the highest-importance 3DCF cells, ready for your embeddings / retriever step. `build_context` is a helper that mirrors the CLI `context` command and returns a serialized `.context.txt` string on demand.
