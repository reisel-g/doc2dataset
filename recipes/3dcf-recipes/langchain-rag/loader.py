"""LangChain helpers for 3DCF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from langchain.schema import Document
from langchain.schema.document import BaseDocumentTransformer
from langchain.schema.loaders import BaseLoader

from three_dcf_py import encode_to_cells, encode_to_context


@dataclass
class LoaderConfig:
    path: str
    preset: str = "reports"
    budget: int = 256


class ThreeDCFDocumentLoader(BaseLoader):
    """Converts a PDF into LangChain Documents using the 3DCF cell abstraction."""

    def __init__(self, config: LoaderConfig) -> None:
        self.config = config

    def load(self) -> List[Document]:
        cells = encode_to_cells(self.config.path, preset=self.config.preset)
        docs: List[Document] = []
        for idx, cell in enumerate(cells):
            docs.append(
                Document(
                    page_content=cell.text,
                    metadata={
                        "cell_index": idx,
                        "page": cell.page,
                        "importance": cell.importance,
                        "bbox": cell.bbox,
                    },
                )
            )
        return docs


class ThreeDCFCompressor(BaseDocumentTransformer):
    """Keeps the highest-importance 3DCF cells within a fixed budget."""

    def __init__(self, keep: int = 256) -> None:
        self.keep = keep

    def transform_documents(
        self, documents: Iterable[Document], **kwargs: object
    ) -> List[Document]:
        ranked = sorted(
            documents,
            key=lambda doc: doc.metadata.get("importance", 0),
            reverse=True,
        )
        return ranked[: self.keep]


def build_context(path: str, preset: str = "reports", budget: int = 256) -> str:
    """One-liner that wraps `encode_to_context` for convenience."""

    context = encode_to_context(path, preset=preset, budget=budget, tokenizer="cl100k_base")
    return context.text
