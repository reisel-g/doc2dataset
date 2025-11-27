"""LlamaIndex reader built on top of the Python bindings."""

from __future__ import annotations

from typing import List

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from three_dcf_py import encode_to_cells


class ThreeDCFReader(BaseReader):
    def __init__(self, preset: str = "reports", budget: int = 256) -> None:
        self.preset = preset
        self.budget = budget

    def load_data(self, file_paths: List[str]) -> List[Document]:
        documents: List[Document] = []
        for path in file_paths:
            cells = encode_to_cells(path, preset=self.preset)
            for idx, cell in enumerate(cells):
                metadata = {
                    "path": path,
                    "page": cell.page,
                    "cell_index": idx,
                    "importance": cell.importance,
                }
                documents.append(Document(text=cell.text, metadata=metadata))
        return documents
