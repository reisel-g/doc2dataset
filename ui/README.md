# 3DCF RAG Web UI

This minimal static page calls the REST endpoints exposed by `three_dcf_service`.

## Usage

1. Start the service: `cargo run -p three_dcf_service`.
2. Serve `ui/` via any static server (for local dev, `python -m http.server -d ui 8080`).
3. Visit `http://localhost:8080` (or wherever you served it).
4. Upload PDFs into a collection and ask questions via the form. All requests hit the local service (`/rag/collections/...` and `/rag/query`).

The UI surfaces metrics (tokens, savings) plus the retrieved sources for transparency.
