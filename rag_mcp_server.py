"""
MCP server exposing RAG (knowledge base search) and web search tools.

Run standalone:  python rag_mcp_server.py
The orchestrator launches this automatically as a subprocess via stdio.
"""

import json
import logging
import sys
import tomllib
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

_config_path = Path(__file__).parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)

_rag_cfg = _config.get("rag", {})
_web_cfg = _config.get("web_search", {})

RAG_ENABLED = _rag_cfg.get("enabled", False)
WEB_ENABLED = _web_cfg.get("enabled", False)

# ── RAG setup (lazy — only if enabled) ───────────────────────────────────────

_collection = None


def _init_rag():
    """Load documents from knowledge/ into an in-memory ChromaDB collection."""
    global _collection

    import chromadb
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    knowledge_dir = Path(__file__).parent / _rag_cfg.get("knowledge_dir", "knowledge")
    chunk_size = _rag_cfg.get("chunk_size", 1000)
    chunk_overlap = _rag_cfg.get("chunk_overlap", 200)
    embedding_model = _rag_cfg.get("embedding_model", "nomic-embed-text")
    ollama_url = _config["ollama"]["base_url"]

    client = chromadb.Client()
    embed_fn = OllamaEmbeddingFunction(
        model_name=embedding_model,
        url=f"{ollama_url}/api/embed",
    )
    _collection = client.get_or_create_collection(
        name="podcast_knowledge", embedding_function=embed_fn,
    )

    if not knowledge_dir.exists():
        logger.warning("knowledge/ directory not found — RAG has no documents.")
        return

    supported = {".txt", ".md", ".pdf"}
    files = [f for f in knowledge_dir.iterdir() if f.is_file() and f.suffix.lower() in supported]
    if not files:
        logger.warning("No supported files in knowledge/.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
    )

    all_chunks, all_ids, all_meta = [], [], []
    for fpath in files:
        text = _read_file(fpath)
        if not text.strip():
            continue
        for i, chunk in enumerate(splitter.split_text(text)):
            all_chunks.append(chunk)
            all_ids.append(f"{fpath.stem}_{i}")
            all_meta.append({"source": fpath.name, "chunk_index": i})

    if all_chunks:
        _collection.upsert(ids=all_ids, documents=all_chunks, metadatas=all_meta)
        logger.info("Ingested %d chunks from %d files.", len(all_chunks), len(files))


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        except ImportError:
            logger.warning("pypdf not installed — skipping %s", path.name)
            return ""
    return path.read_text(encoding="utf-8", errors="replace")


# ── MCP server ───────────────────────────────────────────────────────────────

mcp = FastMCP("podcast-research")


@mcp.tool(enabled=RAG_ENABLED)
def search_knowledge_base(query: str) -> str:
    """Search uploaded documents for information relevant to the query.
    Returns passages from the knowledge base that are highly relevant,
    or a message indicating nothing was found. Always also use your own
    expertise — these results supplement, not replace, your knowledge.
    """
    if _collection is None or _collection.count() == 0:
        return "No documents in the knowledge base. Rely on your own expertise."

    top_k = _rag_cfg.get("top_k", 5)
    threshold = _rag_cfg.get("similarity_threshold", 0.7)

    results = _collection.query(
        query_texts=[query],
        n_results=min(top_k, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    passages = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0],
    ):
        # ChromaDB returns L2 distance by default; lower = more similar.
        # Convert to a 0-1 similarity score (approximate).
        similarity = max(0.0, 1.0 - dist / 2.0)
        if similarity >= threshold:
            passages.append(f"[Source: {meta.get('source', '?')}] (score: {similarity:.2f})\n{doc}")

    if not passages:
        return "No relevant information found in the knowledge base. Rely on your own expertise."

    return "\n\n---\n\n".join(passages)


@mcp.tool(enabled=WEB_ENABLED)
def search_web(query: str) -> str:
    """Search the web for current information about the query.
    Returns snippets from top web results. Always also use your own
    expertise — these results supplement, not replace, your knowledge.
    """
    api_key = _web_cfg.get("tavily_api_key", "")
    if not api_key:
        return "Web search not configured (missing API key). Rely on your own expertise."

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        max_results = _web_cfg.get("max_results", 3)
        response = client.search(query=query, max_results=max_results)

        snippets = []
        for result in response.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            snippets.append(f"[{title}]({url})\n{content}")

        if not snippets:
            return "No web results found. Rely on your own expertise."

        return "\n\n---\n\n".join(snippets)

    except Exception as e:
        logger.error("Web search failed: %s", e)
        return f"Web search failed. Rely on your own expertise."


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if RAG_ENABLED:
        _init_rag()
    logger.info("MCP server starting (rag=%s, web=%s)", RAG_ENABLED, WEB_ENABLED)
    mcp.run(transport="stdio")
