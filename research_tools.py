"""
Native CrewAI tools for RAG (knowledge-base search) and web search.

Usage in orchestrator:
    from research_tools import get_research_tools
    tools = get_research_tools()          # list[Tool], may be empty
    for agent in research_agents:
        agent.tools = tools
"""

import logging
import os
import tomllib
from pathlib import Path

from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

_config_path = Path(__file__).parent / "config.toml"
with open(_config_path, "rb") as f:
    _config = tomllib.load(f)

_rag_cfg = _config.get("rag", {})
_web_cfg = _config.get("web_search", {})

# ── RAG setup (lazy — only if enabled) ───────────────────────────────────────

_collection = None


def _init_rag():
    """Load documents from knowledge/ into an in-memory ChromaDB collection."""
    global _collection

    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    knowledge_dir = Path(__file__).parent / _rag_cfg.get("knowledge_dir", "knowledge")
    chunk_size = _rag_cfg.get("chunk_size", 1000)
    chunk_overlap = _rag_cfg.get("chunk_overlap", 200)
    embedding_model = _rag_cfg.get("embedding_model", "nomic-embed-text")

    # Resolve embedding provider: explicit [rag] setting > [llm].provider > "ollama"
    embedding_provider = _rag_cfg.get(
        "embedding_provider",
        _config.get("llm", {}).get("provider", "ollama"),
    )

    client = chromadb.Client()

    if embedding_provider == "gemini":
        from chromadb import EmbeddingFunction, Documents, Embeddings
        import requests as _requests

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is required for Gemini embeddings. "
                "Set it in .env or your environment."
            )
        _model = embedding_model or "models/gemini-embedding-001"

        class _GeminiEmbedding(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                import time
                # Use batch endpoint to reduce API calls
                url = f"https://generativelanguage.googleapis.com/v1beta/{_model}:batchEmbedContents?key={api_key}"
                # Process in batches of 100 (API limit)
                all_embeddings = []
                for i in range(0, len(input), 100):
                    batch = input[i:i + 100]
                    requests_body = [
                        {"model": _model, "content": {"parts": [{"text": text}]}}
                        for text in batch
                    ]
                    for attempt in range(5):
                        resp = _requests.post(url, json={"requests": requests_body})
                        if resp.status_code == 429:
                            wait = 2 ** attempt
                            logger.warning("Rate limited, waiting %ds...", wait)
                            time.sleep(wait)
                            continue
                        resp.raise_for_status()
                        break
                    else:
                        resp.raise_for_status()
                    for emb in resp.json()["embeddings"]:
                        all_embeddings.append(emb["values"])
                return all_embeddings

        embed_fn = _GeminiEmbedding()
        logger.info("Using Gemini embeddings (model=%s)", _model)
    else:
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

        ollama_url = _config.get("ollama", {}).get("base_url", "http://localhost:11434")
        embed_fn = OllamaEmbeddingFunction(
            model_name=embedding_model or "nomic-embed-text",
            url=f"{ollama_url}/api/embed",
        )
        logger.info("Using Ollama embeddings (model=%s, url=%s)", embedding_model, ollama_url)

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


# ── CrewAI tools ─────────────────────────────────────────────────────────────

@tool("search_knowledge_base")
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
        similarity = max(0.0, 1.0 - dist / 2.0)
        if similarity >= threshold:
            passages.append(f"[Source: {meta.get('source', '?')}] (score: {similarity:.2f})\n{doc}")

    if not passages:
        return "No relevant information found in the knowledge base. Rely on your own expertise."

    return "\n\n---\n\n".join(passages)


@tool("search_web")
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
        return "Web search failed. Rely on your own expertise."


# ── Public API ───────────────────────────────────────────────────────────────

def get_research_tools() -> list:
    """Return the list of enabled research tools (may be empty)."""
    tools = []

    if _rag_cfg.get("enabled", False):
        _init_rag()
        tools.append(search_knowledge_base)
        logger.info("RAG tool enabled")

    if _web_cfg.get("enabled", False):
        tools.append(search_web)
        logger.info("Web search tool enabled")

    return tools
