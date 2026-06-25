"""Dataset-grounded RAG engine and Kimi chat orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack
import joblib

from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
from backend.core.api_manager import APIKeyPoolExhaustedError, APIManager
from backend.core.logger import get_logger
from backend.chatbot.fallback import extractive_summary_from_chunks

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DatasetChunk:
    """Searchable dataset text chunk with source metadata."""

    source: str
    row_id: int
    text: str
    metadata: Mapping[str, Any]


class DatasetRAGEngine:
    """Build and query a lightweight vector index over local career datasets."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, max_rows_per_file: int = 2500) -> None:
        """Initialize the RAG engine with a raw dataset directory."""
        self.raw_dir = raw_dir
        self.max_rows_per_file = max_rows_per_file
        self.chunks: list[DatasetChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None
        # temporary uploaded chunks and combined matrix for ephemeral context
        self._temp_chunks: list[DatasetChunk] = []
        self._temp_matrix: Any = None
        self._combined_matrix: Any = None

    def discover_dataset_files(self, limit: int | None = None) -> list[Path]:
        """Return CSV datasets from the configured raw directory."""
        if not self.raw_dir.exists():
            return []
        files = sorted(path for path in self.raw_dir.glob("*.csv") if path.is_file())
        return files if limit is None else files[:limit]

    def build_index(self) -> int:
        """Load datasets and build a local TF-IDF vector index."""
        self.chunks = self._load_chunks()
        if not self.chunks:
            self._vectorizer = None
            self._matrix = None
            return 0
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
        self._matrix = self._vectorizer.fit_transform([chunk.text for chunk in self.chunks])
        # reset temp structures
        self._temp_chunks = []
        self._temp_matrix = None
        self._combined_matrix = self._matrix
        LOGGER.info("Built RAG index with %s chunks from %s.", len(self.chunks), self.raw_dir)
        return len(self.chunks)

    def save_index(self, out_path: Path) -> None:
        """Persist the vectorizer, matrix, and chunk metadata to disk using joblib."""
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Index not built; call build_index() before saving.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vectorizer": self._vectorizer,
            "matrix": self._matrix,
            "chunks": self.chunks,
        }
        joblib.dump(payload, out_path)
        LOGGER.info("Saved RAG index to %s", out_path)

    def load_index(self, in_path: Path) -> int:
        """Load a previously saved index from disk. Returns number of chunks loaded."""
        if not in_path.exists():
            raise FileNotFoundError(in_path)
        payload = joblib.load(in_path)
        self._vectorizer = payload.get("vectorizer")
        self._matrix = payload.get("matrix")
        self.chunks = payload.get("chunks", [])
        LOGGER.info("Loaded RAG index from %s with %s chunks", in_path, len(self.chunks))
        return len(self.chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[DatasetChunk]:
        """Retrieve the most relevant dataset chunks for a user query."""
        if not query.strip() or top_k <= 0:
            return []
        if self._vectorizer is None or self._matrix is None:
            self.build_index()
        if self._vectorizer is None or self._matrix is None or not self.chunks:
            return []
        matrix = self._combined_matrix if self._combined_matrix is not None else self._matrix
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).ravel()
        ranked_indexes = scores.argsort()[::-1][:top_k]
        # map indexes to chunks (base chunks followed by temp chunks if present)
        all_chunks = list(self.chunks) + list(self._temp_chunks)
        results: list[DatasetChunk] = []
        for index in ranked_indexes:
            idx = int(index)
            if idx < 0 or idx >= len(all_chunks):
                continue
            if float(scores[idx]) <= 0:
                continue
            results.append(all_chunks[idx])
        return results

    def context_block(self, query: str, top_k: int = 5) -> str:
        """Return retrieved context formatted for a chat system prompt."""
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return "No relevant local dataset context was retrieved."
        lines: list[str] = []
        for chunk in chunks:
            lines.append(f"Source: {chunk.source} row {chunk.row_id}\n{chunk.text}")
        return "\n\n".join(lines)

    def add_temporary_chunks(self, texts: Sequence[str], source: str = "uploaded_resume") -> int:
        """Add ephemeral chunks (e.g., uploaded resume text) to the index for the current session.

        These chunks are not persisted and will be cleared by `clear_temporary_chunks()`.
        Returns the number of temporary chunks currently held.
        """
        if self._vectorizer is None or self._matrix is None:
            # ensure base index exists
            self.build_index()
        if self._vectorizer is None:
            return 0
        new_chunks: list[DatasetChunk] = []
        texts_list = [str(t) for t in texts if t]
        if not texts_list:
            return 0
        start_id = -1 - len(self._temp_chunks)
        for i, t in enumerate(texts_list):
            new_chunks.append(DatasetChunk(source=source, row_id=start_id - i, text=t, metadata={}))
        temp_mat = self._vectorizer.transform(texts_list)
        if self._temp_matrix is None:
            self._temp_matrix = temp_mat
        else:
            self._temp_matrix = vstack([self._temp_matrix, temp_mat])
        # update combined matrix and chunks
        self._temp_chunks.extend(new_chunks)
        if self._matrix is not None:
            self._combined_matrix = vstack([self._matrix, self._temp_matrix])
        else:
            self._combined_matrix = self._temp_matrix
        return len(self._temp_chunks)

    def clear_temporary_chunks(self) -> None:
        """Remove any ephemeral chunks and restore the original index state."""
        self._temp_chunks = []
        self._temp_matrix = None
        self._combined_matrix = self._matrix

    def _load_chunks(self) -> list[DatasetChunk]:
        """Read CSV files into compact row-level text chunks."""
        chunks: list[DatasetChunk] = []
        for path in self.discover_dataset_files():
            try:
                dataframe = pd.read_csv(path, nrows=self.max_rows_per_file)
            except Exception as exc:
                LOGGER.warning("Skipping dataset %s because it could not be read: %s", path.name, exc)
                continue
            for row_id, row in dataframe.fillna("").iterrows():
                text = self._row_to_text(row)
                if text:
                    chunks.append(
                        DatasetChunk(
                            source=path.name,
                            row_id=int(row_id),
                            text=text,
                            metadata={"columns": list(dataframe.columns)},
                        )
                    )
        return chunks

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        """Convert a dataframe row into a compact natural-language chunk."""
        parts: list[str] = []
        for column, value in row.items():
            cleaned = str(value).strip()
            if cleaned:
                parts.append(f"{column}: {cleaned}")
        return " | ".join(parts)


class KimiRAGChatbot:
    """Kimi-powered chatbot with local dataset context and session history support."""

    def __init__(
        self,
        api_manager: APIManager | None = None,
        rag_engine: DatasetRAGEngine | None = None,
        reranker: Any | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize chatbot dependencies."""
        self.api_manager = api_manager or APIManager()
        self.rag_engine = rag_engine or DatasetRAGEngine()
        self.reranker = reranker
        self.model = model

    def answer(
        self,
        user_query: str,
        history: Sequence[Mapping[str, str]] | None = None,
        uploaded_resume_context: str = "",
        internet_context: str = "",
    ) -> str:
        """Answer a user query using local dataset context, optional resume text, and Kimi."""
        # If an uploaded resume context is supplied, add it as a temporary chunk so retrieval can prioritize it
        added_temp = 0
        if uploaded_resume_context and uploaded_resume_context.strip():
            try:
                added_temp = self.rag_engine.add_temporary_chunks([uploaded_resume_context], source="uploaded_resume")
            except Exception:
                added_temp = 0

        retrieved = self.rag_engine.retrieve(user_query, top_k=12)
        if self.reranker is not None and retrieved:
            chunk_dicts = [
                {"row_id": c.row_id, "source": c.source, "text": c.text}
                for c in retrieved
            ]
            reranked = self.reranker.rerank(user_query, chunk_dicts)
            retrieved = [DatasetChunk(source=item.source, row_id=item.row_id, text=item.text, metadata={}) for item in reranked[:5]]

        dataset_context = "\n\n".join(f"Source: {c.source} row {c.row_id}\n{c.text}" for c in retrieved)
        system_prompt = (
            "You are an AI job advisor. Ground answers in the supplied local dataset context when it is relevant. "
            "If the dataset is insufficient, state the limitation and give practical next steps. "
            "Do not claim live internet access unless internet context is explicitly supplied.\n\n"
            f"Local dataset context:\n{dataset_context or 'No relevant local dataset context was retrieved.'}\n\n"
            f"Uploaded resume context:\n{uploaded_resume_context[:5000] or 'None'}\n\n"
            f"Internet context supplied by app:\n{internet_context[:3000] or 'None'}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for item in list(history or [])[-10:]:
            role = item.get("role", "")
            content = item.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_query})
        provider = self.api_manager.preferred_chat_provider()
        model = self.model or self.api_manager.provider_model(provider)
        try:
            return self.api_manager.chat_completion(
                messages=messages,
                provider=provider,
                model=model,
                max_tokens=1200,
                temperature=0.25,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            if isinstance(exc, APIKeyPoolExhaustedError):
                fallback = extractive_summary_from_chunks(user_query, retrieved, top_n=3)
                return fallback.summary
            raise
        finally:
            # ensure temporary resume context is cleared after the request
            try:
                if added_temp:
                    self.rag_engine.clear_temporary_chunks()
            except Exception:
                pass


def build_mock_interview_prompt(target_role: str, latest_answer: str, history: Sequence[Mapping[str, str]]) -> str:
    """Build a grounded multi-turn mock interview prompt for the chatbot."""
    recent_history = "\n".join(f"{item.get('role', '')}: {item.get('content', '')}" for item in list(history)[-6:])
    return (
        "Continue a realistic mock interview. Ask one concise follow-up question after briefly evaluating the latest answer. "
        "Use the target role and conversation history.\n\n"
        f"Target role: {target_role}\nConversation:\n{recent_history}\nLatest answer:\n{latest_answer}"
    )
