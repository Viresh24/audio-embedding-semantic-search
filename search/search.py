"""
Query-time search over the collection built by ingest.py.

Covers the four requirements:
  - list_clips()          : list clips + transcripts
  - text_search()         : hybrid lexical (BM25 over transcripts) + semantic (CLAP)
  - audio_to_audio_search(): "search by example" -- find clips like this clip
  - event_search()        : zero-shot "dog barking" / "coughing" style queries

Run from the project root (so the embeddings and ingestion packages are importable):
  python -m search.search text_search "question about pricing"
  python -m search.search event_search "dog barking" --top-k 5
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from embeddings.embeddings import CLAPEmbedder
from ingestion.events import embed_event_query
from ingestion.ingest import COLLECTION


@dataclass
class SearchResult:
    clip_id: str
    file_path: str
    transcript: str
    tags: list
    score: float


def _to_result(point, score) -> SearchResult:
    return SearchResult(
        clip_id=str(point.id),
        file_path=point.payload["file_path"],
        transcript=point.payload["transcript"],
        tags=point.payload.get("tags", []),
        score=score,
    )


class AudioSearchEngine:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=qdrant_url)
        self.clap = CLAPEmbedder()
        self._bm25 = None
        self._bm25_ids = None
        self._build_bm25_index()

    def _all_points(self):
        # Simple full scroll; fine for moderate collection sizes. For large
        # collections, page with `scroll(limit=..., offset=...)`.
        points, _ = self.client.scroll(collection_name=COLLECTION, limit=10_000, with_payload=True)
        return points

    def _build_bm25_index(self):
        points = self._all_points()
        corpus = [p.payload["transcript"].lower().split() for p in points]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_ids = [p.id for p in points]

    def list_clips(self) -> list[SearchResult]:
        return [_to_result(p, score=0.0) for p in self._all_points()]

    def text_search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> list[SearchResult]:
        """
        Hybrid search: blends CLAP semantic similarity with BM25 lexical rank.
        alpha=1.0 -> pure semantic, alpha=0.0 -> pure lexical.
        """
        # Semantic leg: embed the query, search against the "text" vector space
        # (transcript embeddings), which is more reliable for spoken-content
        query_vec = self.clap.embed_text([query])[0]
        semantic_hits = self.client.search(
            collection_name=COLLECTION,
            query_vector=("text", query_vec.tolist()),
            limit=top_k * 3,  # overfetch, then re-rank with the blend below
        )
        semantic_scores = {hit.id: hit.score for hit in semantic_hits}

        # Lexical leg: BM25 over transcripts.
        lexical_scores = {}
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query.lower().split())
            max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
            lexical_scores = {
                pid: s / max_score for pid, s in zip(self._bm25_ids, scores)
            }

        # Blend and rank.
        all_ids = set(semantic_scores) | set(lexical_scores)
        blended = {
            pid: alpha * semantic_scores.get(pid, 0.0) + (1 - alpha) * lexical_scores.get(pid, 0.0)
            for pid in all_ids
        }
        top_ids = sorted(blended, key=blended.get, reverse=True)[:top_k]

        points = self.client.retrieve(collection_name=COLLECTION, ids=top_ids, with_payload=True)
        by_id = {p.id: p for p in points}
        return [_to_result(by_id[i], blended[i]) for i in top_ids if i in by_id]

    def audio_to_audio_search(self, query_audio_path: str, top_k: int = 10) -> list[SearchResult]:
        """Search by example: find clips that sound like the given clip."""
        query_vec = self.clap.embed_audio([query_audio_path])[0]
        hits = self.client.search(
            collection_name=COLLECTION,
            query_vector=("audio", query_vec.tolist()),
            limit=top_k,
        )
        return [_to_result(h, h.score) for h in hits]

    def event_search(self, event_description: str, top_k: int = 10, threshold: float = 0.15) -> list[SearchResult]:
        """
        Zero-shot event detection, e.g. event_search("dog barking").
        Builds the query vector the same way as ingest-time label vectors
        (aliases x prompt templates, averaged), so known labels match the
        precomputed `tags` field and free-text queries get the same ensembling.
        """
        query_vec = embed_event_query(self.clap, event_description)
        hits = self.client.search(
            collection_name=COLLECTION,
            query_vector=("audio", query_vec.tolist()),
            limit=top_k,
            score_threshold=threshold,
        )
        return [_to_result(h, h.score) for h in hits]

    def close(self):
        self.client.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search the audio clip index.")
    parser.add_argument("type", choices=["text_search", "event_search"], help="Which search to run.")
    parser.add_argument(
        "query",
        help="For text_search: what the speech is about. For event_search: a sound event, e.g. 'dog barking'.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Maximum number of results.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    engine = AudioSearchEngine(qdrant_url=args.qdrant_url)
    try:
        print(f"\n-- {args.type}: {args.query!r} --")
        if args.type == "text_search":
            results = engine.text_search(args.query, top_k=args.top_k)
            for r in results:
                print(f"  [{r.score:.3f}] {r.transcript[:60]!r}  tags={r.tags}")
        else:
            results = engine.event_search(args.query, top_k=args.top_k)
            for r in results:
                print(f"  [{r.score:.3f}] {r.file_path}  tags={r.tags}")

        if not results:
            print("  (no results)")
    finally:
        engine.close()
