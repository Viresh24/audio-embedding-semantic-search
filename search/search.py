"""
Query-time search over the collection built by ingest.py.

  - list_clips()           : list clips + transcripts
  - audio_to_audio_search(): "search by example" -- find clips like this clip
  - event_search()         : zero-shot "dog barking" / "coughing" style queries

Run from the project root (so the embeddings and ingestion packages are importable):
  python -m search.search audio_to_audio_search /path/to/clip.wav --top-k 5
  python -m search.search event_search "dog barking" --top-k 5
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from embeddings.embeddings import CLAPEmbedder
from ingestion.events import embed_event_query
from ingestion.ingest import COLLECTION


@dataclass
class SearchResult:
    clip_id: str
    file_path: str
    transcript: str
    labels: list
    score: float


def _to_result(point, score) -> SearchResult:
    return SearchResult(
        clip_id=str(point.id),
        file_path=point.payload["file_path"],
        transcript=point.payload["transcript"],
        labels=point.payload.get("labels", []),
        score=score,
    )


class AudioSearchEngine:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection: str = COLLECTION):
        self.collection = collection
        self.client = QdrantClient(url=qdrant_url)
        self.clap = CLAPEmbedder()

    def _all_points(self):
        # Simple full scroll; fine for moderate collection sizes. For large
        # collections, page with `scroll(limit=..., offset=...)`.
        points, _ = self.client.scroll(collection_name=self.collection, limit=10_000, with_payload=True)
        return points

    def list_clips(self) -> list[SearchResult]:
        return [_to_result(p, score=0.0) for p in self._all_points()]

    def audio_to_audio_search(self, query_audio_path: str, top_k: int = 10) -> list[SearchResult]:
        """Search by example: find clips that sound like the given clip."""
        query_vec = self.clap.embed_audio([query_audio_path])[0]
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=("audio", query_vec.tolist()),
            limit=top_k,
        )
        return [_to_result(h, h.score) for h in hits]

    def event_search(self, event_description: str, top_k: int = 10, threshold: float = 0.15) -> list[SearchResult]:
        """
        Zero-shot event detection, e.g. event_search("dog barking").
        Builds the query vector the same way as ingest-time label vectors
        (aliases x prompt templates, averaged), so known event labels and
        free-text queries get the same ensembling.
        """
        query_vec = embed_event_query(self.clap, event_description)
        hits = self.client.search(
            collection_name=self.collection,
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
    parser.add_argument("type", choices=["audio_to_audio_search", "event_search"], help="Which search to run.")
    parser.add_argument(
        "query",
        help="For audio_to_audio_search: path to an audio clip. For event_search: a sound event, e.g. 'dog barking'.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Maximum number of results.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default=COLLECTION, help="Qdrant collection name.")
    args = parser.parse_args()

    engine = AudioSearchEngine(qdrant_url=args.qdrant_url, collection=args.collection)
    try:
        print(f"\n-- {args.type}: {args.query!r} --")
        if args.type == "audio_to_audio_search":
            results = engine.audio_to_audio_search(args.query, top_k=args.top_k)
        else:
            results = engine.event_search(args.query, top_k=args.top_k)
        for r in results:
            print(f"  [{r.score:.3f}] {r.file_path}  labels={r.labels}")

        if not results:
            print("  (no results)")
    finally:
        engine.close()
