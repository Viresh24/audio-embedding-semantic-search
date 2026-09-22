"""
Evaluate CLAP event_search against gold relevance defined in sentence-embedding space.

A clip is relevant to a prompt if the prompt is close to any of that clip's
dataset-native labels (ESC-50 category, FLEURS "speech", ...). CLAP still ranks audio; the
sentence transformer only decides which clips should have been returned.

Run from the project root after ingesting into audio_clips_eval:

  python -m ingestion.ingest url esc50 --num-samples 80 --collection audio_clips_eval
  python -m ingestion.ingest url fleurs --num-samples 20 --collection audio_clips_eval
  python -m eval.event_search --collection audio_clips_eval --top-k 1,5,10 --tau 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone

from qdrant_client import QdrantClient

from eval.labels import LabelEncoder
from eval.metrics import mean, ranking_metrics
from search.search import AudioSearchEngine

Ns = (1, 5, 10)


def scroll_index(qdrant_url: str, collection: str) -> list[dict]:
    client = QdrantClient(url=qdrant_url)
    try:
        points, _ = client.scroll(collection_name=collection, limit=10_000, with_payload=True)
    finally:
        client.close()
    clips = []
    for p in points:
        payload = p.payload or {}
        labels = payload.get("labels") or []
        if not labels:
            continue
        clips.append({"id": str(p.id), "labels": labels})
    return clips


def evaluate(
    collection: str,
    queries: list[str],
    tau: float,
    top_m: int | None,
    ns: tuple[int, ...],
    threshold: float,
    encoder_name: str,
    qdrant_url: str,
    seed: int = 0,
) -> dict:
    clips = scroll_index(qdrant_url, collection)
    if not clips:
        raise SystemExit(f"No clips with labels in '{collection}'. Ingest ESC-50 into that collection first.")

    by_label: dict[str, list[str]] = defaultdict(list)
    for c in clips:
        for lab in c["labels"]:
            by_label[lab].append(c["id"])
    labels = sorted(by_label)
    id_to_labels = {c["id"]: c["labels"] for c in clips}

    encoder = LabelEncoder(encoder_name)
    engine = AudioSearchEngine(qdrant_url=qdrant_url, collection=collection)
    max_n = max(ns)
    rng = random.Random(seed)
    all_ids = [c["id"] for c in clips]

    per_query = []
    skipped = []
    try:
        for q in queries:
            matched = encoder.matching_labels(q, labels, tau=tau, top_m=top_m)
            relevant = {cid for lab in matched for cid in by_label[lab]}
            if not relevant:
                skipped.append({"query": q, "reason": "no matching labels", "label_sims": encoder.label_sims(q, labels)})
                continue

            # A clip with several matching labels gets the gain of its closest one.
            gain = {cid: max(matched[lab] for lab in id_to_labels[cid] if lab in matched) for cid in relevant}
            hits = engine.event_search(q, top_k=max_n, threshold=threshold)
            ranked = [h.clip_id for h in hits]
            row = {
                "query": q,
                "matching_labels": matched,
                "n_relevant": len(relevant),
                "first_fp_labels": next((id_to_labels.get(cid) for cid in ranked if cid not in relevant), None),
            }
            row.update(ranking_metrics(ranked, relevant, gain, ns))

            random_ranked = list(all_ids)
            rng.shuffle(random_ranked)
            rand_m = ranking_metrics(random_ranked, relevant, gain, ns)
            row["random_mrr"] = rand_m["mrr"]
            row["random_recall@5"] = rand_m.get("recall@5", 0.0)

            # Text-to-text upper bound: clips whose label matched, ordered by sim(q, L).
            label_ranked = sorted(relevant, key=lambda cid: gain[cid], reverse=True)
            bound = ranking_metrics(label_ranked, relevant, gain, ns)
            row["label_index_mrr"] = bound["mrr"]
            row["label_index_recall@5"] = bound.get("recall@5", 0.0)
            per_query.append(row)
    finally:
        engine.close()

    macro = {}
    if per_query:
        for n in ns:
            for key in (f"recall@{n}", f"precision@{n}", f"hit@{n}", f"ndcg@{n}"):
                macro[key] = mean(per_query, key)
        macro["mrr"] = mean(per_query, "mrr")
        macro["random_mrr"] = mean(per_query, "random_mrr")
        macro["label_index_mrr"] = mean(per_query, "label_index_mrr")

    return {
        "collection": collection,
        "encoder": encoder_name,
        "tau": tau,
        "top_m_labels": top_m,
        "threshold": threshold,
        "n_clips": len(clips),
        "labels": labels,
        "n_queries_scored": len(per_query),
        "skipped": skipped,
        "macro": macro,
        "per_query": per_query,
    }


def _print_table(report: dict, ns: tuple[int, ...]):
    print(f"\ncollection={report['collection']}  clips={report['n_clips']}  "
          f"scored={report['n_queries_scored']}  skipped={len(report['skipped'])}  tau={report['tau']}")
    macro = report["macro"]
    if macro:
        cols = ["mrr"] + [f"recall@{n}" for n in ns] + [f"precision@{n}" for n in ns] + [f"ndcg@{n}" for n in ns]
        print("macro  " + "  ".join(f"{k}={macro[k]:.3f}" for k in cols if k in macro))
        print(f"       random_mrr={macro['random_mrr']:.3f}  label_index_mrr={macro['label_index_mrr']:.3f}")
    for row in report["per_query"]:
        labs = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(row["matching_labels"].items(), key=lambda kv: -kv[1])[:4])
        print(f"  {row['query']!r:28s}  R={row['n_relevant']:3d}  mrr={row['mrr']:.3f}  "
              f"R@5={row['recall@5']:.2f}  P@5={row['precision@5']:.2f}  match=[{labs}]  "
              f"fp={row['first_fp_labels']!r}")
    for s in report["skipped"]:
        print(f"  skipped {s['query']!r}: {s['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate event_search with ST-defined label relevance.")
    parser.add_argument("--collection", default="audio_clips_eval")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--top-k", default="1,5,10", help="Comma-separated cutoffs, e.g. 1,5,10.")
    parser.add_argument("--tau", type=float, default=0.45, help="Min ST cosine between prompt and a clip label.")
    parser.add_argument("--top-m-labels", type=int, default=None, help="If set, take this many nearest labels instead of tau.")
    parser.add_argument("--threshold", type=float, default=0.0, help="event_search score_threshold (0 = no cutoff).")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--queries", default=os.path.join(os.path.dirname(__file__), "event_queries.json"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()

    ns = tuple(int(x) for x in args.top_k.split(",") if x.strip())
    with open(args.queries) as f:
        queries = json.load(f)

    report = evaluate(
        collection=args.collection,
        queries=queries,
        tau=args.tau,
        top_m=args.top_m_labels,
        ns=ns,
        threshold=args.threshold,
        encoder_name=args.encoder,
        qdrant_url=args.qdrant_url,
    )
    _print_table(report, ns)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out_dir, f"event_search_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
