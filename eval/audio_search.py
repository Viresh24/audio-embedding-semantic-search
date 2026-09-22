"""
Evaluate audio_to_audio_search ("search by example") with held-out clips.

Index one split of a labelled dataset, then run this against a different
split (test/validation). Each held-out clip is streamed from Hugging Face,
written to a wav, and passed to audio_to_audio_search as the query. A result
is relevant if it shares a label with the query clip. Held-out clips are
never added to the index.

Run from the project root:

  python -m ingestion.ingest url audioset --split train --num-samples 2000 --collection audio_clips_eval
  python -m eval.audio_search audioset --split test --num-samples 200 --collection audio_clips_eval
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

from qdrant_client import QdrantClient

from eval.metrics import mean, ranking_metrics
from ingestion.ingest import clip_file_prefix, labels_for_row, resolve_hf_dataset, stream_hf_dataset, write_wav
from search.search import AudioSearchEngine


def scroll_index(qdrant_url: str, collection: str) -> tuple[list[dict], set[tuple]]:
    """Returns (clips with labels as {id, labels}, the (dataset, config, split) triples they came from)."""
    client = QdrantClient(url=qdrant_url)
    try:
        if not client.collection_exists(collection):
            raise SystemExit(f"Collection '{collection}' doesn't exist. Ingest the index split first.")
        clips, sources, offset = [], set(), None
        while True:
            points, offset = client.scroll(collection_name=collection, limit=512, offset=offset, with_payload=True)
            for p in points:
                payload = p.payload or {}
                sources.add((payload.get("dataset"), payload.get("config"), payload.get("split")))
                if payload.get("labels"):
                    clips.append({"id": str(p.id), "labels": payload["labels"]})
            if offset is None:
                return clips, sources
    finally:
        client.close()


def _average_precision(ranked: list[str], relevant: set[str]) -> float:
    """AP over the ranked list, normalized by how many relevant clips could fit in it."""
    found, total = 0, 0.0
    for i, cid in enumerate(ranked):
        if cid in relevant:
            found += 1
            total += found / (i + 1)
    denom = min(len(relevant), len(ranked))
    return total / denom if denom else 0.0


def _vote(hit_labels: list[list[str]]) -> str | None:
    """Most common label among the hits; ties go to the label seen first (the closer hit)."""
    counts = Counter(lab for labels in hit_labels for lab in labels)
    return counts.most_common(1)[0][0] if counts else None


def evaluate(
    dataset_name: str,
    split: str,
    num_samples: int,
    collection: str,
    ns: tuple[int, ...],
    vote_k: int,
    qdrant_url: str,
    config: str | None = None,
    audio_column: str | None = None,
    transcript_column: str | None = None,
    cache_dir: str = "/tmp/hf_audio_clips",
) -> dict:
    ds_info = resolve_hf_dataset(dataset_name, config, audio_column, transcript_column)
    index, sources = scroll_index(qdrant_url, collection)
    if not index:
        raise SystemExit(f"No labelled clips in '{collection}'.")
    if (ds_info["repo_id"], ds_info["config"], split) in sources:
        raise SystemExit(
            f"'{collection}' already holds {ds_info['repo_id']} {split} clips; "
            "query with a split that wasn't indexed."
        )

    by_label: dict[str, set[str]] = defaultdict(set)
    for c in index:
        for lab in c["labels"]:
            by_label[lab].add(c["id"])

    os.makedirs(cache_dir, exist_ok=True)
    rows = stream_hf_dataset(ds_info, split)
    file_prefix = clip_file_prefix(ds_info, split)
    max_n = max(ns)

    engine = AudioSearchEngine(qdrant_url=qdrant_url, collection=collection)
    per_query, skipped = [], []
    try:
        for i, row in enumerate(rows):
            if num_samples and i >= num_samples:
                break
            try:
                labels = labels_for_row(row, ds_info)
                relevant = set().union(*(by_label.get(lab, set()) for lab in labels))
                if not relevant:
                    skipped.append({"row": i, "labels": labels, "reason": "no index clips share its labels"})
                    continue

                wav_path = os.path.join(cache_dir, f"{file_prefix}_{i:04d}.wav")
                write_wav(row, ds_info["audio_column"], wav_path)
                hits = engine.audio_to_audio_search(wav_path, top_k=max(max_n, vote_k))
            except Exception as e:
                skipped.append({"row": i, "reason": f"error: {e}"})
                continue

            ranked = [h.clip_id for h in hits][:max_n]
            hit_labels = {h.clip_id: h.labels for h in hits}
            predicted = _vote([h.labels for h in hits[:vote_k]])
            row_report = {
                "row": i,
                "file_path": wav_path,
                "labels": labels,
                "n_relevant": len(relevant),
                "predicted_label": predicted,
                f"vote_accuracy@{vote_k}": 1.0 if predicted in labels else 0.0,
                f"ap@{max_n}": _average_precision(ranked, relevant),
                "random_precision": len(relevant) / len(index),
                "wrong_labels": [lab for cid in ranked if cid not in relevant for lab in hit_labels[cid]],
            }
            row_report.update(ranking_metrics(ranked, relevant, {cid: 1.0 for cid in relevant}, ns))
            per_query.append(row_report)
            print(f"  [{i + 1}/{num_samples or 'all'}] {labels}  predicted={predicted!r}  "
                  f"P@{max_n}={row_report[f'precision@{max_n}']:.2f}")
    finally:
        engine.close()

    metric_keys = [k for k, v in per_query[0].items() if isinstance(v, float)] if per_query else []
    macro = {k: mean(per_query, k) for k in metric_keys}

    rows_by_label = defaultdict(list)
    for r in per_query:
        for lab in r["labels"]:
            rows_by_label[lab].append(r)
    per_label = {}
    for lab, label_rows in rows_by_label.items():
        confusions = Counter(w for r in label_rows for w in r["wrong_labels"])
        per_label[lab] = {
            "n_queries": len(label_rows),
            **{k: mean(label_rows, k) for k in metric_keys},
            "top_confusions": confusions.most_common(3),
        }

    return {
        "collection": collection,
        "n_index_clips": len(index),
        "query_dataset": ds_info["repo_id"],
        "query_config": ds_info["config"],
        "query_split": split,
        "n_queries_scored": len(per_query),
        "vote_k": vote_k,
        "skipped": skipped,
        "macro": macro,
        "per_label": per_label,
        "per_query": per_query,
    }


def _print_report(report: dict, ns: tuple[int, ...]):
    max_n, vote_k = max(ns), report["vote_k"]
    print(f"\nindex={report['collection']} ({report['n_index_clips']} clips)  "
          f"queries={report['query_dataset']} {report['query_split']} "
          f"(scored {report['n_queries_scored']}, skipped {len(report['skipped'])})")
    macro = report["macro"]
    if not macro:
        return
    cols = ["mrr", f"ap@{max_n}", f"vote_accuracy@{vote_k}"] + [f"precision@{n}" for n in ns] + [
        f"recall@{n}" for n in ns
    ]
    print("macro  " + "  ".join(f"{k}={macro[k]:.3f}" for k in cols if k in macro))
    print(f"       random_precision={macro['random_precision']:.3f}")

    print(f"\nper label, worst precision@{max_n} first:")
    for lab, m in sorted(report["per_label"].items(), key=lambda kv: kv[1][f"precision@{max_n}"]):
        conf = ", ".join(f"{w} x{c}" for w, c in m["top_confusions"])
        print(f"  {lab[:28]:28s} n={m['n_queries']:3d}  P@{max_n}={m[f'precision@{max_n}']:.2f}  "
              f"mrr={m['mrr']:.2f}  vote@{vote_k}={m[f'vote_accuracy@{vote_k}']:.2f}  confused_with=[{conf}]")
    reasons = Counter(s["reason"] for s in report["skipped"])
    for reason, n in reasons.most_common():
        print(f"  skipped {n}: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio_to_audio_search with held-out clips as queries.")
    parser.add_argument("dataset", help="Dataset short name (e.g. audioset) or a Hugging Face repo ID.")
    parser.add_argument("--split", default="test", help="Held-out split to query with; must not be indexed.")
    parser.add_argument("--config", default=None, help="Dataset config (defaults per dataset).")
    parser.add_argument("--num-samples", type=int, default=200, help="Query clips to stream (0 = all).")
    parser.add_argument("--audio-column", default=None, help="Override the audio column name.")
    parser.add_argument("--transcript-column", default=None, help="Override the transcript column name.")
    parser.add_argument("--collection", default="audio_clips_eval", help="Index collection being searched.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--top-k", default="1,5,10", help="Comma-separated cutoffs, e.g. 1,5,10.")
    parser.add_argument("--vote-k", type=int, default=5, help="Hits that vote on each query's predicted label.")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = parser.parse_args()

    try:
        resolve_hf_dataset(args.dataset, args.config, args.audio_column, args.transcript_column)
    except ValueError as e:
        parser.error(str(e))

    ns = tuple(int(x) for x in args.top_k.split(",") if x.strip())
    report = evaluate(
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        collection=args.collection,
        ns=ns,
        vote_k=args.vote_k,
        qdrant_url=args.qdrant_url,
        config=args.config,
        audio_column=args.audio_column,
        transcript_column=args.transcript_column,
    )
    _print_report(report, ns)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.out_dir, f"audio_search_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
