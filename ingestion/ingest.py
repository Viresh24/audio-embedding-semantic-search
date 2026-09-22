"""
Ingestion pipeline:

  1. A speech dataset streamed from Hugging Face (ingest_from_url), e.g.
     FLEURS or VoxPopuli -- uses the dataset's own ground-truth transcript,
     no ASR needed. See HF_DATASETS for the supported short names.

For each clip, either path does the same core steps:
  a. Get a transcript (Whisper, or the dataset's provided transcription).
  b. Embed the raw audio with CLAP -> audio_vector, used for audio-to-audio
     search and zero-shot sound-event scoring.
  c. Embed the transcript with CLAP -> text_vector, used for text->audio
     semantic search.
  d. Store the dataset's own labels for the clip as-is in `labels` (e.g. an
     ESC-50 category, AudioSet's human-readable labels, or "speech" for
     speech datasets).
  e. Score the clip against a fixed set of sound-event labels (see
     ingestion/events.py) and store the scores.
  f. Upsert everything into the vector DB.

NOTE: FLEURS and VoxPopuli are all speech -- there's no barking/coughing/
applause etc. in them. event_search() will have nothing relevant to match
against speech-only data. Mix in a handful of clips from a sound-event set
(e.g. ESC-50, Freesound) alongside it if you need to demo event_search.

Run from the project root (so the embeddings package is importable):
  python -m ingestion.ingest dir /path/to/clips_dir
  python -m ingestion.ingest url fleurs --config en_us --num-samples 25
  python -m ingestion.ingest url voxpopuli --config en --num-samples 25
  python -m ingestion.ingest url esc50 --num-samples 100 --collection audio_clips_eval
  python -m ingestion.ingest url audioset --split train --num-samples 2000 --collection audio_clips_eval
  python -m ingestion.ingest clear fleurs   # remove one dataset's clips
  python -m ingestion.ingest clear          # drop the whole collection
  python -m ingestion.ingest calibrate      # per-label event stats from stored clips
  python -m ingestion.ingest rescore        # re-score stored clips, no re-embedding
"""

import glob
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    SetPayload,
    SetPayloadOperation,
    VectorParams,
)

from embeddings.embeddings import CLAPEmbedder, Transcriber
from ingestion.events import (
    build_label_vectors,
    compute_calibration,
    event_payload,
    load_calibration,
    save_calibration,
    score_clip,
)

COLLECTION = "audio_clips"
VECTOR_DIM = 512  # CLAP's output dimension
MIN_CALIBRATION_CLIPS = 200
UPSERT_BATCH_SIZE = 500

# Short dataset names accepted by ingest_from_url. Each dataset names its
# transcript column and language configs differently, so those live here.
# Any other "org/name" Hugging Face repo ID can be passed directly as long as
# its column names are supplied.
HF_DATASETS = {
    "fleurs": {
        "repo_id": "google/fleurs",
        "default_config": "en_us",  # fr_fr, de_de, es_419, ...
        "default_split": "validation",
        "audio_column": "audio",
        "transcript_column": "raw_transcription",
        "label_default": "speech",
    },
    "voxpopuli": {
        "repo_id": "facebook/voxpopuli",
        "default_config": "en",  # de, fr, es, en_accented, ...
        "default_split": "validation",
        "audio_column": "audio",
        "transcript_column": "raw_text",
        "label_default": "speech",
    },
    "esc50": {
        "repo_id": "ashraq/esc50",
        "default_config": None,
        "default_split": "train",
        "audio_column": "audio",
        "transcript_column": "category",
        "label_column": "category",
    },
    # Balanced AudioSet: ~18.7k train / ~17.1k test 10s clips, each with a list
    # of human-readable labels. No speech transcripts.
    "audioset": {
        "repo_id": "agkphysics/AudioSet",
        "default_config": "balanced",
        "default_split": "train",
        "audio_column": "audio",
        "transcript_column": None,
        "label_column": "human_labels",
    },
}


def humanize_label(text: str) -> str:
    """Turn dataset slugs like keyboard_typing into words a sentence encoder can match."""
    return str(text).replace("_", " ").strip()


def labels_for_row(row: dict, spec: dict) -> list[str]:
    """The row's labels as the dataset gives them; a list-valued column (AudioSet) is kept whole."""
    col = spec.get("label_column")
    if col and row.get(col) is not None:
        value = row[col]
        return [str(v) for v in value] if isinstance(value, list) else [humanize_label(value)]
    if spec.get("label_default"):
        return [spec["label_default"]]
    return [humanize_label(row[spec["transcript_column"]])]


def get_or_create_collection(client: QdrantClient, collection: str = COLLECTION):
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={
                # Two named vectors per point so audio-to-audio and text-to-audio
                # search can each hit the vector space they were designed for.
                "audio": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                "text": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
        )


def clear_dataset(
    dataset_name: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    collection: str = COLLECTION,
) -> int:
    """
    Removes clips from the collection and returns how many were removed.

    With `dataset_name` (the short name or repo ID passed to ingest_from_url,
    stored as the `source` payload field), only that dataset's clips are
    deleted. Without it, the whole collection is dropped; the next ingest
    recreates it.
    """
    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(collection):
        return 0

    if dataset_name is None:
        count = client.count(collection_name=collection, exact=True).count
        client.delete_collection(collection_name=collection)
        return count

    source_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=dataset_name))])
    count = client.count(collection_name=collection, count_filter=source_filter, exact=True).count
    if count:
        client.delete(collection_name=collection, points_selector=FilterSelector(filter=source_filter), wait=True)
    return count


def _iter_points(
    client: QdrantClient,
    scroll_filter: Filter | None = None,
    page_size: int = 256,
    collection: str = COLLECTION,
):
    """Yields pages of points with their audio vectors and payloads."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=["audio"],
        )
        if points:
            yield points
        if offset is None:
            return


def _source_filter(source: str | None) -> Filter | None:
    if source is None:
        return None
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


def calibrate_events(
    source: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    collection: str = COLLECTION,
) -> dict | None:
    """
    Computes per-label mean/std of event cosine scores from the audio vectors
    already stored in the collection, and saves them for z-scoring.
    `source` restricts the reference set to one dataset. Returns the stats,
    or None if there are no clips.
    """
    client = QdrantClient(url=qdrant_url)
    try:
        if not client.collection_exists(collection):
            print(f"Collection '{collection}' doesn't exist; nothing to calibrate on.")
            return None

        vecs, sources = [], {}
        for page in _iter_points(client, _source_filter(source), collection=collection):
            for p in page:
                vecs.append(p.vector["audio"])
                src = (p.payload or {}).get("source", "unknown")
                sources[src] = sources.get(src, 0) + 1
        if not vecs:
            print("No clips matched; nothing to calibrate on.")
            return None

        if len(vecs) < MIN_CALIBRATION_CLIPS:
            print(f"WARNING: only {len(vecs)} clips; per-label stats are noisy below ~{MIN_CALIBRATION_CLIPS}.")
        if len(sources) == 1:
            print(
                f"WARNING: all clips come from one source ({next(iter(sources))}); z-scores will be relative "
                "to that source only. Mix in other audio (e.g. ESC-50) for a more general baseline."
            )

        clap = CLAPEmbedder()
        names, label_matrix = build_label_vectors(clap)
        stats = compute_calibration(names, label_matrix, vecs, sources)
        save_calibration(stats)
        print(f"Calibrated {len(names)} labels on {len(vecs)} clips from {sources}.")
        return stats
    finally:
        client.close()


def rescore_clips(
    source: str | None = None,
    qdrant_url: str = "http://localhost:6333",
    collection: str = COLLECTION,
) -> int:
    """
    Re-scores existing clips from their stored audio vectors and rewrites
    their event payload fields. No audio is re-embedded. Returns the count.
    """
    client = QdrantClient(url=qdrant_url)
    try:
        if not client.collection_exists(collection):
            return 0

        clap = CLAPEmbedder()
        names, label_matrix = build_label_vectors(clap)
        logit_scale = clap.logit_scale
        calib = load_calibration()
        print(f"Rescoring {'with' if calib else 'without'} calibration stats.")

        count = 0
        for page in _iter_points(client, _source_filter(source), collection=collection):
            ops = []
            for p in page:
                result = score_clip(p.vector["audio"], names, label_matrix, logit_scale, calib)
                ops.append(SetPayloadOperation(set_payload=SetPayload(payload=event_payload(result), points=[p.id])))
            client.batch_update_points(collection_name=collection, update_operations=ops, wait=True)
            count += len(ops)
        return count
    finally:
        client.close()


def resolve_hf_dataset(
    dataset_name: str,
    config: str | None = None,
    audio_column: str | None = None,
    transcript_column: str | None = None,
) -> dict:
    """
    Turns a short name from HF_DATASETS (e.g. "fleurs") or a raw "org/name"
    Hugging Face repo ID into {repo_id, config, audio_column, transcript_column}.
    Explicit arguments override the HF_DATASETS defaults.
    """
    if dataset_name in HF_DATASETS:
        spec = HF_DATASETS[dataset_name]
    elif "/" in dataset_name:
        spec = {
            "repo_id": dataset_name,
            "default_config": None,
            "default_split": "validation",
            "audio_column": "audio",
            "transcript_column": None,
        }
    else:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Use one of {sorted(HF_DATASETS)} "
            "or a full Hugging Face repo ID like 'org/name'."
        )

    resolved = {
        "repo_id": spec["repo_id"],
        "config": config or spec.get("default_config"),
        "split": spec.get("default_split") or "validation",
        "audio_column": audio_column or spec["audio_column"],
        "transcript_column": transcript_column or spec["transcript_column"],
        "label_column": spec.get("label_column"),
        "label_default": spec.get("label_default"),
    }
    if dataset_name not in HF_DATASETS and not resolved["transcript_column"]:
        raise ValueError(f"{dataset_name!r} isn't in HF_DATASETS, so transcript_column must be given.")
    return resolved


def stream_hf_dataset(ds_info: dict, split: str):
    """Streams one split with the audio column left undecoded (raw bytes; see write_wav)."""
    from datasets import Audio, load_dataset

    repo_id, config = ds_info["repo_id"], ds_info["config"]
    print(f"Streaming {repo_id} ({config or '-'}/{split}) from Hugging Face...")
    if config:
        ds = load_dataset(repo_id, config, split=split, streaming=True)
    else:
        ds = load_dataset(repo_id, split=split, streaming=True)
    return ds.cast_column(ds_info["audio_column"], Audio(decode=False))


def clip_file_prefix(ds_info: dict, split: str) -> str:
    # Includes dataset, config and split so clips from different sources never
    # share a wav path, and therefore never share a Qdrant point ID.
    return f"{ds_info['repo_id'].replace('/', '__')}_{ds_info['config'] or 'default'}_{split}"


def write_wav(row: dict, audio_column: str, wav_path: str):
    """Decodes a row's raw audio bytes with soundfile and writes them to wav_path."""
    import io

    import soundfile as sf

    audio_array, sampling_rate = sf.read(io.BytesIO(row[audio_column]["bytes"]))
    sf.write(wav_path, audio_array, sampling_rate)


def ingest_from_url(
    dataset_name: str,
    config: str | None = None,
    split: str | None = None,
    num_samples: int = 25,
    qdrant_url: str = "http://localhost:6333",
    audio_column: str | None = None,
    transcript_column: str | None = None,
    cache_dir: str = "/tmp/hf_audio_clips",
    collection: str = COLLECTION,
):
    """
    Streams a speech dataset from Hugging Face (no full-dataset download),
    writes clips from `split` to local wav files, and upserts them in batches
    of UPSERT_BATCH_SIZE. A sample that fails to decode, embed, or upsert is
    skipped; the run prints ingested vs skipped counts at the end.

    `dataset_name` is a short name from HF_DATASETS ("fleurs", "voxpopuli")
    or any "org/name" repo ID. `config` is the dataset's subset, which for
    these datasets is the language (en_us for FLEURS, en for VoxPopuli).
    `num_samples=0` ingests every row.

    Requires: pip install datasets soundfile. No Hugging Face login needed for
    the built-in datasets, though setting HF_TOKEN raises rate limits.

    The audio column is read undecoded (raw bytes) and decoded with soundfile,
    because datasets>=4 otherwise requires torchcodec to decode audio.

    The dataset's transcript is used as-is (no Whisper) since it's already a
    human-verified ground truth -- faster and more accurate than re-running ASR.
    """
    ds_info = resolve_hf_dataset(dataset_name, config, audio_column, transcript_column)
    repo_id, config = ds_info["repo_id"], ds_info["config"]
    split = split or ds_info["split"]
    audio_column, transcript_column = ds_info["audio_column"], ds_info["transcript_column"]

    os.makedirs(cache_dir, exist_ok=True)
    ds = stream_hf_dataset(ds_info, split)
    file_prefix = clip_file_prefix(ds_info, split)

    clap = CLAPEmbedder()
    client = QdrantClient(url=qdrant_url)
    get_or_create_collection(client, collection)
    label_names, label_matrix = build_label_vectors(clap)
    logit_scale = clap.logit_scale
    calib = load_calibration()

    def flush(batch: list) -> tuple[int, int]:
        """Upserts a batch; on failure retries one point at a time. Returns (ok, skipped)."""
        if not batch:
            return 0, 0
        try:
            client.upsert(collection_name=collection, points=batch)
            return len(batch), 0
        except Exception as e:
            print(f"  batch upsert failed ({e}); retrying points individually")
            ok = skipped_pts = 0
            for point in batch:
                try:
                    client.upsert(collection_name=collection, points=[point])
                    ok += 1
                except Exception as point_err:
                    skipped_pts += 1
                    print(f"  skip upsert {point.id}: {point_err}")
            return ok, skipped_pts

    points = []
    ingested = skipped = 0
    try:
        for i, row in enumerate(ds):
            if num_samples and i >= num_samples:
                break
            try:
                transcript = (row[transcript_column] or "").strip() if transcript_column else ""
                labels = labels_for_row(row, ds_info)

                wav_path = os.path.join(cache_dir, f"{file_prefix}_{i:04d}.wav")
                write_wav(row, audio_column, wav_path)

                clip_id = str(uuid.uuid5(uuid.NAMESPACE_URL, wav_path))
                audio_vec = clap.embed_audio([wav_path])[0]
                text_vec = clap.embed_text([transcript])[0] if transcript else audio_vec
                events = event_payload(score_clip(audio_vec, label_names, label_matrix, logit_scale, calib))

                points.append(
                    PointStruct(
                        id=clip_id,
                        vector={"audio": audio_vec.tolist(), "text": text_vec.tolist()},
                        payload={
                            "file_path": wav_path,
                            "transcript": transcript,
                            "labels": labels,
                            "source": dataset_name,
                            "dataset": repo_id,
                            "config": config,
                            "split": split,
                            **events,
                        },
                    )
                )
                print(f"  [{i + 1}/{num_samples or 'all'}] {transcript[:60]!r}  labels={labels}")
                if len(points) >= UPSERT_BATCH_SIZE:
                    ok, fail = flush(points)
                    ingested += ok
                    skipped += fail
                    points = []
            except Exception as e:
                skipped += 1
                print(f"  skip sample {i}: {e}")

        ok, fail = flush(points)
        ingested += ok
        skipped += fail
    finally:
        client.close()

    print(f"Ingested {ingested} {dataset_name} clips into '{collection}'; skipped {skipped}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest audio clips into the search index.")
    parser.add_argument(
        "source",
        choices=["dir", "url", "clear", "calibrate", "rescore"],
        help=(
            "Where to ingest from; 'clear' to remove clips; 'calibrate' to compute event-score stats "
            "from stored clips; 'rescore' to re-score stored clips without re-embedding."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "For 'dir': path to a clips directory. For 'url': a dataset short name "
            f"({', '.join(sorted(HF_DATASETS))}) or a Hugging Face repo ID like 'org/name'. "
            "For 'clear': the dataset to remove; omit it to drop the whole collection. "
            "For 'calibrate' / 'rescore': restrict to one dataset; omit it to use every clip."
        ),
    )
    parser.add_argument("--config", default=None, help="url only: dataset config/language (defaults per dataset).")
    parser.add_argument("--split", default=None, help="url only: dataset split (defaults per dataset).")
    parser.add_argument("--num-samples", type=int, default=25, help="url only: clips to pull (0 = all).")
    parser.add_argument("--audio-column", default=None, help="url only: override the audio column name.")
    parser.add_argument("--transcript-column", default=None, help="url only: override the transcript column name.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default=COLLECTION, help="Qdrant collection name.")
    args = parser.parse_args()

    if args.source == "clear":
        removed = clear_dataset(args.target, qdrant_url=args.qdrant_url, collection=args.collection)
        scope = f"'{args.target}' clips" if args.target else "clips (collection dropped)"
        print(f"Removed {removed} {scope} from '{args.collection}'.")
    elif args.source == "calibrate":
        calibrate_events(args.target, qdrant_url=args.qdrant_url, collection=args.collection)
    elif args.source == "rescore":
        n = rescore_clips(args.target, args.qdrant_url, collection=args.collection)
        print(f"Rescored {n} clips in '{args.collection}'.")
    elif not args.target:
        parser.error(f"source '{args.source}' requires a target")
    else:
        try:
            resolve_hf_dataset(args.target, args.config, args.audio_column, args.transcript_column)
        except ValueError as e:
            parser.error(str(e))
        ingest_from_url(
            dataset_name=args.target,
            config=args.config,
            split=args.split,
            num_samples=args.num_samples,
            qdrant_url=args.qdrant_url,
            audio_column=args.audio_column,
            transcript_column=args.transcript_column,
            collection=args.collection,
        )