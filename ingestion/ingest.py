"""
Ingestion pipeline. Two sources are supported:

  1. A local directory of .wav/.mp3 files (ingest_directory) -- transcribed
     with Whisper.
  2. A speech dataset streamed from Hugging Face (ingest_from_url), e.g.
     FLEURS or VoxPopuli -- uses the dataset's own ground-truth transcript,
     no ASR needed. See HF_DATASETS for the supported short names.

For each clip, either path does the same core steps:
  a. Get a transcript (Whisper, or the dataset's provided transcription).
  b. Embed the raw audio with CLAP -> audio_vector, used for audio-to-audio
     search and zero-shot sound-event tagging.
  c. Embed the transcript with CLAP -> text_vector, used for text->audio
     semantic search.
  d. Tag the clip against a fixed vocabulary of sound events at ingest time,
     so tag filters don't require a CLAP call at query time.
  e. Upsert everything into the vector DB.

NOTE: FLEURS and VoxPopuli are all speech -- there's no barking/coughing/
applause etc. in them. event_search() will have nothing relevant to match
against speech-only data. Mix in a handful of clips from a sound-event set
(e.g. ESC-50, Freesound) alongside it if you need to demo event_search.

Run:
  python ingest.py dir /path/to/clips_dir
  python ingest.py url fleurs --config en_us --num-samples 25
  python ingest.py url voxpopuli --config en --num-samples 25
  python ingest.py url some-org/some-dataset --config en --transcript-column text
  python ingest.py clear fleurs   # remove one dataset's clips
  python ingest.py clear          # drop the whole collection
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
    VectorParams,
)

from embeddings import CLAPEmbedder, Transcriber

COLLECTION = "audio_clips"
VECTOR_DIM = 512  # CLAP's output dimension

# Fixed vocabulary for zero-shot event tagging at ingest time. Extend freely --
# no retraining needed, CLAP does this zero-shot via the "sound of {class}" prompt.
EVENT_VOCAB = [
    "dog barking", "coughing", "laughter", "applause", "crying baby",
    "phone ringing", "typing on a keyboard", "door closing", "car horn",
    "background music", "silence",
]

# Short dataset names accepted by ingest_from_url. Each dataset names its
# transcript column and language configs differently, so those live here.
# Any other "org/name" Hugging Face repo ID can be passed directly as long as
# its column names are supplied.
HF_DATASETS = {
    "fleurs": {
        "repo_id": "google/fleurs",
        "default_config": "en_us",  # fr_fr, de_de, es_419, ...
        "audio_column": "audio",
        "transcript_column": "raw_transcription",
    },
    "voxpopuli": {
        "repo_id": "facebook/voxpopuli",
        "default_config": "en",  # de, fr, es, en_accented, ...
        "audio_column": "audio",
        "transcript_column": "raw_text",
    },
}


def get_or_create_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                # Two named vectors per point so audio-to-audio and text-to-audio
                # search can each hit the vector space they were designed for.
                "audio": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                "text": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            },
        )


def clear_dataset(dataset_name: str | None = None, qdrant_url: str = "http://localhost:6333") -> int:
    """
    Removes clips from the collection and returns how many were removed.

    With `dataset_name` (the short name or repo ID passed to ingest_from_url,
    stored as the `source` payload field), only that dataset's clips are
    deleted. Without it, the whole collection is dropped; the next ingest
    recreates it. Clips from ingest_directory have no `source`, so they can
    only be removed by dropping the whole collection.
    """
    client = QdrantClient(url=qdrant_url)
    if not client.collection_exists(COLLECTION):
        return 0

    if dataset_name is None:
        count = client.count(collection_name=COLLECTION, exact=True).count
        client.delete_collection(collection_name=COLLECTION)
        return count

    source_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=dataset_name))])
    count = client.count(collection_name=COLLECTION, count_filter=source_filter, exact=True).count
    if count:
        client.delete(collection_name=COLLECTION, points_selector=FilterSelector(filter=source_filter), wait=True)
    return count


def tag_clip(clap: CLAPEmbedder, audio_vector, event_text_vectors, threshold: float = 0.2):
    """Cosine-similarity zero-shot tagging against EVENT_VOCAB."""
    sims = event_text_vectors @ audio_vector
    return [EVENT_VOCAB[i] for i, s in enumerate(sims) if s >= threshold]


def ingest_directory(clips_dir: str, qdrant_url: str = "http://localhost:6333"):
    clap = CLAPEmbedder()
    transcriber = Transcriber(model_size="base")
    client = QdrantClient(url=qdrant_url)
    get_or_create_collection(client)

    # Precompute event-vocab text embeddings once, reuse for every clip.
    event_text_vectors = clap.embed_text([f"sound of {e}" for e in EVENT_VOCAB])

    audio_files = sorted(
        glob.glob(os.path.join(clips_dir, "*.wav")) +
        glob.glob(os.path.join(clips_dir, "*.mp3"))
    )
    print(f"Found {len(audio_files)} clips in {clips_dir}")

    points = []
    for path in audio_files:
        clip_id = str(uuid.uuid5(uuid.NAMESPACE_URL, path))
        transcript = transcriber.transcribe(path)

        audio_vec = clap.embed_audio([path])[0]
        text_vec = clap.embed_text([transcript])[0] if transcript else audio_vec

        tags = tag_clip(clap, audio_vec, event_text_vectors)

        points.append(
            PointStruct(
                id=clip_id,
                vector={"audio": audio_vec.tolist(), "text": text_vec.tolist()},
                payload={
                    "file_path": path,
                    "transcript": transcript,
                    "tags": tags,
                },
            )
        )
        print(f"  ingested {os.path.basename(path)}  tags={tags}")

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    print(f"Upserted {len(points)} clips into '{COLLECTION}'.")


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
        spec = {"repo_id": dataset_name, "default_config": None, "audio_column": "audio", "transcript_column": None}
    else:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Use one of {sorted(HF_DATASETS)} "
            "or a full Hugging Face repo ID like 'org/name'."
        )

    resolved = {
        "repo_id": spec["repo_id"],
        "config": config or spec["default_config"],
        "audio_column": audio_column or spec["audio_column"],
        "transcript_column": transcript_column or spec["transcript_column"],
    }
    if not resolved["transcript_column"]:
        raise ValueError(f"{dataset_name!r} isn't in HF_DATASETS, so transcript_column must be given.")
    return resolved


def ingest_from_url(
    dataset_name: str,
    config: str | None = None,
    split: str = "validation",
    num_samples: int = 25,
    qdrant_url: str = "http://localhost:6333",
    audio_column: str | None = None,
    transcript_column: str | None = None,
    cache_dir: str = "/tmp/hf_audio_clips",
):
    """
    Streams a speech dataset from Hugging Face (no full-dataset download),
    writes the first `num_samples` clips from `split` to local wav files, and
    reuses the same CLAP + tagging + upsert logic as ingest_directory.

    `dataset_name` is a short name from HF_DATASETS ("fleurs", "voxpopuli")
    or any "org/name" repo ID. `config` is the dataset's subset, which for
    these datasets is the language (en_us for FLEURS, en for VoxPopuli).

    Requires: pip install datasets soundfile. No Hugging Face login needed for
    the built-in datasets, though setting HF_TOKEN raises rate limits.

    The audio column is read undecoded (raw bytes) and decoded with soundfile,
    because datasets>=4 otherwise requires torchcodec to decode audio.

    The dataset's transcript is used as-is (no Whisper) since it's already a
    human-verified ground truth -- faster and more accurate than re-running ASR.
    """
    import io

    from datasets import Audio, load_dataset
    import soundfile as sf

    ds_info = resolve_hf_dataset(dataset_name, config, audio_column, transcript_column)
    repo_id, config = ds_info["repo_id"], ds_info["config"]
    audio_column, transcript_column = ds_info["audio_column"], ds_info["transcript_column"]

    os.makedirs(cache_dir, exist_ok=True)
    print(f"Streaming {repo_id} ({config}/{split}) from Hugging Face...")
    ds = load_dataset(repo_id, config, split=split, streaming=True)
    ds = ds.cast_column(audio_column, Audio(decode=False))

    # Includes dataset, config and split so clips from different sources never
    # share a wav path, and therefore never share a Qdrant point ID.
    file_prefix = f"{repo_id.replace('/', '__')}_{config}_{split}"

    clap = CLAPEmbedder()
    client = QdrantClient(url=qdrant_url)
    get_or_create_collection(client)
    event_text_vectors = clap.embed_text([f"sound of {e}" for e in EVENT_VOCAB])

    points = []
    for i, row in enumerate(ds):
        if i >= num_samples:
            break

        audio_array, sampling_rate = sf.read(io.BytesIO(row[audio_column]["bytes"]))
        transcript = (row[transcript_column] or "").strip()

        wav_path = os.path.join(cache_dir, f"{file_prefix}_{i:04d}.wav")
        sf.write(wav_path, audio_array, sampling_rate)

        clip_id = str(uuid.uuid5(uuid.NAMESPACE_URL, wav_path))
        audio_vec = clap.embed_audio([wav_path])[0]
        text_vec = clap.embed_text([transcript])[0] if transcript else audio_vec
        tags = tag_clip(clap, audio_vec, event_text_vectors)

        points.append(
            PointStruct(
                id=clip_id,
                vector={"audio": audio_vec.tolist(), "text": text_vec.tolist()},
                payload={
                    "file_path": wav_path,
                    "transcript": transcript,
                    "tags": tags,
                    "source": dataset_name,
                    "dataset": repo_id,
                    "config": config,
                    "split": split,
                },
            )
        )
        print(f"  [{i + 1}/{num_samples}] {transcript[:60]!r}  tags={tags}")

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
    print(f"Upserted {len(points)} {dataset_name} clips into '{COLLECTION}'.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest audio clips into the search index.")
    parser.add_argument(
        "source", choices=["dir", "url", "clear"], help="Where to ingest from, or 'clear' to remove clips."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=(
            "For 'dir': path to a clips directory. For 'url': a dataset short name "
            f"({', '.join(sorted(HF_DATASETS))}) or a Hugging Face repo ID like 'org/name'. "
            "For 'clear': the dataset to remove; omit it to drop the whole collection."
        ),
    )
    parser.add_argument("--config", default=None, help="url only: dataset config/language (defaults per dataset).")
    parser.add_argument("--split", default="validation", help="url only: dataset split.")
    parser.add_argument("--num-samples", type=int, default=25, help="url only: clips to pull.")
    parser.add_argument("--audio-column", default=None, help="url only: override the audio column name.")
    parser.add_argument("--transcript-column", default=None, help="url only: override the transcript column name.")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    if args.source == "clear":
        removed = clear_dataset(args.target, qdrant_url=args.qdrant_url)
        scope = f"'{args.target}' clips" if args.target else "clips (collection dropped)"
        print(f"Removed {removed} {scope} from '{COLLECTION}'.")
    elif not args.target:
        parser.error(f"source '{args.source}' requires a target")
    elif args.source == "dir":
        ingest_directory(args.target, qdrant_url=args.qdrant_url)
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
        )