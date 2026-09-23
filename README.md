# Audio Embedding Semantic Search

Semantic search over a library of audio clips, built on CLAP (a shared
audio/text embedding space) and stored in Qdrant. Clips are streamed from
Hugging Face datasets, embedded, scored against a zero-shot sound-event
vocabulary, and indexed so they can be searched by example clip or by a
free-text sound description ("dog barking"). Two eval harnesses measure
retrieval quality on labelled datasets.

## Layout

- `embeddings/embeddings.py`: `CLAPEmbedder` (audio and text embeddings,
  `laion/clap-htsat-fused`, 512-dim, L2-normalized) and `Transcriber`
  (Whisper ASR, not used by the current ingestion path).
- `ingestion/ingest.py`: streams a Hugging Face dataset, writes each clip to a
  wav, embeds it, scores it against the event vocabulary, and upserts it into
  Qdrant. Also clears, calibrates, and rescores stored clips.
- `ingestion/events.py`: the zero-shot event vocabulary (`EVENT_LABELS`,
  `CONTRAST_LABELS`, `PROMPT_TEMPLATES`) and the scoring and calibration logic
  shared by ingestion and search.
- `ingestion/event_calibration.json`: per-label score statistics written by
  `calibrate` and loaded automatically at ingest and rescore time.
- `search/search.py`: `AudioSearchEngine` with `list_clips()`,
  `audio_to_audio_search(path)`, and `event_search("dog barking")`.
- `eval/event_search.py`, `eval/audio_search.py`: evaluation harnesses for the
  two search modes. `eval/metrics.py` holds the shared ranking metrics,
  `eval/labels.py` the sentence-transformer relevance judge, and
  `eval/event_queries.json` the event-search prompts.
- `eval/results/`: timestamped JSON reports from each eval run.

## Setup

Requires Python 3.10+ and Docker (or a hosted Qdrant instance).

```bash
cd audio-embedding-semantic-search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Qdrant on localhost:6333; every command accepts --qdrant-url to point elsewhere
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

The first run downloads the CLAP checkpoint (and, for evals, the
`all-MiniLM-L6-v2` sentence transformer). Built-in datasets need no Hugging
Face login; setting `HF_TOKEN` raises streaming rate limits.

**Run every command from the `audio-embedding-semantic-search/` directory**
with `python -m ...`, so the `embeddings`, `ingestion`, `search`, and `eval`
packages resolve.

## Ingestion

```bash
python -m ingestion.ingest url <dataset> [--config C] [--split S] [--num-samples N] [--collection NAME]
```

Clips are streamed (no full-dataset download), written to
`/tmp/hf_audio_clips/`, and upserted in batches of 500. `--num-samples 0`
ingests the whole split. The default collection is `audio_clips`. Samples that
fail to decode, embed, or upsert are skipped, and the run ends with an
ingested/skipped count.

Built-in dataset short names (`HF_DATASETS` in `ingestion/ingest.py`):

| Name | Hugging Face repo | Default config / split | Stored `labels` |
|---|---|---|---|
| `fleurs` | `google/fleurs` | `en_us` / `validation` | `["speech"]` |
| `voxpopuli` | `facebook/voxpopuli` | `en` / `validation` | `["speech"]` |
| `esc50` | `ashraq/esc50` | none / `train` | ESC-50 category, e.g. `["keyboard typing"]` |
| `audioset` | `agkphysics/AudioSet` | `balanced` / `train` | AudioSet human labels, e.g. `["Dog", "Bark", "Animal"]` |

Examples:

```bash
python -m ingestion.ingest url fleurs --config fr_fr --num-samples 25
python -m ingestion.ingest url voxpopuli --config en --num-samples 25
python -m ingestion.ingest url esc50 --num-samples 100
python -m ingestion.ingest url audioset --split train --num-samples 2000 --collection audio_clips_eval
```

Any other Hugging Face repo can be passed as `org/name`, but then
`--transcript-column` is required (and `--audio-column` if it isn't `audio`).

Each Qdrant point holds two named vectors, `audio` (CLAP embedding of the
waveform) and `text` (CLAP embedding of the transcript, or a copy of `audio`
when the dataset has none), plus a payload with `file_path`, `transcript`,
`labels`, `source`, `dataset`, `config`, `split`, and the event scores
`event_scores`, `event_probs`, `event_z`, and `event_version`. Point IDs are
derived from the wav path, so re-ingesting the same rows overwrites them
instead of duplicating them.

FLEURS and VoxPopuli are read speech only, so `event_search` has nothing to
find in them. Mix in ESC-50 or AudioSet clips (same `--collection`) to exercise
event search.

### Clearing, calibrating, and rescoring

```bash
python -m ingestion.ingest clear fleurs      # delete one source's clips
python -m ingestion.ingest clear             # drop the whole collection
python -m ingestion.ingest calibrate         # per-label event-score mean/std from stored clips
python -m ingestion.ingest calibrate esc50   # restrict the reference set to one source
python -m ingestion.ingest rescore           # recompute event_* fields from stored audio vectors
```

All of these accept `--collection`. `calibrate` reads the stored audio vectors
(no re-embedding), computes the mean and std of each label's cosine score, and
writes `ingestion/event_calibration.json`. It warns below ~200 clips or when
every clip comes from one source, because the z-scores are then only relative
to that set. Existing clips aren't updated by `calibrate`; run `rescore`
afterwards to fill in `event_z`. New ingests pick up the calibration file
automatically.

The calibration file is tied to `EVENT_VERSION`, a hash of the labels,
phrases, templates, and scoring method. After editing `EVENT_LABELS`,
`CONTRAST_LABELS`, or `PROMPT_TEMPLATES`, the old file is ignored with a
warning; re-run `calibrate` and then `rescore`.

## Search

```bash
python -m search.search event_search "dog barking" --top-k 5
python -m search.search audio_to_audio_search /path/to/clip.wav --top-k 5 --collection audio_clips_eval
```

Each result prints its cosine score, wav path, and dataset labels.

From your own code (run from the project root):

```python
from search.search import AudioSearchEngine

engine = AudioSearchEngine(collection="audio_clips")
clips = engine.list_clips()
hits = engine.audio_to_audio_search("query_clip.wav", top_k=10)
hits = engine.event_search("coughing", top_k=10, threshold=0.15)
engine.close()
```

- `audio_to_audio_search` embeds the query clip with CLAP and searches the
  `audio` vectors.
- `event_search` builds the query vector the same way ingest builds label
  vectors: if the query is a known label in `ingestion/events.py`, all of its
  aliases times every prompt template are averaged; otherwise the free-text
  query is averaged over the templates. It then searches the `audio` vectors.
  The default `threshold=0.15` drops low-cosine hits; pass `threshold=0` to
  get a full ranking.
- `list_clips` returns up to 10,000 clips in a single scroll.

## Evaluation

Both evals default to the `audio_clips_eval` collection so the demo
`audio_clips` index is left alone. Each run prints a summary table and writes
a JSON report to `eval/results/<eval>_<UTC timestamp>.json`. See `EVAL.md` for
more on methodology.

### Event search

`event_search` ranks clips by audio, but relevance is decided in text space: a
sentence transformer (`all-MiniLM-L6-v2`) marks a clip relevant to a prompt
if the prompt's similarity to any of the clip's dataset labels is at least
`--tau`. Queries come from `eval/event_queries.json` and include both label
names and paraphrases ("canine making noise").

```bash
python -m ingestion.ingest url esc50 --num-samples 80 --collection audio_clips_eval
python -m ingestion.ingest url fleurs --num-samples 20 --collection audio_clips_eval
python -m eval.event_search --collection audio_clips_eval --top-k 1,5,10 --tau 0.45
```

| Flag | Default | Meaning |
|---|---|---|
| `--top-k` | `1,5,10` | Cutoffs for recall, precision, hit, and nDCG |
| `--tau` | `0.45` | Minimum prompt-to-label similarity for a label to count as relevant |
| `--top-m-labels` | none | Use the M nearest labels instead of `--tau` |
| `--threshold` | `0.0` | `event_search` score cutoff (0 means no cutoff) |
| `--encoder` | `sentence-transformers/all-MiniLM-L6-v2` | Relevance judge |
| `--queries` | `eval/event_queries.json` | JSON list of prompt strings |

The report gives macro MRR and recall, precision, and nDCG at each cutoff,
alongside two reference points: `random_mrr` (a shuffled ranking) and
`label_index_mrr` (a text-only upper bound that ranks relevant clips by label
similarity). Per query it also shows the matched labels and the labels of the
first false positive. Queries with no matching label are listed as skipped.

### Audio-to-audio search

Held-out clips are used as queries. Index one split of a labelled dataset,
then query with a different split. Each query clip is streamed, written to a
wav, and searched with, and is never added to the index. A hit is relevant if
it shares a label with the query. The eval refuses to run if the query split
is already in the collection.

```bash
python -m ingestion.ingest url audioset --split train --num-samples 2000 --collection audio_clips_eval
python -m eval.audio_search audioset --split test --num-samples 200 --collection audio_clips_eval
```

| Flag | Default | Meaning |
|---|---|---|
| `--split` | `test` | Held-out split to query with |
| `--num-samples` | `200` | Query clips to stream (0 means all) |
| `--top-k` | `1,5,10` | Ranking cutoffs; the largest is also used for AP |
| `--vote-k` | `5` | Top hits that vote on the predicted label |
| `--config`, `--audio-column`, `--transcript-column` | per dataset | Same overrides as ingestion |

The report gives macro MRR, AP@K, precision and recall at each cutoff, label
accuracy by majority vote of the top `--vote-k` hits, and a
`random_precision` baseline. It also includes a per-label breakdown (worst
first) with the labels each one is most often confused with. Queries whose
labels appear nowhere in the index are skipped.

Use AudioSet for this eval. ESC-50 on Hugging Face only has a `train` split,
so there is no held-out split to query with.

## Design notes and where to extend

- **Two named vectors per clip.** `audio` powers search-by-example and event
  scoring. `text` holds the CLAP embedding of the transcript for text-to-text
  matching in the same space. No search mode queries it yet.
- **Event scores are stored, not used for ranking.** Every clip gets three
  views of its cosine to each event and contrast label: raw `event_scores`,
  `event_probs` (softmax with CLAP's `logit_scale`, where contrast labels like
  `speech` absorb mass on speech clips), and calibrated `event_z`. They're
  available for filtering or tagging, but `event_search` ranks by vector
  similarity alone. A clip's `labels` always come from the source dataset.
- **Open event vocabulary.** Add entries (with aliases) to `EVENT_LABELS` in
  `ingestion/events.py` at any time, then run `calibrate` and `rescore`. No
  audio is re-embedded.
- **Long clips.** CLAP works on short (~10 s) windows. For long audio, chunk
  it, embed each chunk, and pool the chunk vectors before upserting.
- **Swap points.** `QdrantClient` can be replaced with another vector store,
  and the CLAP checkpoint (fused vs. unfused) trades speed for quality.
  `Transcriber` is available for datasets without ground-truth transcripts,
  but it isn't wired into ingestion yet.
