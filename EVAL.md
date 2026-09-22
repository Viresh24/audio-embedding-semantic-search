# Event-search evaluation

CLAP `event_search` ranks clips by **audio**. Gold relevance is **text–text**: a
sentence transformer decides whether a prompt is about the same sound as a
clip's dataset-native `labels` (not `EVENT_LABELS`).

## Build the eval index

From `audio-embedding-semantic-search/`, with Qdrant running:

```bash
python -m ingestion.ingest url esc50 --num-samples 80 --collection audio_clips_eval
python -m ingestion.ingest url fleurs --num-samples 20 --collection audio_clips_eval
```

ESC-50 stores `labels` as the category with underscores turned into spaces
(`["keyboard typing"]`). FLEURS stores `labels=["speech"]` as a distractor.

Use `--collection audio_clips_eval` so the demo `audio_clips` index is untouched.
`event_search` is scored with `score_threshold=0` so a cosine cutoff does not
silently drop relevant clips.

## Run

```bash
  python -m eval.event_search --collection audio_clips_eval --top-k 1,5,10 --tau 0.45
```

Queries live in `eval/event_queries.json` (dataset labels plus paraphrases such
as "canine making noise"). Encoder: `sentence-transformers/all-MiniLM-L6-v2`.

JSON reports are written to `eval/results/`.

# Audio-to-audio search evaluation

`audio_to_audio_search` is scored with held-out clips as queries. Index one
split of a labelled dataset, then query with a different split: each held-out
clip is streamed from Hugging Face, written to a wav, and searched with. A
result is relevant if it shares a label with the query clip. Held-out clips are
never added to the index, and the eval refuses to run if the query split is
already in the collection.

```bash
python -m ingestion.ingest url audioset --split train --num-samples 2000 --collection audio_clips_eval
python -m eval.audio_search audioset --split test --num-samples 200 --collection audio_clips_eval
```

`audioset` is balanced AudioSet (`agkphysics/AudioSet`, config `balanced`),
whose clips carry lists of human-readable labels (`["Dog", "Bark", "Animal"]`)
stored as-is. ESC-50 on Hugging Face only has a `train` split, so it can't be
used here.

Queries whose labels appear nowhere in the index are skipped. Output: macro
MRR, AP@K, precision/recall@N, label accuracy by majority vote of the top
`--vote-k` hits, a random-ranking baseline, and a per-label breakdown with the
labels each one is most often confused with. JSON reports go to `eval/results/`.
