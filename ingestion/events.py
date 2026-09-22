"""
Zero-shot sound-event scoring with CLAP, shared by ingestion and search.

Each label is embedded as the average of many (alias x prompt template) text
embeddings, which is far less sensitive to phrasing than a single
"sound of {x}" prompt. Every clip is scored against event labels and
contrast labels (speech, room noise, ...), which give speech-heavy clips
somewhere to put their probability mass.

Per clip we keep three views of the same cosine similarities:
  - scores: raw cosine per label
  - probs:  softmax(logit_scale * cosines) over all labels, i.e. CLAP's own
            text-to-audio classification distribution
  - z:      (cosine - mean) / std per label, using stats from a calibration
            set, which removes each label's built-in bias

These scores are stored alongside each clip; they don't decide its labels,
which come straight from the source dataset (see ingestion/ingest.py).
"""

import hashlib
import json
import os
import warnings

import numpy as np

EVENT_LABELS = {
    "dog barking": ["dog barking", "Bark", "a dog barks", "dogs barking in the distance"],
    "coughing": ["coughing", "Cough", "a person coughs", "someone coughing repeatedly"],
    "laughter": ["laughter", "Laughter", "people laughing", "a person laughs out loud"],
    "applause": ["applause", "Applause", "an audience clapping", "people clapping their hands"],
    "crying baby": ["crying baby", "Baby cry, infant cry", "a baby crying", "an infant wailing"],
    "phone ringing": ["phone ringing", "Telephone bell ringing", "a telephone rings", "a cell phone ringtone"],
    "typing on a keyboard": [
        "typing on a keyboard", "Computer keyboard", "keyboard typing", "someone typing on a computer",
    ],
    "door closing": ["door closing", "Door", "a door slams shut", "a door being closed"],
    "car horn": ["car horn", "Vehicle horn, car horn, honking", "a car honks", "a honking horn in traffic"],
    "background music": ["background music", "Music", "music playing in the background", "instrumental music"],
    "silence": ["silence", "Silence", "a quiet room with no sound", "near silence"],
}

CONTRAST_LABELS = {
    "speech": ["speech", "Speech", "a person talking", "someone speaking"],
    "male speech": ["male speech", "Male speech, man speaking", "a man speaking"],
    "female speech": ["female speech", "Female speech, woman speaking", "a woman speaking"],
    "conversation": ["conversation", "Conversation", "people having a conversation"],
    "ambient room noise": ["ambient room noise", "room tone", "quiet background noise indoors"],
    "outdoor noise": ["outdoor noise", "Outside, urban or manmade", "street noise outdoors"],
}

PROMPT_TEMPLATES = [
    "{}",
    "the sound of {}",
    "a recording of {}",
    "{} can be heard",
    "an audio clip of {}",
    "this is the sound of {}",
    "a sound recording of {}",
]

DEFAULT_CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "event_calibration.json")

_SCORING_METHOD = "alias-template-mean/softmax-logit-scale/z-score"
EVENT_VERSION = hashlib.sha1(
    json.dumps(
        {"events": EVENT_LABELS, "contrast": CONTRAST_LABELS, "templates": PROMPT_TEMPLATES, "method": _SCORING_METHOD},
        sort_keys=True,
    ).encode()
).hexdigest()[:10]

_EMBED_BATCH = 128


def _cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # numpy's Apple Accelerate BLAS raises spurious divide/overflow/invalid
    # flags on matmuls of ordinary unit vectors; the results are correct.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return a @ b


def _ensemble_vectors(clap, phrase_lists: list[list[str]]) -> np.ndarray:
    """One L2-normalized vector per phrase list: mean over every phrase x template."""
    prompts, owners = [], []
    for i, phrases in enumerate(phrase_lists):
        for phrase in phrases:
            for template in PROMPT_TEMPLATES:
                prompts.append(template.format(phrase))
                owners.append(i)

    vecs = np.concatenate(
        [clap.embed_text(prompts[i : i + _EMBED_BATCH]) for i in range(0, len(prompts), _EMBED_BATCH)]
    )
    owners = np.array(owners)
    out = np.stack([vecs[owners == i].mean(axis=0) for i in range(len(phrase_lists))])
    return out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9, None)


def build_label_vectors(clap) -> tuple[list[str], np.ndarray]:
    """Returns (names, matrix): event labels first, then contrast labels; matrix is (L, 512)."""
    all_labels = {**EVENT_LABELS, **CONTRAST_LABELS}
    names = list(all_labels)
    return names, _ensemble_vectors(clap, [all_labels[n] for n in names])


def embed_event_query(clap, query: str) -> np.ndarray:
    """Query vector for event search, built the same way as the label vectors."""
    phrases = EVENT_LABELS.get(query) or CONTRAST_LABELS.get(query) or [query]
    return _ensemble_vectors(clap, [phrases])[0]


def score_clip(
    audio_vec: np.ndarray,
    names: list[str],
    label_matrix: np.ndarray,
    logit_scale: float,
    calib: dict | None = None,
) -> dict:
    """Returns {"scores", "probs", "z"}; "z" is None without calibration stats."""
    cos = _cosines(label_matrix, np.asarray(audio_vec))
    logits = logit_scale * cos
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    scores = dict(zip(names, cos.tolist()))
    prob_map = dict(zip(names, probs.tolist()))

    z = None
    if calib is not None:
        stats = calib["labels"]
        z = {n: (scores[n] - stats[n]["mean"]) / max(stats[n]["std"], 1e-6) for n in names if n in stats}

    return {"scores": scores, "probs": prob_map, "z": z}


def event_payload(result: dict) -> dict:
    """Qdrant payload fields for a score_clip result."""
    def rnd(d):
        return {k: round(v, 4) for k, v in d.items()}

    return {
        "event_scores": rnd(result["scores"]),
        "event_probs": rnd(result["probs"]),
        "event_z": rnd(result["z"]) if result["z"] is not None else None,
        "event_version": EVENT_VERSION,
    }


def compute_calibration(names: list[str], label_matrix: np.ndarray, audio_vecs: np.ndarray, sources: dict) -> dict:
    """Per-label mean/std of cosine scores over a reference set of clip audio vectors."""
    cos = _cosines(np.asarray(audio_vecs), label_matrix.T)
    return {
        "version": EVENT_VERSION,
        "n_clips": int(cos.shape[0]),
        "sources": sources,
        "labels": {
            n: {"mean": float(cos[:, i].mean()), "std": float(cos[:, i].std()), "n": int(cos.shape[0])}
            for i, n in enumerate(names)
        },
    }


def save_calibration(stats: dict, path: str = DEFAULT_CALIBRATION_PATH):
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_calibration(path: str = DEFAULT_CALIBRATION_PATH) -> dict | None:
    """Returns calibration stats, or None if missing or built for a different EVENT_VERSION."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        stats = json.load(f)
    if stats.get("version") != EVENT_VERSION:
        warnings.warn(
            f"Ignoring {path}: built for event version {stats.get('version')!r}, current is {EVENT_VERSION!r}. "
            "Re-run `python -m ingestion.ingest calibrate`."
        )
        return None
    return stats
