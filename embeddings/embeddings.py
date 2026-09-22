"""
Thin wrappers around the two models the pipeline needs:

- CLAPEmbedder: shared audio/text embedding space (search, sound-event scoring)
- Transcriber:  speech-to-text (Whisper) for the transcript column + lexical search
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ClipEmbedding:
    clip_id: str
    audio_vector: np.ndarray   # CLAP embedding of the raw waveform
    text_vector: np.ndarray    # CLAP embedding of the transcript
    transcript: str


class CLAPEmbedder:
    """
    Wraps laion_clap. Audio and text vectors live in the same space, so
    cosine similarity between them is meaningful in both directions:
    text->audio, audio->audio, and text->text.
    """

    def __init__(self, checkpoint: str = "laion/clap-htsat-fused", device: str = "cpu"):
        import laion_clap  # imported lazily so this module can be reused without the dep installed

        self.model = laion_clap.CLAP_Module(enable_fusion=True, device=device)
        self.model.load_ckpt(verbose=False)  # downloads/loads the pretrained checkpoint

    def embed_audio(self, file_paths: list[str]) -> np.ndarray:
        """Batch-embed one or more audio files. Returns (N, 512) L2-normalized vectors."""
        vecs = self.model.get_audio_embedding_from_filelist(x=file_paths, use_tensor=False)
        return self._l2_normalize(vecs)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Batch-embed one or more text strings. Returns (N, 512) L2-normalized vectors."""
        vecs = self.model.get_text_embedding(texts, use_tensor=False)
        return self._l2_normalize(vecs)

    @property
    def logit_scale(self) -> float:
        """CLAP's learned temperature for audio-text logits (cosine * logit_scale)."""
        return self.model.model.logit_scale_a.detach().exp().item()

    @staticmethod
    def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)


class Transcriber:
    """Wraps Whisper for speech-to-text. Swap out for a hosted ASR API if preferred."""

    def __init__(self, model_size: str = "base"):
        import whisper

        self.model = whisper.load_model(model_size)

    def transcribe(self, file_path: str) -> str:
        result = self.model.transcribe(file_path)
        return result["text"].strip()
