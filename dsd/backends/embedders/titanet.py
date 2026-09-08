"""NVIDIA TitaNet-Large speaker embeddings.

TitaNet-l is a **16 kHz** model, and `infer_segment` does NOT resample the way
`infer_file` does -- it hands the array straight to the preprocessor. Feeding it
8 kHz telephone audio makes the model hear every voice an octave low and twice
as slow, which destroys speaker separability: two different people scored 0.766
mean similarity that way, versus 0.629 once resampled. Nothing errors, the
embeddings just quietly stop meaning anything, so the resample has to happen
here, in the one place that knows the model's rate.

`infer_segment` only takes one segment at a time. This wraps `forward()`
directly to embed a padded batch, which is what makes the stage practical over
thousands of calls. `input_signal_length` masks the padding, so batching is
safe: measured against `infer_segment` over six segments of 1.5-18 s, cosine
agreement is 0.99999994 and the largest element difference 1e-4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ...core.audio import resample
from ...registry import EMBEDDERS

TITANET_SAMPLE_RATE = 16000
DEFAULT_MODEL_PATH = "/home/akbar/craft/prod/steno2/core/ml/model_weights/titanet-l.nemo"


class TitanetEmbedder:
    sample_rate = TITANET_SAMPLE_RATE
    dim = 192

    def __init__(self, options: dict):
        self.model_path = options.get("model_path", DEFAULT_MODEL_PATH)
        self.device = options.get("device", "auto")
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        import nemo.collections.asr as nemo_asr
        import torch

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"TitaNet weights not found: {self.model_path}")

        print(f"[embed] loading titanet-l on {device}: {self.model_path}")
        model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
            restore_path=self.model_path, map_location=device
        )
        model.to(device)
        model.eval()
        self._model = model
        return model

    def embed_batch(self, segments: Sequence[np.ndarray], sr: int) -> np.ndarray:
        import torch

        if not segments:
            return np.zeros((0, self.dim), dtype=np.float32)

        model = self._load()
        device = next(model.parameters()).device

        prepared = [
            resample(np.asarray(s, dtype=np.float32), sr, self.sample_rate) for s in segments
        ]
        lengths = torch.tensor([len(s) for s in prepared], device=device, dtype=torch.long)

        padded = np.zeros((len(prepared), int(lengths.max())), dtype=np.float32)
        for i, segment in enumerate(prepared):
            padded[i, : len(segment)] = segment

        with torch.no_grad():
            _logits, embeddings = model.forward(
                input_signal=torch.from_numpy(padded).to(device),
                input_signal_length=lengths,
            )

        embeddings = embeddings.float().cpu().numpy()
        # L2-normalize so the Gram matrix of a stack IS the cosine matrix.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)


@EMBEDDERS.register("titanet")
def _build(options: dict) -> TitanetEmbedder:
    return TitanetEmbedder(options)
