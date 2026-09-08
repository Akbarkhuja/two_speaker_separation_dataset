"""MOSS-Transcribe-Diarize over an OpenAI-compatible transcription endpoint.

The model is a joint ASR + diarization model that answers with

    [start][Sxx]transcribed text[end][start][Sxx]...[end]

Only (start, end, speaker) is kept; no transcript text ever reaches disk.

Assumes a server is already up, e.g. from `sandbox/moss_serving/`:

    docker compose up -d moss-server            # vLLM,  port 8000
    docker compose --profile sglang up -d ...   # SGLang Omni

Note it is a 0.9 B model needing ~6 GB of VRAM, so on a single 8 GB card it
cannot share the GPU with the Sortformer or TitaNet stages. It is also far
slower than Sortformer per file; prefer `sortformer` unless you specifically
want MOSS's segmentation.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from ...core.rttm import Segment
from ...registry import DIARIZERS
from .base import per_channel

# [12.26][S02]some text[13.81] -- the body may span newlines.
SEGMENT_RE = re.compile(
    r"\[\s*(\d+(?:\.\d+)?)\s*\]\s*\[\s*(S\d+)\s*\]\s*(.*?)\s*\[\s*(\d+(?:\.\d+)?)\s*\]",
    re.DOTALL,
)

# The speaker tag as it appears at the head of a verbose_json segment's text.
SPEAKER_TAG_RE = re.compile(r"\[\s*(S\d+)\s*\]")


class MossHttpDiarizer:
    needs_audio = True

    def __init__(self, options: dict):
        host = options.get("host", "127.0.0.1")
        port = int(options.get("port", 8000))
        self.url = options.get("url") or f"http://{host}:{port}/v1/audio/transcriptions"
        self.model = options.get("model", "OpenMOSS-Team/MOSS-Transcribe-Diarize")
        self.max_new_tokens = int(options.get("max_new_tokens", 16384))
        self.timeout = float(options.get("timeout", 1800.0))
        self.retries = int(options.get("retries", 3))
        self._session = None

    def _get_session(self):
        """One pooled session for the whole run; per-call sessions thrash TCP."""
        if self._session is not None:
            return self._session

        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=16,
            pool_maxsize=16,
            max_retries=Retry(
                total=self.retries,
                backoff_factor=1.0,
                status_forcelist=[500, 502, 503, 504],
            ),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        self._session = session
        return session

    def _transcribe(self, samples: np.ndarray, sr: int) -> dict:
        buffer = io.BytesIO()
        sf.write(buffer, samples, sr, format="WAV", subtype="PCM_16")
        response = self._get_session().post(
            self.url,
            data={
                "model": self.model,
                "response_format": "verbose_json", 
                # "verbose_json" options make response option more structured
                # "segments" key added
                "temperature": "0",
                # Truncation is silent, so this is set generously; a long call
                # that hits the cap loses its tail with no error.
                "max_new_tokens": str(self.max_new_tokens),
            },
            files={"file": ("chunk.wav", buffer.getvalue(), "audio/wav")},
            timeout=(10, self.timeout),
        )
        response.raise_for_status()
        # The whole payload rather than one key. `verbose_json` adds server-parsed
        # `segments`, but not every build returns them, and `_parse` needs `text`
        # to fall back on when they are missing.
        return response.json()

    def diarize(
        self,
        call: str,
        audio_path: Path,
        data: np.ndarray | None,
        sr: int | None,
    ) -> list[Segment]:
        def diarize_mono(channels: list[np.ndarray], sample_rate: int):
            return [_parse(self._transcribe(ch, sample_rate)) for ch in channels]

        return per_channel(data, sr, diarize_mono)


def _parse(payload: dict) -> list[Segment]:
    """Prefer the server's parsed segments; fall back to the raw transcript.

    Mirrors `sandbox/moss_serving/diarize_batch.py:segments_from_response`, and
    for the same reason: the documented `verbose_json` segment is
    `{"id", "start", "end", "text"}` with the speaker carried *inside* the text
    as `[S01]`, while some builds add a `speaker` key instead. Reading
    `seg["speaker"]` unconditionally raises KeyError on the documented shape, and
    a server answering plain `json` returns no `segments` at all.
    """
    raw = payload.get("segments")
    segments: list[Segment] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            start, end = item.get("start"), item.get("end")
            if start is None or end is None:
                continue
            speaker = (
                item.get("speaker") or item.get("speaker_id") or item.get("speaker_label")
            )
            if not speaker:
                tag = SPEAKER_TAG_RE.search(item.get("text") or "")
                # "S01" only as a last resort: for per-channel diarization the
                # channel is the identity anyway, so one label per channel is
                # still correct -- it just cannot report a second speaker.
                speaker = tag.group(1) if tag else "S01"
            segments.append(Segment(start=float(start), end=float(end), speaker=str(speaker)))

    if not segments:
        segments = [
            Segment(start=float(start), end=float(end), speaker=speaker)
            for start, speaker, _body, end in (
                m.groups() for m in SEGMENT_RE.finditer(payload.get("text") or "")
            )
        ]

    segments.sort(key=lambda s: s.start)
    return segments


@DIARIZERS.register("moss_http")
def _build(options: dict) -> MossHttpDiarizer:
    return MossHttpDiarizer(options)
