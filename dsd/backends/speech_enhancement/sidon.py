"""Speech restoration via the Sidon HTTP service, returned at a wider rate than it was sent.

Served by the sibling project `/home/akbar/craft/prod/sidon_serving`:

    cd /home/akbar/craft/prod && SIDON_DEVICES=cuda:0 \\
        sidon_serving/venv/bin/python -m uvicorn sidon_serving.server.app:app --port 8099

Sidon (`sarulab-speech/sidon-v0.1`) is a **generative** restorer, not a masking
denoiser: it predicts w2v-BERT features from the degraded input and resynthesises
the waveform with a vocoder whose decoder upsamples 8*5*4*3*2 = 960x at 50 frames
per second -- so it always synthesises at **48 kHz**, whatever went in. Given 8 kHz
telephone speech it reconstructs full-band speech; the band above 4 kHz is
generated, not interpolated.

That is why `output_sample_rate` is **always sent**, and why it is 24 kHz here:

  - Left out, the service falls back to its `SIDON_OUTPUT_RATE` policy, whose
    default is the *input* rate. An 8 kHz request then comes back at 8 kHz and
    everything Sidon restored above 4 kHz -- the expensive half of what it does --
    is computed and discarded. The service's own README says as much. Sending it
    explicitly also means an operator changing that env var cannot silently
    change the dataset.
  - 24 kHz is the rate DialogueSidon's decoder emits (480x at 50 frames/s), and
    that decoder is frozen during fine-tuning. Its pretraining targets were Sidon
    output downsampled 48 -> 24 kHz, from 8 kHz Fisher/CALLHOME input: narrowband
    in, wideband out. 8 kHz targets would teach it that the right answer above
    4 kHz is silence.

The service path per request is: peak-normalise, encode at 16 kHz, synthesise at
48 kHz, resample to `output_sample_rate` (soxr_hq), pin the length to
`round(n * out / in)` (`SIDON_EXACT_LENGTH`, on by default), then restore the
input's level (`level_match`, sent explicitly for the same reason as the rate).

Both directions use `pcm_f32le`, as the MossFormerGAN client does: no container
to parse and no requantisation in transit.

Note the endpoint is `/v1/restore`, not `/v1/enhance`. Until this backend existed
the pipeline's only client spoke `/v1/enhance`, which the Sidon service does not
serve -- so whatever produced `work/enhanced/` before was not Sidon.
"""

from __future__ import annotations

import base64

import numpy as np

from ...registry import ENHANCERS
from .base import expected_length

# What `/healthz` must say, so a different model behind the port is refused
# rather than cached under Sidon's name.
EXPECTED_MODEL = "Sidon"


class HttpSidon:
    expected_model = EXPECTED_MODEL
    # Measured on the deployed service: 0.06 h of 8 kHz speech spans in 62
    # requests at ~25x realtime. Only feeds the stage's up-front estimate.
    realtime_factor = 25.0

    def __init__(self, options: dict):
        base = options.get("base_url", "http://localhost:8099").rstrip("/")
        self.url = options.get("url") or f"{base}/v1/restore"
        self.health_url = f"{base}/healthz"
        self.timeout = float(options.get("timeout", 900.0))
        self.retries = int(options.get("retries", 3))
        # None keeps the input rate -- legal, but it throws Sidon's band away,
        # which is the whole reason this backend exists. The stage passes
        # `enhance.output_sample_rate` in here.
        rate = options.get("output_sample_rate")
        self.output_sample_rate = int(rate) if rate else None
        self.level_match = str(options.get("level_match", "peak"))
        self._session = None
        self.audio_seconds = 0.0
        self.processing_seconds = 0.0
        self.requests = 0
        # Requests where the service had to trim or pad to the pinned length.
        # A frame of drift is normal for a resynthesiser; reported, not hidden.
        self.length_adjusted = 0

    def _get_session(self):
        if self._session is not None:
            return self._session

        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=8,
            pool_maxsize=8,
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

    def health(self) -> dict:
        response = self._get_session().get(self.health_url, timeout=30)
        response.raise_for_status()
        return response.json()

    def output_rate(self, sr: int) -> int:
        return self.output_sample_rate or int(sr)

    # ------------------------------------------------------------------ #
    def enhance(self, samples: np.ndarray, sr: int) -> np.ndarray:
        span = np.ascontiguousarray(samples, dtype=np.float32)
        out_sr = self.output_rate(sr)
        if span.size == 0:
            return span

        response = self._get_session().post(
            self.url,
            json={
                "audio": base64.b64encode(span.astype("<f4").tobytes()).decode(),
                "encoding": "pcm_f32le",
                "sample_rate": int(sr),
                "output_sample_rate": int(out_sr),
                "level_match": self.level_match,
                "response_format": "pcm_f32le",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        restored = np.frombuffer(base64.b64decode(payload["audio"]), dtype="<f4")

        returned_rate = int(payload.get("sample_rate", -1))
        if returned_rate != out_sr:
            raise ValueError(
                f"restorer returned {returned_rate} Hz where {out_sr} Hz was requested"
            )
        heard_rate = int(payload.get("input_sample_rate", sr))
        if heard_rate != sr:
            raise ValueError(
                f"restorer read the span at {heard_rate} Hz, but it was sent at {sr} Hz"
            )
        model_rate = int(payload.get("model_sample_rate", 0))
        if model_rate and model_rate < out_sr:
            # Anything above the model's own Nyquist would be interpolation.
            raise ValueError(
                f"restorer synthesises at {model_rate} Hz; {out_sr} Hz output would carry "
                "an invented band"
            )
        want = expected_length(len(span), sr, out_sr)
        if len(restored) != want:
            # Refuse rather than pad or trim: the splice places this at an exact
            # time offset, and a quiet length change shifts every label after it.
            raise ValueError(
                f"restorer returned {len(restored)} samples @ {out_sr} Hz for {len(span)} "
                f"@ {sr} Hz; the splice requires exactly {want} (same duration). Is the "
                "service running with SIDON_EXACT_LENGTH=0?"
            )

        self.audio_seconds += len(span) / sr
        self.processing_seconds += float(payload.get("processing_ms", 0.0)) / 1000.0
        self.requests += 1
        self.length_adjusted += int(bool(payload.get("length_adjusted", False)))
        return np.ascontiguousarray(restored, dtype=np.float32)


@ENHANCERS.register("http_sidon")
def _build(options: dict) -> HttpSidon:
    return HttpSidon(options)
