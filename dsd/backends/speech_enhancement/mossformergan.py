"""Speech enhancement via the MossFormerGAN_SE_16K HTTP service.

Served by the sibling project `/home/akbar/craft/prod/mossformergan_serve`:

    cd /home/akbar/craft/prod/mossformergan_serve && docker compose up -d

`POST /v1/enhance` takes base64 audio and answers with base64 audio plus
timing. Both directions use `pcm_f32le` rather than WAV: there is no container
header to parse, and nothing is requantised to 16-bit in transit -- these spans
are about to be summed and rescaled, so an extra rounding on the way through
buys nothing.

Two details of the service contract matter here:

  - `sample_rate` is **required**. Raw PCM carries no header, so the server
    cannot infer it and will reject the request without it.
  - `output_sample_rate` is deliberately **omitted**, because it defaults to the
    input rate. The model runs at 16 kHz internally and resamples in both
    directions itself; asking for 8 kHz back keeps the corpus at its native
    telephone rate instead of carrying a 4-8 kHz band that was never recorded.
    Measured on one 30 s call: 240000 samples in, 240000 out, and the round trip
    through `torchaudio.functional.resample` is exact at this ratio.

Throughput is ~3.2x realtime and does **not** improve with client concurrency --
measured 3.17x / 3.28x / 3.27x at 1 / 4 / 8 workers, because `/healthz` reports
one replica with the adaptive batch cap at 1. Extra workers only queue.

The default port is 8000, which is where the service actually runs -- note that
is also `moss_http`'s default, so the enhancement server and the MOSS diarizer
cannot both be up.
"""

from __future__ import annotations

import base64

import numpy as np

from ...registry import ENHANCERS


class HttpMossFormerGAN:
    def __init__(self, options: dict):
        base = options.get("base_url", "http://localhost:8000").rstrip("/")
        self.url = options.get("url") or f"{base}/v1/enhance"
        self.health_url = f"{base}/healthz"
        self.timeout = float(options.get("timeout", 600.0))
        self.retries = int(options.get("retries", 3))
        self._session = None
        # Aggregated so the stage can report what the service actually did
        # rather than what it was asked for.
        self.audio_seconds = 0.0
        self.processing_seconds = 0.0
        self.requests = 0

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
        """Ask the service what it is; used to fail early with a useful message."""
        response = self._get_session().get(self.health_url, timeout=30)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------ #
    def enhance(self, samples: np.ndarray, sr: int) -> np.ndarray:
        span = np.ascontiguousarray(samples, dtype=np.float32)
        if span.size == 0:
            return span

        response = self._get_session().post(
            self.url,
            json={
                "audio": base64.b64encode(span.astype("<f4").tobytes()).decode(),
                "encoding": "pcm_f32le",
                "sample_rate": int(sr),
                # `output_sample_rate` omitted on purpose -- see the module docstring.
                "response_format": "pcm_f32le",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        enhanced = np.frombuffer(base64.b64decode(payload["audio"]), dtype="<f4")

        returned_rate = int(payload.get("sample_rate", sr))
        if returned_rate != sr:
            raise ValueError(
                f"enhancer returned {returned_rate} Hz for a {sr} Hz request; the splice "
                "offsets assume the rate is unchanged"
            )
        if len(enhanced) != len(span):
            # Refuse rather than pad or trim. Every downstream label is a sample
            # offset into this channel, so absorbing a length change here would
            # shift the whole call by a silent amount.
            raise ValueError(
                f"enhancer returned {len(enhanced)} samples for {len(span)}; the splice "
                "requires an exact length match"
            )

        self.audio_seconds += len(span) / sr
        self.processing_seconds += float(payload.get("processing_ms", 0.0)) / 1000.0
        self.requests += 1
        return np.ascontiguousarray(enhanced, dtype=np.float32)


@ENHANCERS.register("http_mossformergan")
def _build(options: dict) -> HttpMossFormerGAN:
    return HttpMossFormerGAN(options)
