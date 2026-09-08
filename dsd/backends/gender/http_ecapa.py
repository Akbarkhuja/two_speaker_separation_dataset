"""Gender classification via the ECAPA-TDNN ONNX Docker service.

    docker pull akbarumirzokov/gender-detection:latest
    docker run --name gender-detection-api \
      -e BIND=0.0.0.0:8001 -p 8001:8001 \
      akbarumirzokov/gender-detection:latest

Note the port. The image defaults to 8000, which is also where the MOSS
diarization server listens (`sandbox/moss_serving/docker-compose.yml`); the two
cannot both have it. Changing the published port alone is not enough -- the
server binds 8000 inside the container regardless, so `BIND` has to move too.

`POST /predict` takes an `audio` file field and answers
`{"label": "female", "confidence": 0.94, "num_segments": 1, ...}`. The service
converts to mono and resamples to 8 kHz itself and soft-votes over 3 s windows
internally, so segments are sent at their native 8 kHz with no preprocessing.
"""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from ...registry import GENDER


class HttpEcapaGender:
    def __init__(self, options: dict):
        base = options.get("base_url", "http://localhost:8001").rstrip("/")
        self.url = options.get("url") or f"{base}/predict"
        self.timeout = float(options.get("timeout", 120.0))
        self.retries = int(options.get("retries", 3))
        self._session = None

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
        self._session = session
        return session

    def predict(self, samples: np.ndarray, sr: int) -> tuple[str, float]:
        # In-memory WAV: the notebook wrote /tmp/<uuid>.wav per segment and
        # never deleted them, which over a full corpus run leaves tens of
        # thousands of files behind.
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(samples, dtype=np.float32), sr, format="WAV", subtype="PCM_16")

        response = self._get_session().post(
            self.url,
            headers={"accept": "application/json"},
            files={"audio": ("segment.wav", buffer.getvalue(), "audio/wav")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["label"]).strip().lower(), float(payload.get("confidence", 1.0))


@GENDER.register("http_ecapa")
def _build(options: dict) -> HttpEcapaGender:
    return HttpEcapaGender(options)
