"""`http_mossformergan`: a historical name, kept as an alias of the Sidon client.

The backend name predates the model behind it. The service this pipeline talks
to is Sidon (`sarulab-speech/sidon-v0.1`) at `POST /v1/restore`, a **generative**
restorer that synthesises at 48 kHz -- not MossFormerGAN, a masking denoiser that
cannot create band. The name is kept so existing configs and command lines keep
working; what it builds is exactly `sidon.HttpSidon`, including the explicit
`output_sample_rate`. See `sidon.py` for why that field is always sent and why
it is 24 kHz.

A record of what this name used to mean, because the caches it left behind are
still on disk. The old client posted to `/v1/enhance` -- the contract of the
MossFormerGAN and RE-USE servers, which Sidon does not serve -- and omitted
`output_sample_rate`, so its output came back at 8 kHz. Inside speech, those
caches (`work/enhanced/`, `work/enhanced_mossformergan/`) correlate with the
original recording at 0.997-0.999 sample for sample: masking-model output. They
are 8 kHz and carry no marker, so the enhance stage refuses to extend them and
`build` refuses to mix them with 24 kHz output.
"""

from __future__ import annotations

from ...registry import ENHANCERS
from .sidon import HttpSidon

# Importable under the old name for anything that still does.
HttpMossFormerGAN = HttpSidon


@ENHANCERS.register("http_mossformergan")
def _build(options: dict) -> HttpSidon:
    return HttpSidon(options)
