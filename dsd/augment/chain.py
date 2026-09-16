"""The degradation chain: draw a recipe, then apply it.

DialogueSidon's Appendix A applies six degradations to each track independently,
each with probability 0.5, in a fixed order:

    reverberation -> background noise -> band limitation -> clipping
                  -> codec -> packet loss

The order is not arbitrary and is kept exactly. Reverb is a property of the room
and happens before anything electrical; the noise is picked up by the microphone,
so it is added after the room but before the line; band limitation and the codec
are the line itself; packet loss happens to the encoded stream, so it comes last.
Clipping sits between the microphone and the codec, where an overdriven preamp
would be.

The paper's seventh step, `x = w*y1 + (1-w)*y2` with `w ~ U(0.3, 0.7)`, is not
here. It is the same operation as `build`'s existing `scale_to_sir`: that `w`
range is an SIR of +-7.36 dB. Doing it again would randomise the level twice.
Set `build.sir_db: [-7.36, 7.36]` to reproduce the paper's distribution exactly.

Splitting sampling from application is what makes a variant reproducible and
inspectable: `sample()` returns a `Recipe` of plain JSON that goes into
`meta.json`, and `apply()` is a pure function of the signal and that recipe, so
re-running it produces the same samples without re-drawing anything.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import numpy as np

from . import artifacts, codec, noise as noise_mod, reverb

# The order above, as the code applies it.
STEPS = ("reverb", "noise", "band_limit", "clip", "codec", "packet_loss")


@dataclass
class Recipe:
    """Every draw that shapes one track of one variant. `None` means "not applied"."""

    reverb: dict | None = None
    noise: dict | None = None
    band_limit: dict | None = None
    clip: dict | None = None
    codec: dict | None = None
    packet_loss: dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "Recipe":
        return cls(**{k: data.get(k) for k in STEPS})

    def any_applied(self) -> bool:
        return any(getattr(self, step) is not None for step in STEPS)


@dataclass
class Banks:
    """The asset pools `apply` draws from, built once per run by `build`."""

    rirs: reverb.RIRBank | None = None
    noise: noise_mod.NoiseBank | None = None
    codecs: list[str] = field(default_factory=list)


def sample(rng: random.Random, acfg, banks: Banks) -> Recipe:
    """Draw one track's degradations. Each fires independently with `acfg.prob`.

    The concrete impulse response and noise clip are chosen *here*, not in
    `apply`, so the recipe names them: a variant in the dataset can be traced
    back to the exact room and the exact noise recording that made it.
    """
    recipe = Recipe()

    if _fires(rng, acfg.prob) and banks.rirs is not None and banks.rirs.available():
        drawn = banks.rirs.draw(rng, acfg.reverb.simulated_frac)
        if drawn is not None:
            kind, path = drawn
            recipe.reverb = {"kind": kind, "ir": path}

    if _fires(rng, acfg.prob) and banks.noise is not None and len(banks.noise):
        recipe.noise = {
            "snr_db": round(rng.uniform(*acfg.noise.snr_db), 3),
            # The clip and the excerpt offset are both decided by this seed, so
            # the recipe stays two numbers rather than a path plus a sample index.
            "seed": rng.randrange(2**32),
        }

    if _fires(rng, acfg.prob):
        recipe.band_limit = {"cutoff_hz": round(rng.uniform(*acfg.band_limit.cutoff_hz), 1)}

    if _fires(rng, acfg.prob):
        recipe.clip = {
            "low_pct": round(rng.uniform(*acfg.clip.low_pct), 3),
            "high_pct": round(rng.uniform(*acfg.clip.high_pct), 3),
        }

    if _fires(rng, acfg.prob) and banks.codecs:
        kind = rng.choice(banks.codecs)
        entry: dict = {"kind": kind}
        if kind in ("opus", "mp3"):
            span = acfg.codec.opus_kbps if kind == "opus" else acfg.codec.mp3_kbps
            entry["kbps"] = int(rng.randint(int(span[0]), int(span[1])))
        recipe.codec = entry

    if _fires(rng, acfg.prob):
        recipe.packet_loss = {
            "frac": acfg.packet_loss.frac,
            "segment_ms": list(acfg.packet_loss.segment_ms),
            "seed": rng.randrange(2**32),
        }

    return recipe


def apply(
    x: np.ndarray,
    recipe: Recipe,
    sr: int,
    banks: Banks,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Run `recipe` over `x`. Pure: same inputs, same samples, every time.

    Length-preserving at every step, so the result still lines up with the VAD
    masks `build` holds for this channel.
    """
    y = np.asarray(x, dtype=np.float32)

    if recipe.reverb is not None and banks.rirs is not None:
        y = reverb.apply(y, banks.rirs.load(recipe.reverb["ir"]))

    if recipe.noise is not None and banks.noise is not None and len(banks.noise):
        bed, path = banks.noise.build(y.size, random.Random(recipe.noise["seed"]))
        # Recorded on the way through so meta.json names the clip, without
        # sampling having to touch the filesystem.
        recipe.noise.setdefault("file", path)
        y = noise_mod.add_noise(y, bed, mask, recipe.noise["snr_db"])

    if recipe.band_limit is not None:
        y = artifacts.band_limit(y, sr, recipe.band_limit["cutoff_hz"])

    if recipe.clip is not None:
        y = artifacts.clip(y, recipe.clip["low_pct"], recipe.clip["high_pct"])

    if recipe.codec is not None:
        y = codec.round_trip(y, sr, recipe.codec["kind"], recipe.codec.get("kbps"))

    if recipe.packet_loss is not None:
        y, _ = artifacts.packet_loss(
            y,
            sr,
            random.Random(recipe.packet_loss["seed"]),
            recipe.packet_loss["frac"],
            tuple(recipe.packet_loss["segment_ms"]),
        )

    return np.asarray(y, dtype=np.float32)


def _fires(rng: random.Random, probability: float) -> bool:
    """One Bernoulli draw, taken before the bank it guards is even looked at.

    Written as the left operand of every `and` above so the draw happens whether
    or not the step can run. A missing bank then costs one step, not the
    alignment of every draw after it.
    """
    return rng.random() < probability
