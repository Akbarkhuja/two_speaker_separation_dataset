"""DialogueSidon-style degradation of a single track.

The pieces, in the order the chain applies them:

    reverb.py     simulated shoebox rooms and real measured impulse responses
    noise.py      a WHAM! clip, looped to length, added at a target SNR
    artifacts.py  band limitation, clipping, packet loss
    codec.py      a telephony codec round trip through ffmpeg
    chain.py      draws a Recipe and applies it

`chain.sample()` and `chain.apply()` are the only two functions `build` calls;
everything else is reachable for tests and for use on its own.

The assets these draw from are prepared once by the `augment` stage, into
`work/augment/`. Nothing here reads the sibling project's data directory
directly -- that path is the `augment` stage's input, not this package's.
"""

from .chain import STEPS, Banks, Recipe, apply, sample

__all__ = ["STEPS", "Banks", "Recipe", "apply", "sample"]
