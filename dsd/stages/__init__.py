"""The stages, in the order they run.

`ORDER` is the single source of truth: the CLI builds a subcommand per entry,
`run --stages a,b,c` resolves names against it, and `run` with no `--stages`
executes PIPELINE start to finish. Adding a stage means writing the module and
appending it here.

`verify` is deliberately outside PIPELINE -- it reads the finished dataset back
off disk and is run on demand, not as part of building.
"""

from . import (
    augment,
    build,
    cluster,
    diarize,
    embed,
    enhance,
    filter_channels,
    gender,
    select,
    vad,
    verify,
)

ORDER = [
    diarize,
    filter_channels,
    vad,
    embed,
    cluster,
    gender,
    select,
    enhance,
    augment,
    build,
    verify,
]

PIPELINE = [module.NAME for module in ORDER if module.NAME != "verify"]

BY_NAME = {module.NAME: module for module in ORDER}

__all__ = ["ORDER", "PIPELINE", "BY_NAME"]
