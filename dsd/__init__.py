"""Build a 2-speaker separation dataset from dual-channel telephony recordings.

Each source call is a stereo recording where the two parties sit on separate,
physically isolated channels, so once the non-speech is silenced each channel is
a clean source and their sum is an exact, fully-labelled mixture.

The pipeline is eight resumable stages, each reading the previous one's artifact
out of `work/`:

    diarize -> filter -> vad -> embed -> cluster -> gender -> select -> build

Run `python -m dsd --help` for the CLI.
"""

__version__ = "0.1.0"
