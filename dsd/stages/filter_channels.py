"""Stage 2 -- keep only the calls where each channel holds exactly one speaker.

A call is usable as separation data only if channel 0 is one person start to
finish and channel 1 is a different person start to finish. Anything else -- a
call handed to a second agent, a speakerphone with two people on one side, a
diarizer that split one voice in two -- would put two speakers into a single
"source" and teach the model that a source may contain a conversation.

Recovering which channel a speaker is on has two modes, because two different
kinds of RTTM have to work here:

  `channel`  Trust the RTTM's channel field. Correct for RTTMs this pipeline
             produced, where stage 1 demuxed the audio and wrote the real index.

  `energy`   Assign each speaker label to the channel carrying most of its
             energy. Required for `Datasets/rttms/`, which came from
             `diarize_batch.py --mode mono`: it writes a constant in the channel
             field, so all 203444 lines say channel 1 and the field carries no
             information at all. Reading it as a channel -- which the notebook
             and `sandbox/sep_data_pipeline/filter_1.py` both do -- silently
             lands every speaker on the same channel.

`auto` picks between them by looking at whether the channel field actually
varies.

"Exactly one speaker" is measured in speech, not in label count -- a diarizer
run on one telephone channel invents a second label freely. See `_check` for the
threshold and why the rejected time is carried forward in `excluded`.

The energy purity check runs in *both* modes. In `channel` mode it is a guard
rather than a mapping: crosstalk that the diarizer heard as a second speaker
shows up as a label whose energy is spread across channels instead of
concentrated in one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..core.audio import read_audio
from ..core.discovery import collect_audio, index_by_call
from ..core.manifest import write_json
from ..core.rttm import intervals_of, merge_intervals, read_rttm, total_speech
from .base import banner, progress, require, summarize_counts

NAME = "filter"
REQUIRES = ("diarize",)


def add_args(parser) -> None:
    parser.add_argument(
        "--channel-mode",
        choices=["auto", "channel", "energy"],
        help="how to map a speaker label to a channel",
    )
    parser.add_argument("--workers", type=int, help="concurrent files (audio decode is the cost)")
    parser.add_argument("--limit", type=int, help="inspect only the first N calls")


def run(cfg, args) -> None:
    rttm_dir = cfg.paths.rttm_dir
    require({"RTTM directory": rttm_dir}, NAME)

    audios = collect_audio(cfg.paths.audio_dirs, cfg.paths.pattern)
    index = index_by_call(audios)

    rttms = sorted(rttm_dir.glob("*.rttm"))
    if args.limit:
        rttms = rttms[: args.limit]

    mode = args.channel_mode or cfg.filter.channel_mode
    workers = args.workers or cfg.filter.workers
    banner(NAME, f"{len(rttms)} RTTMs, channel_mode={mode}, {workers} workers")

    suitable: dict[str, dict] = {}
    unsuitable: dict[str, dict] = {}
    reasons: dict[str, int] = {}

    def inspect(rttm_path: Path) -> tuple[str, bool, dict]:
        call = rttm_path.stem
        audio_path = index.get(call)
        if audio_path is None:
            return call, False, {"reason": "no_audio"}
        return call, *_check(call, audio_path, rttm_path, cfg, mode)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(inspect, path) for path in rttms]
        for future in progress(as_completed(futures), "filter", total=len(futures)):
            try:
                call, ok, detail = future.result()
            except Exception as exc:
                # A file that cannot even be opened is a rejection, not a crash.
                call, ok, detail = "?", False, {"reason": "error", "error": repr(exc)}
            if ok:
                suitable[call] = detail
            else:
                unsuitable[call] = detail
                reasons[detail["reason"]] = reasons.get(detail["reason"], 0) + 1

    write_json(cfg.paths.suitable_json, dict(sorted(suitable.items())))
    write_json(cfg.paths.unsuitable_json, dict(sorted(unsuitable.items())))

    total = len(rttms)
    kept = len(suitable)
    banner(NAME, f"suitable: {kept}/{total} ({kept / total * 100:.1f}%)" if total else "no RTTMs")
    banner(NAME, "rejected:")
    summarize_counts(NAME, reasons, total)
    banner(NAME, f"wrote {cfg.paths.suitable_json.name} and {cfg.paths.unsuitable_json.name}")


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
def _check(call: str, audio_path: str, rttm_path: Path, cfg, mode: str) -> tuple[bool, dict]:
    segments = read_rttm(rttm_path)
    if not segments:
        return False, {"reason": "empty_rttm"}

    speakers = sorted({s.speaker for s in segments})
    expected = cfg.filter.expected_channels

    data, sr = read_audio(audio_path)
    n_channels = data.shape[1]
    if n_channels != expected:
        return False, {"reason": "wrong_channel_count", "channels": n_channels}

    resolved = mode
    if resolved == "auto":
        distinct = {s.channel for s in segments}
        resolved = "channel" if len(distinct) > 1 else "energy"

    energy = _speaker_channel_energy(segments, speakers, data, sr)
    totals = energy.sum(axis=1, keepdims=True)
    share = np.divide(energy, totals, out=np.zeros_like(energy), where=totals > 0)

    if resolved == "channel":
        assigned = []
        for speaker in speakers:
            channels = {s.channel for s in segments if s.speaker == speaker}
            if len(channels) != 1:
                return False, {"reason": "speaker_spans_channels", "speaker": speaker}
            assigned.append(next(iter(channels)) - 1)  # RTTM channels are 1-based
    else:
        assigned = [int(np.argmax(row)) for row in share]

    if any(c < 0 or c >= n_channels for c in assigned):
        return False, {"reason": "channel_out_of_range", "assigned": assigned}

    speech = {speaker: total_speech(segments, speaker) for speaker in speakers}

    labels_by_channel: dict[int, list[str]] = {}
    for index, speaker in enumerate(speakers):
        labels_by_channel.setdefault(assigned[index], []).append(speaker)

    if len(labels_by_channel) != expected:
        # A channel nobody was assigned to is a channel that was silent end to
        # end, or one whose only speaker was louder on the other side.
        return False, {
            "reason": f"n_channels={len(labels_by_channel)}",
            "speakers": speakers,
            "assigned": assigned,
        }

    # A channel may carry more than one label without carrying more than one
    # person. Measured over 30 calls, a second label is a median of 1.5 s
    # against a main speaker's 40 s -- the diarizer flipping on a breath, a
    # noise burst, or the faint bleed of the other party. Rejecting those the
    # same way as a genuine second speaker threw away 50% of the corpus.
    #
    # So a minor label is only a second *person* if it clears both an absolute
    # and a relative bar. Below that it is noise, and rather than trusting that
    # judgement its time is recorded in `excluded` and cut out of the source
    # downstream -- if the call really did contain a brief third voice, that
    # voice is removed instead of being mixed into somebody's clean channel.
    dominant: dict[int, str] = {}
    excluded: dict[int, list[tuple[float, float]]] = {}
    for channel, labels in labels_by_channel.items():
        labels = sorted(labels, key=lambda s: -speech[s])
        main = labels[0]
        dominant[channel] = main

        for minor in labels[1:]:
            share_of_main = speech[minor] / speech[main] if speech[main] > 0 else 1.0
            if (
                speech[minor] >= cfg.filter.min_label_speech_sec
                and share_of_main >= cfg.filter.min_label_share
            ):
                return False, {
                    "reason": "two_speakers_one_channel",
                    "channel": channel,
                    "labels": labels,
                    "speech_sec": {k: round(speech[k], 3) for k in labels},
                }
            excluded.setdefault(channel, []).extend(intervals_of(segments, minor))

    rows = {speaker: i for i, speaker in enumerate(speakers)}
    purity = {
        label: float(share[rows[label], channel]) for channel, label in dominant.items()
    }
    # Strict in `energy` mode, where purity is what makes the argmax assignment
    # trustworthy; loose in `channel` mode, where the assignment is the demux
    # and this only catches a speaker somehow louder on the other channel.
    floor = (
        cfg.filter.purity_min
        if resolved == "energy"
        else cfg.filter.purity_min_channel_mode
    )
    if min(purity.values()) < floor:
        return False, {"reason": "low_purity", "purity": purity, "floor": floor}

    if min(speech[label] for label in dominant.values()) < cfg.filter.min_speech_sec:
        return False, {
            "reason": "channel_silent",
            "speech_sec": {k: round(speech[k], 3) for k in dominant.values()},
        }

    return True, {
        "audio": str(audio_path),
        "rttm": str(rttm_path),
        "mode": resolved,
        "duration": round(len(data) / sr, 3),
        "sample_rate": sr,
        # channel index (as a string key, since this becomes JSON) -> speaker label
        "channels": {str(channel): label for channel, label in dominant.items()},
        "purity": {k: round(v, 4) for k, v in purity.items()},
        "speech_sec": {label: round(speech[label], 3) for label in dominant.values()},
        # Minor-label time to cut out of this channel's speech, in seconds.
        "excluded": {
            str(channel): [[round(a, 3), round(b, 3)] for a, b in merge_intervals(spans)]
            for channel, spans in excluded.items()
        },
    }


def _speaker_channel_energy(segments, speakers, data: np.ndarray, sr: int) -> np.ndarray:
    """Sum of squared samples per (speaker, channel) over that speaker's segments."""
    rows = {speaker: i for i, speaker in enumerate(speakers)}
    energy = np.zeros((len(speakers), data.shape[1]), dtype=np.float64)
    for segment in segments:
        row = rows.get(segment.speaker)
        if row is None:
            continue
        start = max(0, int(segment.start * sr))
        end = min(len(data), int(segment.end * sr))
        if end > start:
            energy[row] += np.square(data[start:end], dtype=np.float64).sum(axis=0)
    return energy
