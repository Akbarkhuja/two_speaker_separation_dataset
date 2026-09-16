"""Stage 9 -- prepare the degradation assets, into `work/augment/`.

This stage never touches a call. It builds the three banks that `build` draws
from when it writes degraded copies of the mixture, and it does so once:

    irs/        the 270 MIT survey impulse responses, resampled to our rate
    rirs/       N simulated pyroomacoustics shoebox rooms
    noise.json  the WHAM! clips found free of intelligible speech
    index.json  what was built, from where, and under which settings

Its own stage rather than part of `build` for the same reason `enhance` is:
`build` is the stage that gets re-run constantly while tuning, and simulating
two thousand rooms with an image-source model, or running a VAD over six
thousand noise clips, is not something to repeat on every rebuild. Both are
also pure functions of `cfg.seed`, so caching them costs nothing in variety.

The two source corpora live in the sibling `diar_syn_dataset` project. They are
read here and nowhere else; everything downstream reads `work/augment/`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import soundfile as sf

from ..augment import chain
from ..augment import noise as noise_mod
from ..augment import reverb as reverb_mod
from ..backends import vad as vad_backends  # noqa: F401  (registers backends)
from ..core.audio import resample, write_wav
from ..core.manifest import write_json
from ..registry import VADS
from .base import banner, progress

NAME = "augment"
REQUIRES = ()


def add_args(parser) -> None:
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild banks that already exist"
    )
    parser.add_argument(
        "--simulated", type=int, help="how many shoebox rooms to simulate"
    )
    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="skip the speech screen and use every noise clip",
    )


# --------------------------------------------------------------------------- #
# what `build` calls
# --------------------------------------------------------------------------- #
def index_path(cfg) -> Path:
    return cfg.paths.augment_dir / "index.json"


def load_banks(cfg) -> chain.Banks:
    """Open the prepared banks. Raises if this stage has not run.

    Called once per `build` run, not per call: the impulse responses and noise
    clips are cached in memory inside the bank objects, and rebuilding them per
    call would re-read the same files thousands of times.
    """
    index = index_path(cfg)
    if not index.exists():
        raise SystemExit(
            f"[{NAME}] build.augment.variants is set but {index} is missing.\n"
            f"[{NAME}] run `python -m dsd augment` first, or set "
            "build.augment.variants=0 to write only the clean mixture."
        )

    directory = cfg.paths.augment_dir
    noise_files = json.loads((directory / "noise.json").read_text(encoding="utf-8"))
    available = chain.codec.available()
    wanted = list(cfg.build.augment.codec.kinds)
    codecs = [k for k in wanted if k in available]
    if len(codecs) < len(wanted):
        banner(
            NAME,
            f"ffmpeg cannot encode {sorted(set(wanted) - set(codecs))}; "
            f"the codec step will draw from {codecs or 'nothing'}",
        )

    return chain.Banks(
        rirs=reverb_mod.RIRBank(
            directory / "rirs", directory / "irs", cfg.sample_rate
        ),
        noise=noise_mod.NoiseBank(noise_files, cfg.sample_rate),
        codecs=codecs,
    )


# --------------------------------------------------------------------------- #
def run(cfg, args) -> None:
    acfg = cfg.augment
    directory = cfg.paths.augment_dir
    directory.mkdir(parents=True, exist_ok=True)

    n_simulated = args.simulated if args.simulated is not None else acfg.simulated_rirs
    banner(NAME, f"assets -> {directory}  (sample rate {cfg.sample_rate})")

    measured = _prepare_measured(cfg, acfg, directory / "irs", args.overwrite)
    simulated = _prepare_simulated(cfg, acfg, directory / "rirs", n_simulated, args.overwrite)
    noise_files = _prepare_noise(cfg, acfg, directory / "noise.json", args)

    codecs_wanted = list(cfg.build.augment.codec.kinds)
    codecs_have = [k for k in codecs_wanted if k in chain.codec.available()]

    write_json(
        index_path(cfg),
        {
            "sample_rate": cfg.sample_rate,
            "seed": cfg.seed,
            "measured_irs": measured,
            "simulated_rirs": simulated,
            "noise_clips": len(noise_files),
            "codecs_available": codecs_have,
            "codecs_missing": sorted(set(codecs_wanted) - set(codecs_have)),
            "sources": {
                "ir_dir": acfg.ir_dir,
                "noise_dir": acfg.noise_dir,
                "noise_screen_json": acfg.noise_screen_json,
            },
            "config": {
                "rt60": list(acfg.rt60),
                "room_dim": list(acfg.room_dim),
                "ir_max_sec": acfg.ir_max_sec,
                "max_noise_speech_frac": acfg.max_noise_speech_frac,
            },
        },
    )

    banner(
        NAME,
        f"ready: {simulated} simulated rooms, {measured} measured, "
        f"{len(noise_files):,} noise clips, codecs {codecs_have or 'none'}",
    )
    if not (simulated or measured):
        banner(NAME, "WARNING: no impulse responses -- the reverb step will never fire")
    if not noise_files:
        banner(NAME, "WARNING: no noise clips -- the background-noise step will never fire")
    if not codecs_have:
        banner(NAME, "WARNING: no usable codec -- the codec step will never fire")


# --------------------------------------------------------------------------- #
def _prepare_measured(cfg, acfg, out_dir: Path, overwrite: bool) -> int:
    """Resample the measured impulse responses into the bank, at our rate.

    They arrive peak-aligned and unit-peak already, but resampling moves the
    peak by a fraction of a sample and rescales it, so `align` is re-applied
    afterwards rather than trusted from the source.
    """
    if acfg.ir_dir is None:
        banner(NAME, "augment.ir_dir is null; no measured impulse responses")
        return 0
    source = Path(acfg.ir_dir)
    if not source.is_dir():
        banner(NAME, f"augment.ir_dir does not exist: {source}")
        return 0

    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in {".wav", ".flac"})
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in progress(files, "irs"):
        target = out_dir / f"{path.stem}.wav"
        if target.exists() and not overwrite:
            written += 1
            continue
        try:
            data, file_sr = sf.read(path, dtype="float32", always_2d=True)
            ir = reverb_mod.align(
                resample(data[:, 0], file_sr, cfg.sample_rate),
                cfg.sample_rate,
                acfg.ir_max_sec,
            )
        except Exception as exc:
            banner(NAME, f"skipping {path.name}: {exc!r}")
            continue
        if ir.size < 2:
            continue
        write_wav(target, ir, cfg.sample_rate, subtype="FLOAT")
        written += 1

    banner(NAME, f"measured impulse responses: {written} in {out_dir}")
    return written


def _prepare_simulated(cfg, acfg, out_dir: Path, count: int, overwrite: bool) -> int:
    """Simulate shoebox rooms until the bank holds `count` of them.

    Resumable by construction: the room for index `i` is drawn from a seed
    derived from `i`, so an interrupted run continues rather than restarting,
    and the bank is identical however many times it took to fill.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if count <= 0:
        banner(NAME, "augment.simulated_rirs is 0; no simulated rooms")
        return 0

    try:
        import pyroomacoustics  # noqa: F401
    except ImportError:
        banner(
            NAME,
            "pyroomacoustics is not installed, so no rooms can be simulated "
            "(`pip install pyroomacoustics`); falling back to measured IRs only",
        )
        return len(list(out_dir.glob("*.wav")))

    todo = [
        i
        for i in range(count)
        if overwrite or not (out_dir / f"rir{i:05d}.wav").exists()
    ]
    if todo:
        meta: list[dict] = []
        for index in progress(todo, "rooms"):
            rng = random.Random(f"{cfg.seed}:rir:{index}")
            try:
                ir, draws = reverb_mod.simulate(
                    rng, cfg.sample_rate, tuple(acfg.rt60), tuple(acfg.room_dim),
                    acfg.ir_max_sec,
                )
            except Exception as exc:
                banner(NAME, f"room {index} failed: {exc!r}")
                continue
            write_wav(out_dir / f"rir{index:05d}.wav", ir, cfg.sample_rate, subtype="FLOAT")
            meta.append({"index": index, **draws})
        if meta:
            # The draws behind each room, so an RT60 histogram of the bank does
            # not require re-measuring the impulse responses.
            _append_json(out_dir.parent / "rirs.json", meta)

    written = len(list(out_dir.glob("*.wav")))
    banner(NAME, f"simulated rooms: {written} in {out_dir}")
    return written


def _prepare_noise(cfg, acfg, out_path: Path, args) -> list[str]:
    """The screened noise pool, as a list of absolute paths."""
    if out_path.exists() and not args.overwrite:
        files = json.loads(out_path.read_text(encoding="utf-8"))
        banner(NAME, f"noise pool: {len(files):,} clips (cached in {out_path.name})")
        return files

    if acfg.noise_dir is None:
        banner(NAME, "augment.noise_dir is null; no background noise")
        write_json(out_path, [])
        return []
    source = Path(acfg.noise_dir)
    if not source.is_dir():
        banner(NAME, f"augment.noise_dir does not exist: {source}")
        write_json(out_path, [])
        return []

    every = sorted(
        str(p) for p in source.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg"}
    )
    banner(NAME, f"noise corpus: {len(every):,} clips under {source}")
    if not every:
        write_json(out_path, [])
        return []

    if args.no_screen:
        banner(NAME, "--no-screen: keeping every clip, speech and all")
        write_json(out_path, every)
        return every

    imported = _import_screen(acfg.noise_screen_json, source, every)
    if imported is not None:
        banner(
            NAME,
            f"noise pool: {len(imported):,} clips, imported from "
            f"{Path(acfg.noise_screen_json).name} (no VAD pass needed)",
        )
        write_json(out_path, imported)
        return imported

    kept = _screen(cfg, acfg, every, args)
    write_json(out_path, kept)
    return kept


def _import_screen(screen_json, source: Path, every: list[str]) -> list[str] | None:
    """Re-root a pre-computed allowlist onto this machine's copy of the corpus.

    The sibling project stored its paths relative to its own root
    (`data/augmentation_data/wham_noise/tt/....wav`). Only the part below the
    corpus directory is portable, so that is what is matched on; anything that
    does not resolve to a file we actually have is dropped rather than trusted.
    """
    if not screen_json:
        return None
    path = Path(screen_json)
    if not path.exists():
        return None
    try:
        listed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(listed, list) or not listed:
        return None

    have = {str(Path(p).relative_to(source)): p for p in every}
    marker = f"{source.name}/"
    kept = []
    for entry in listed:
        text = str(entry).replace("\\", "/")
        index = text.rfind(marker)
        relative = text[index + len(marker) :] if index >= 0 else Path(text).name
        match = have.get(relative)
        if match is not None:
            kept.append(match)
    # A handful of misses is re-rooting working; almost none matching means the
    # allowlist describes a different corpus and should not be trusted.
    return kept if len(kept) >= 0.5 * len(listed) else None


def _screen(cfg, acfg, every: list[str], args) -> list[str]:
    """Drop noise clips that contain intelligible speech.

    A clip with a talker in it becomes an unlabelled third voice in the mixture,
    with no target for the model to put it in -- strictly worse than no noise.
    Screened at our own sample rate with the pipeline's own VAD, so the decision
    matches the one every other stage would make.
    """
    rng = random.Random(f"{cfg.seed}:noise-screen")
    files = list(every)
    if acfg.screen_limit and len(files) > acfg.screen_limit:
        files = sorted(rng.sample(files, acfg.screen_limit))
        banner(NAME, f"screening a random {len(files):,} of {len(every):,} clips")

    detector = VADS.create(cfg.vad.backend, cfg.vad.options.get(cfg.vad.backend, {}))
    kept: list[str] = []
    for path in progress(files, "screen"):
        try:
            data, file_sr = sf.read(path, dtype="float32", always_2d=True)
            mono = resample(data[:, 0], file_sr, cfg.sample_rate)
            if mono.size < cfg.sample_rate // 2:
                continue
            speech = sum(b - a for a, b in detector.speech(mono, cfg.sample_rate))
        except Exception:
            continue
        if speech / (mono.size / cfg.sample_rate) <= acfg.max_noise_speech_frac:
            kept.append(path)

    banner(
        NAME,
        f"noise pool: kept {len(kept):,}/{len(files):,} clips "
        f"(speech fraction <= {acfg.max_noise_speech_frac:.0%})",
    )
    return kept


def _append_json(path: Path, rows: list[dict]) -> None:
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = []
    merged = {row["index"]: row for row in existing + rows}
    write_json(path, [merged[k] for k in sorted(merged)])
