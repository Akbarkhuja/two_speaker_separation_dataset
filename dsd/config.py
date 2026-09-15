"""The whole pipeline's configuration as one nested dataclass tree.

Loaded from `configs/default.yaml`, overridable per-run with `--set a.b=c`. Two
rules keep this from turning into a swamp:

  - Stage settings that the *stage* reads are typed dataclass fields.
  - Settings that only a single backend understands live in an untyped
    `options[<backend name>]` dict, so registering a new backend never means
    editing this file.

`cfg` is threaded into functions by argument, never read from a module global,
so a caller can build a variant with `replace(cfg.build, chunks=True)` and not
disturb anything else.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# stage configs
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    root: Path = Path(".")
    # Directories searched for source audio; `pattern` is applied inside each.
    audio_dirs: list[str] = field(default_factory=lambda: ["Datasets/audios"])
    pattern: str = "**/*.opus"
    work_dir: Path = Path("work")
    dataset_dir: Path = Path("dataset")

    def resolve(self) -> None:
        """Make every path absolute against `root`, once, at load time.

        Stages write into `work_dir` from whatever CWD the user happened to be
        in; leaving these relative means a run from a subdirectory silently
        starts a second, empty `work/`.
        """
        self.root = self.root.expanduser().resolve()
        self.work_dir = self._under_root(self.work_dir)
        self.dataset_dir = self._under_root(self.dataset_dir)
        self.audio_dirs = [str(self._under_root(Path(d))) for d in self.audio_dirs]

    def _under_root(self, path: Path) -> Path:
        path = path.expanduser()
        return path if path.is_absolute() else (self.root / path)

    # Convenience accessors so stages never rebuild these strings by hand.
    @property
    def rttm_dir(self) -> Path:
        return self.work_dir / "rttms"

    @property
    def vad_dir(self) -> Path:
        return self.work_dir / "vad"

    @property
    def suitable_json(self) -> Path:
        return self.work_dir / "suitable.json"

    @property
    def unsuitable_json(self) -> Path:
        return self.work_dir / "unsuitable.json"

    @property
    def embeddings_npz(self) -> Path:
        return self.work_dir / "embeddings.npz"

    @property
    def embed_index_json(self) -> Path:
        return self.work_dir / "embed_index.json"

    @property
    def speakers_json(self) -> Path:
        return self.work_dir / "speakers.json"

    @property
    def gender_json(self) -> Path:
        return self.work_dir / "gender.json"

    @property
    def selection_json(self) -> Path:
        return self.work_dir / "selection.json"

    @property
    def enhanced_dir(self) -> Path:
        return self.work_dir / "enhanced"


@dataclass
class DiarizeConfig:
    backend: str = "sortformer"
    # Per-file worker threads. Only useful for network-bound backends; the
    # local GPU ones ignore it and run serially.
    workers: int = 1
    options: dict[str, dict] = field(default_factory=dict)


@dataclass
class FilterConfig:
    # 'auto' picks `channel` when the RTTM's channel field varies and `energy`
    # when it is constant -- which is exactly the mono-MOSS case in
    # Datasets/rttms, where the field is always "1" and carries no information.
    channel_mode: str = "auto"
    expected_channels: int = 2
    # Fraction of a speaker's segment energy that must land in its assigned
    # channel, applied only in `energy` mode -- there the argmax IS the
    # assignment, so this is what makes it trustworthy.
    purity_min: float = 0.9
    # The same measure in `channel` mode, where the assignment came from the
    # demux and cannot be wrong, so this is only a sanity check. It has to be
    # loose: the measure divides by the *other* channel's energy during this
    # speaker's turns, so ordinary line noise on a quiet channel drags a
    # perfectly clean source down to 0.72-0.86. Only a value below 0.5 -- a
    # speaker louder on someone else's channel than their own -- means
    # something is actually broken.
    purity_min_channel_mode: float = 0.5
    min_speech_sec: float = 3.0
    # A diarizer label on a channel only counts as a second *person* if it
    # carries this much speech, both absolutely and relative to that channel's
    # main speaker. Below it the label is treated as noise and its time is
    # excluded from the source rather than the whole call being thrown away.
    min_label_speech_sec: float = 2.0
    min_label_share: float = 0.10
    # Decoding every opus file is the cost here, and it releases the GIL.
    workers: int = 8


@dataclass
class VadConfig:
    backend: str = "silero"
    options: dict[str, dict] = field(default_factory=dict)


@dataclass
class EmbedConfig:
    backend: str = "titanet"
    # Segments shorter than this carry too little voice to embed well; longer
    # ones are split so no single chunk dominates the duration-weighted mean.
    min_speech_duration: float = 1.2
    longest_chunk_duration: float = 20.0
    expected_speakers: int = 2
    batch_size: int = 16
    options: dict[str, dict] = field(default_factory=dict)


@dataclass
class ClusterConfig:
    backend: str = "constrained_ahc"
    # On the rescaled (cos + 1) / 2 scale, i.e. raw cosine 0.6.
    threshold: float = 0.8
    report: bool = True
    options: dict[str, dict] = field(default_factory=dict)


@dataclass
class GenderConfig:
    backend: str = "http_ecapa"
    # Number of longest VAD segments per speaker to classify, and the total
    # audio budget across them.
    segments_per_speaker: int = 3
    max_total_sec: float = 30.0
    min_segment_sec: float = 1.0
    workers: int = 4
    options: dict[str, dict] = field(default_factory=dict)


@dataclass
class SelectConfig:
    # Seconds of one speaker's own speech -- VAD speech minus `excluded` spans,
    # exactly what lands in s1/s2 -- allowed across the whole selection. This is
    # the primary cap: what a model hears of a voice is time, not call count, and
    # per-side speech in one call ranges from 5 s to ~14 min. Measured at matched
    # size (~40 h), a 20-minute cap kept 23% more calls and 15% more speakers than
    # the equivalent call cap, and held the loudest voice at 20.0 min instead of
    # 31.5. `null` or <= 0 turns it off.
    max_duration_per_speaker: float | None = 1200.0
    # Optional secondary cap on appearances. `null` or <= 0 turns it off.
    max_calls_per_speaker: int | None = None
    balance_gender: bool = True
    # Tolerated deviation from 50/50 before the majority class is downsampled.
    gender_tolerance: float = 0.02
    splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.90, "dev": 0.05, "test": 0.05}
    )
    min_speech_sec: float = 5.0


@dataclass
class EnhanceConfig:
    backend: str = "http_mossformergan"
    # 'speech' enhances only the VAD spans; everything outside them is zerofied
    # by `build` regardless, so enhancing it is pure waste. 'full' does the
    # whole channel, which is what the model was trained on but ~1.7x the audio.
    regions: str = "speech"
    # VAD averages ~43 segments per call. Merging with a gap keeps that from
    # becoming a request storm for no gain.
    merge_gap: float = 0.5
    # Run-in context each side of a span, so the model does not start cold on a
    # word boundary. Trimmed off again before splicing.
    context_sec: float = 0.5
    min_span_sec: float = 0.3
    # Container for work/enhanced/. FLAC is lossless at PCM_16 and about half
    # the size of WAV over speech.
    format: str = "flac"
    # Measured throughput does not rise with concurrency (3.17x / 3.28x / 3.27x
    # realtime at 1 / 4 / 8 workers), but a second request in flight keeps the
    # GPU fed while the client encodes the next one.
    workers: int = 2
    options: dict[str, dict] = field(default_factory=dict)


@dataclass
class BuildConfig:
    # 'auto' uses work/enhanced/<call> when it exists; 'always' fails when it
    # does not, so a half-enhanced dataset cannot be built by accident; 'never'
    # ignores the cache.
    use_enhanced: str = "auto"
    # What goes into mix.wav.
    #   False (default) -- the two channels as recorded are summed, so the
    #     background between the turns survives into the model's input while
    #     the targets stay clean. `mix == s1 + s2` no longer holds: the
    #     difference is exactly that background.
    #   True -- both channels are zerofied first and the mixture is their sum,
    #     which makes `mix == s1 + s2` exact but leaves the model's input
    #     digitally silent whenever nobody is talking.
    # The targets s1/s2 are zerofied either way.
    zerofy_mix: bool = False
    # Raised-cosine fade at every VAD mask edge. Hard zeroing leaves a click at
    # each boundary that correlates perfectly with the label, and a separation
    # model will learn the clicks instead of the voices.
    fade_ms: float = 10.0
    # Overlap boosting. Off by default: rolling s2 rearranges the call in time
    # to manufacture overlap, which is exactly what makes a built call sound
    # shuffled rather than like a recorded conversation. With it off every call
    # is left as recorded and `natural_frac` / `target_overlap` /
    # `seam_guard_ms` below are unused.
    shuffle: bool = False
    # Fraction of calls left exactly as recorded (~3% natural overlap); the
    # rest get s2 shifted to land inside `target_overlap`. Only consulted when
    # `shuffle` is on.
    natural_frac: float = 0.6
    target_overlap: tuple[float, float] = (0.15, 0.60)
    # The shift is circular, so it wraps somewhere. Only offsets whose wrap
    # seams sit in this much silence are considered, which keeps an utterance
    # from being cut in half with its two halves at opposite ends of the file.
    # 200 ms is comfortably clear of the 10 ms zerofy fade and of a plosive onset.
    seam_guard_ms: float = 200.0
    sir_db: tuple[float, float] = (-5.0, 5.0)
    peak_ceiling: float = 0.99
    chunks: bool = False
    chunk_sec: float = 4.0
    chunk_hop: float = 2.0
    min_active_per_src: float = 0.5


@dataclass
class Config:
    seed: int = 1234
    sample_rate: int = 8000
    paths: PathsConfig = field(default_factory=PathsConfig)
    diarize: DiarizeConfig = field(default_factory=DiarizeConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    gender: GenderConfig = field(default_factory=GenderConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    enhance: EnhanceConfig = field(default_factory=EnhanceConfig)
    build: BuildConfig = field(default_factory=BuildConfig)

    # ----------------------------------------------------------------- #
    # loading
    # ----------------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: list[str] | None = None,
        root: str | Path | None = None,
    ) -> "Config":
        data: dict[str, Any] = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            data = raw or {}
        for item in overrides or []:
            _apply_override(data, item)
        if root is not None:
            data.setdefault("paths", {})["root"] = str(root)

        cfg = _from_dict(cls, data)
        cfg.paths.resolve()
        _resolve_option_paths(cfg)
        return cfg

    def to_dict(self) -> dict:
        return _to_plain(dataclasses.asdict(self))


def _resolve_option_paths(cfg: "Config") -> None:
    """Make relative `*_dir` / `*_path` backend options absolute against root.

    Backend options are untyped by design, so nothing else knows they hold
    paths. Without this, `rttm_dir: Datasets/rttms` resolves against whatever
    directory the user ran from, and the stage reports "no RTTM for <call>"
    rather than a missing directory. Absolute values are left alone, so model
    paths are untouched.
    """
    root = cfg.paths.root
    for stage in dataclasses.fields(cfg):
        section = getattr(cfg, stage.name)
        options = getattr(section, "options", None)
        if not isinstance(options, dict):
            continue
        for backend_options in options.values():
            if not isinstance(backend_options, dict):
                continue
            for key, value in backend_options.items():
                if (
                    (key.endswith("_dir") or key.endswith("_path"))
                    and isinstance(value, str)
                    and not Path(value).is_absolute()
                ):
                    backend_options[key] = str(root / value)


# --------------------------------------------------------------------------- #
# dict <-> dataclass plumbing
# --------------------------------------------------------------------------- #
def _apply_override(data: dict, item: str) -> None:
    """Apply one `a.b.c=value` override in place; value is parsed as YAML.

    YAML parsing is what makes `--set build.chunks=true` and
    `--set build.sir_db=[-2,2]` behave as the types the dataclass expects
    instead of arriving as strings.
    """
    key, sep, value = item.partition("=")
    if not sep:
        raise ValueError(f"--set expects key=value, got {item!r}")

    node = data
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"--set {item!r}: {part!r} is not a section")
    node[parts[-1]] = yaml.safe_load(value)


def _from_dict(cls: type, data: Any) -> Any:
    """Recursively build a dataclass tree from plain dicts, coercing field types."""
    if not dataclasses.is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {', '.join(sorted(unknown))}; "
            f"expected one of {', '.join(sorted(known))}"
        )

    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce(hints[name], value)
    return cls(**kwargs)


def _coerce(hint: Any, value: Any) -> Any:
    if dataclasses.is_dataclass(hint):
        return _from_dict(hint, value)
    if hint is Path:
        return Path(value)

    origin = typing.get_origin(hint)
    if origin in (types.UnionType, typing.Union):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        return _coerce(args[0], value) if (value is not None and args) else value
    if origin is tuple:
        return tuple(value)
    if origin is list:
        return list(value)
    return value


def _to_plain(value: Any) -> Any:
    """Make an asdict() result JSON-serializable (Path -> str, tuple -> list)."""
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
