# dual_separation_dataset

Turns dual-channel telephony recordings into a 2-speaker separation dataset:
one mixture plus two ground-truth source signals per call. **The mixture is always
the two original `.opus` channels summed**, so it keeps the noise floor, the room
and the telephone line exactly as recorded. The two targets are each speaker's own
channel — denoised, when the `enhance` stage has run — with everything outside
their speech faded out. The model's input is a real call; its outputs are clean
speech. `build.zerofy_mix=true` gives the strict `mix == s1 + s2` dataset instead,
the one mode where the mixture is built from the targets.

The corpus works for this because **the two parties sit on physically isolated
channels**. Measured across 158 calls, cross-channel `|corr|` is mean 0.0006 /
max 0.005 — there is no echo bleed, so once the non-speech is silenced each
channel *is* a clean source and their sum is an exact, fully-labelled mixture
rather than an approximation.

---

## Setup

Everything runs in the existing `nemo` conda env:

```bash
conda activate nemo          # /home/akbar/miniconda3/envs/nemo
cd /home/akbar/craft/prod/dual_separation_dataset
python -m dsd --help
```

See `requirements.txt` for what is used and which stage needs it. Heavy imports
(torch, NeMo, silero, librosa) are lazy — `python -m dsd backends` and `--help`
load none of them.

---

## Running it

```bash
python -m dsd run                       # every stage, in order
python -m dsd run --stages vad,embed    # a slice
python -m dsd run --from cluster        # from a stage onward
python -m dsd diarize --limit 30        # one stage
python -m dsd verify                    # check a finished dataset
python -m dsd backends                  # what is registered
python -m dsd config                    # the resolved configuration
```

Global flags go **before** the subcommand:

```bash
python -m dsd --set build.chunks=true --set select.max_calls_per_speaker=2 run
python -m dsd --set paths.work_dir=work_experiment run --stages diarize,filter
```

Every stage is idempotent and resumable. Per-file stages (`diarize`, `vad`)
skip a call whose output already exists; whole-artifact stages skip unless
`--overwrite`. A call that raises is recorded in `work/<stage>_failed.tsv` and
the run continues — with 5600 calls, one unreadable file must not cost the rest.

### On real data

```bash
python -m dsd run --stages diarize,filter,vad,embed,cluster,select,build \
                  --limit 30 --chunks   # `enhance` left out: it is the slow one
python -m dsd verify
```

`gender` is omitted because it needs a service (below); `select` skips balancing
with a warning when `work/gender.json` is absent, so the chain still completes.

---

## Tests

```bash
python tests/smoke.py             # 62 checks in ~6 s. No GPU, no network, no weights.
python tests/smoke.py --real 10   # additionally: 10 real calls through the real models
python tests/smoke.py --only fade # run checks matching a substring
```

`dsd verify` checks a finished dataset; `tests/smoke.py` checks the code that
produced it. Four groups:

- **unit** — pure functions against brute force where an independent
  implementation is cheap: interval subtraction on a sample grid, the overlap FFT
  against `np.roll` at *every* shift, `mix == s1 + s2` surviving the peak guard.
- **config** — override typing, `*_dir` resolution against root, registry contract.
- **http** — the MOSS, gender and enhancement clients replayed against the
  payloads their servers are documented to return, via a local stub server.
- **e2e** — a synthetic corpus of eight planted calls driven through all nine
  stages with stub backends. Because the corpus is constructed, the assertions are
  exact: which calls survive the filter and *why each other one dies*, that the
  clustering recovers precisely the eight voices put in, that splits are
  speaker-disjoint, and that an excluded minor-label span is actually silent in
  the built wav.

Adding a backend? Register a stub in the test the way `StubVAD` / `StubEmbedder` /
`StubGender` / `StubEnhancer` do — the e2e run needs no GPU because of them.

---

## The stages

```
diarize -> filter -> vad -> embed -> cluster -> gender -> select -> enhance -> augment -> build
```

| stage | writes | what it does |
|---|---|---|
| `diarize` | `work/rttms/<call>.rttm` | Demuxes each channel and diarizes it alone, so the **channel is the speaker identity**. Labels are namespaced `ch{c}_{label}`. |
| `filter` | `work/suitable.json`, `work/unsuitable.json` | Keeps calls where each channel holds exactly one speaker. Every rejection carries a `reason`. |
| `vad` | `work/vad/<call>.rttm` | Per-channel Silero speech masks — the source of truth for zerofying and chunking. |
| `embed` | `work/embeddings.npz`, `work/embed_index.json` | One TitaNet-L vector per speaker per call, duration-weighted. |
| `cluster` | `work/speakers.json` | Global speaker identities via constrained complete-linkage AHC. |
| `gender` | `work/gender.json` | Male/female per global speaker, for balancing. |
| `select` | `work/selection.json` | Per-speaker **speech-time** cap, gender balance, speaker-disjoint splits. |
| `enhance` | `work/enhanced/<call>.flac` | Denoises both channels via the MossFormerGAN service, **before** they are summed. Optional but cached. |
| `augment` | `work/augment/` | Prepares the degradation assets: simulated shoebox rooms, measured impulse responses, and the screened noise pool. Touches no call; runs in ~7 s and caches. |
| `build` | `dataset/` | Mixture from the **original** channels; targets from the enhanced (or original) channels, zerofied. Shift and SIR are applied to both copies alike. Also writes the degraded mixtures. Prunes call directories the selection dropped. |

### Augmentation

`build` can write degraded copies of each mixture beside the clean one, following
DialogueSidon (arXiv:2604.09344, Appendix A): reverberation, background noise, band
limitation, clipping, a codec and packet loss, each firing at p = 0.5 **per track**.
The two channels are degraded independently and only then summed — they are two
telephone legs with their own line and handset, so one shared room and one shared
codec would be the wrong task.

`s1`/`s2` are never degraded and are shared by every variant: the training pair is
(degraded mixture, clean targets). Four variants is the paper's number and turns
~95 h of calls into ~475 h of pairs for ~22 GB. `build.augment.variants: 0` switches
it off entirely.

Two of the paper's seven steps are adapted, because this corpus is 8 kHz telephony
and 99.9% of its energy is already below 3.8 kHz: its band limitation resamples only
to rates at or above 8 kHz and would be an identity here, so a sub-Nyquist cutoff is
drawn instead; and MP3 at 65–245 kbps measures 25 dB SNR on this audio, so the codec
pool is G.711 µ-law/A-law, GSM 06.10 and low-bitrate Opus, which is what a call
actually goes through. The paper's seventh step, the mixing weight `w ~ U(0.3,0.7)`,
is `build.sir_db` under another name and is deliberately not applied twice.

### Output layout

```
dataset/
  {train,dev,test}/<call>/{mix,s1,s2}.wav + meta.json
  {train,dev,test}/<call>/mix_aug{0..N}.wav              # with build.augment.variants
  chunks/{train,dev,test}/{mix,s1,s2}/<call>_<idx>.wav   # with --chunks
  chunks/{train,dev,test}/mix_aug{0..N}/<call>_<idx>.wav

  manifest.jsonl                                          # one row per (call, variant)
  train.jsonl  dev.jsonl  test.jsonl                      # the same rows, per split
  chunks/manifest.jsonl                                   # every chunk
  chunks/train.jsonl  chunks/dev.jsonl  chunks/test.jsonl

  stats.json                                              # realized vs target
```

Point a trainer straight at a split — `dataset/chunks/train.jsonl`. Every configured
split gets a file even when it holds nothing, so a path in a training command is
never missing; `verify` checks the split files partition the combined manifest
exactly, which is what catches one left stale by an earlier build.

With augmentation on, a call contributes several rows that share a `call`, an `s1`
and an `s2` and differ only in `mix`. The clean mixture carries `variant: null` and
each degraded one its index, so a trainer can take the file whole or filter to
either half. Anything counting calls or speakers has to fold on `call` first.

8 kHz PCM_16, the source rate — nothing is resampled on the way out. The only
resampling anywhere is the 16 kHz that TitaNet forces internally.

---

## Things worth knowing

**The shipped `Datasets/rttms/` cannot be filtered as-is.** Those 5602 files are
*mono-mode* MOSS output: across all 203 444 lines the channel field is always
`1` and speakers are `S01`/`S02`/`S03`. Reading field 3 as a channel — which the
original notebook and `sandbox/sep_data_pipeline/filter_1.py` both do — silently
puts every speaker on one channel. The `filter` stage detects a constant channel
field and switches to `energy` mode, recovering the mapping from per-channel
energy instead. Use them with `--set diarize.backend=from_dir`.

**Sortformer runs offline, not streaming.** The notebook copied a 1.04 s-latency
streaming preset from an NVIDIA tutorial. For batch work it is strictly worse —
measured on one 389 s call, both channels:

| | speed | result |
|---|---|---|
| offline | **259x** realtime | ch1 → 1 speaker label |
| streaming | 21x realtime | ch1 → 2 speaker labels |

Twelve times slower *and* noisier: the rotating speaker cache re-identifies the
same voice and splits it, which then makes the filter reject a good call. Over
the corpus that is ~1.9 h versus ~23 h. Full context also holds up on the worst
case — the longest call (1807 s, both channels) takes 12.7 s at 2.5 GiB VRAM.
Set `diarize.options.sortformer.streaming: true` to reproduce the old numbers.

**"One speaker per channel" is measured in speech, not label count.** A diarizer
run on a single telephone channel invents a second label freely. Measured over
30 calls, a second label is a median of **1.5 s** against a main speaker's 40 s —
a breath, a noise burst, faint bleed. Rejecting those like a genuine second
speaker dropped the pass rate to 10%. A minor label now only counts as a second
*person* if it clears both `min_label_speech_sec` (2.0) and `min_label_share`
(0.10). Below that its time goes into `excluded` and is **cut out of the source**
by `embed` and `build` — so if it really was a brief third voice, that voice is
removed rather than mixed into someone's clean channel.

**Purity is mode-dependent.** In `energy` mode purity is what makes the argmax
assignment trustworthy, so `purity_min` is 0.9. In `channel` mode the assignment
came from the demux and cannot be wrong, so `purity_min_channel_mode` is only
0.5. It has to be loose: the measure divides by the *other* channel's energy
during this speaker's turns, so ordinary line noise on a quiet channel drags a
perfectly clean source to 0.72–0.86. Only below 0.5 is something actually broken.

**Natural overlap is ~3%, so most mixtures are boosted.** Measured with Silero
across 25 calls: mean 3.2%, p90 7.0%. Realistic telephony, but a model trained
only on that barely has to separate anything. Since the channels are isolated,
`build` can roll `s2` in time and the mixture stays perfectly labelled.
`natural_frac` (0.6) of calls are left as recorded; the rest are shifted to land
inside `target_overlap`. The shift is found by cross-correlating the two speech
masks, giving the overlap for *every* shift in one FFT.

**The overlap shift wraps, so it is constrained to wrap in silence.** `np.roll` is
circular: rolling `s2` by `k` cuts the original at sample `n - k` and joins original sample
`n - 1` to sample `0`. Left unconstrained, the first of those landed inside an utterance on
**22.4% of shifted calls** — chopping a word and putting its two halves at opposite ends of
the file, which sounds exactly like the audio has been shuffled. `shift_for_target` now
considers only offsets whose seams fall in `build.seam_guard_ms` (200 ms) of silence.
Measured over the 983 shifted calls of a real build: **220 chopped utterances before, 0
after**, with 781 calls still shifted and overlap unchanged (mean 0.264, 99.9% in band). The
202 that decline are calls whose recording starts or ends mid-speech, where no offset is
safe; `meta.json` records `shift_reason` and `stats.json` counts them.

**The mixture is formed before zerofying, not after.** Summing two channels that
had already been zerofied gave a mixture that was digitally silent whenever
nobody was talking — a signal that occurs in no real call, and one a model finds
long before it finds the voices. `build` now sums the channels as recorded and
zerofies only the targets, so the input carries the line noise, the room and the
breath between the turns while `s1`/`s2` hold speech alone. That is the ordinary
"noisy mixture, clean targets" setup: the model is asked to separate *and* clean
up, which is what it will have to do in production anyway.

The cost is the exact identity. `mix - (s1 + s2)` is now precisely the removed
background — on the corpus it sits 20–40 dB under the voices, and `meta.json`
records `background_snr_db` per call. Set `build.zerofy_mix: true` to get the
old behaviour back, where the mixture is the sum of the two zerofied channels
and `mix == s1 + s2` holds sample-for-sample.

**What `verify` checks depends on which of those you built.** It reads the mode
from `stats.json`. Under `zerofy_mix` it asserts the full identity at `1e-4`,
the PCM_16 quantization floor (three independently rounded files, ~1.5 LSB;
observed worst case `3.05e-05`). Otherwise it asserts two things: the identity
still holds exactly wherever both speakers are at full gain, and what the
mixture holds beyond `s1 + s2` does not *look* like `s1 + s2`. The second test
is the one that matters, because at ~3% natural overlap plenty of calls have no
sample where both speakers are active. It works because the leftover should be
the other channel's background, recorded on a physically isolated line: a
correct call measures `|corr| ≈ 0.015`, a mixture swapped in from another call
`0.66`, a mixture scaled 20% against its sources `0.97`. The limit is `0.5`.

Either way, when anything clips all three signals are scaled by the same factor.
Scaling only the mixture would change the level the model has to reproduce.

The peak guard covers **all three signals, not just the mixture**. A source can
exceed full scale while the sum does not: `scale_to_sir` pushes s2 well above 1.0
at a negative SIR, and wherever the two sources partially cancel the mixture
still fits under the ceiling. The PCM_16 write then clips s2 alone. Guarding only
the mixture broke the identity on **22 of 2513 calls**, with residuals to
`1.06e-01` — every one a negative-SIR call whose s2 was pinned at 1.0000 while
the mixture sat exactly at the 0.99 ceiling.

**Zerofying fades its edges.** Hard-zeroing at every VAD boundary leaves a click
correlated perfectly with the label, and a separation model will learn the
clicks instead of the voices. `build.fade_ms` (10 ms) applies a raised-cosine
taper at each edge. The same taper covers the one join a circular `--shuffle`
roll creates inside the file: on a zerofied channel both sides of it were exact
zeros, but on a raw channel they are two unrelated samples of noise floor, and a
step at a fixed offset is its own learnable artifact.

**Spans excluded as a possible third voice leave the mixture too.** They are cut
from the *channel*, not just from the VAD mask. Now that the mixture is not the
sum of the targets, removing such a span from `s1`/`s2` alone would leave an
unlabelled voice in the model's input with no target to match it — worse than
the short gap that cutting it leaves behind.

**Enhancement runs before the sum, and in its own stage.** Speech enhancement is
non-linear, so `enhance(s1 + s2) != enhance(s1) + enhance(s2)` — it has to be applied to
the two isolated channels and the mixture formed afterwards, or the sources stop being
the mixture's speech content. It is a separate stage rather than part of `build` because throughput is fixed at
**~3.2x realtime and does not improve with concurrency** (measured 3.17x / 3.28x / 3.27x at
1 / 4 / 8 workers: one replica, adaptive batch cap 1), while `build` is the stage you re-run
most. Only the VAD speech spans are sent — everything else is zerofied by `build` anyway —
which takes the selected set from ~44 h to **~22 h**. The service returns 8 kHz when
`output_sample_rate` is omitted, and length is preserved exactly, so every VAD offset stays
valid; verified on real audio, the untouched 65% of a call differs by at most half a PCM_16
LSB while speech differs by ~120.

**`mix.wav` comes from the original audio, never the enhanced copy.** An earlier
build read one file for both halves of the pair, so once `enhance` had run the
*mixture* was denoised too — measured, a built `mix.wav` matched the enhanced
`.flac` to the PCM_16 floor (corr 1.000000) and differed from the `.opus` by up to
0.78. Enhancing the model's input throws away the very noise it has to learn to
cope with. `build` now reads both files: the mixture from the `.opus`, the targets
from the cache. Everything done to channel 2 — the roll, its seam taper and the
SIR gain — is applied to both copies, or the mixture and the target stop
describing the same signal. `meta.json` records `mix_source` and `target_source`
separately.

This costs the exact sum: `mix - (s1 + s2)` holds the background *and* whatever the
enhancer removed, which correlates with the voices (mean 0.42, max 0.69 over 12
calls). So `verify` checks this mode differently — where one speaker talks alone,
the mixture and that target must correlate above 0.8 (measured 0.993), and every
file's RMS must match the fingerprint `build` recorded, which is what catches a
rescaled or swapped mixture that correlation alone cannot see.

**Speakers are capped by speech time, not call count.** What a model hears of a voice
is seconds, and one side of one call carries anywhere from 5 s to ~14 min — so a call count
bounds nothing. `select.max_duration_per_speaker` (default 1200 s) caps each voice's own
speech: its VAD speech minus `excluded` spans, i.e. exactly what lands in `s1`/`s2`. Call
wall-clock would overcharge a quiet speaker 2.6–8.3x, and the embedding stage's `speech_sec`
undercounts by 15–39%. Admission is strict — a call is kept only if it leaves both speakers
within budget. Compared at matched size on the real corpus:

| ~size | cap | calls | speakers | loudest voice | top-10 share |
|---|---|---|---|---|---|
| 15 h | calls ≤ 3 | 253 | 312 | 11.8 min | 10.3% |
| 15 h | speech ≤ 5 min | 317 | 372 | 5.0 min | 6.2% |
| 40 h | calls ≤ 15 | 1082 | 853 | 31.5 min | 10.8% |
| 40 h | speech ≤ 20 min | 1328 | 979 | 20.0 min | 7.5% |

`max_calls_per_speaker` is still available as a secondary cap (default off).

**The cap also decides whether the splits work.** Speaker-disjoint splitting can only divide
the speaker graph along its connected components, and speakers who appear in many calls
glue it into one. On the real corpus the largest component holds **99.1% of selected calls
at a 20-minute cap** (splits 910 / 4 / 4) and 95.3% at 15 minutes — but **32.2% at 10
minutes**, where the splits land at 393 / 23 / 22, essentially the 90/5/5 target. It is a
sharp threshold, not a gradual one: cutting the busiest voice from 26 calls to 11 is what
stops the graph percolating. If `dev`/`test` come out with a handful of calls, tighten the
cap before anything else.

**Splits are speaker-disjoint by construction.** Calls are edges between two
global speakers, and whole connected components go to one split. A per-call
random split leaks the same voice into train and test, which is the standard way
a separation benchmark ends up flattering itself.

---

## Services

Two backends need something running. They both default to **port 8000**, so they
cannot both be up:

```bash
# gender stage -- moved to 8001 to avoid the collision.
# BIND must move too; the server binds 8000 inside the container regardless.
docker pull akbarumirzokov/gender-detection:latest
docker run --name gender-detection-api \
  -e BIND=0.0.0.0:8001 -p 8001:8001 \
  akbarumirzokov/gender-detection:latest

# moss_http diarizer -- see sandbox/moss_serving/
docker compose up -d moss-server           # vLLM, port 8000
```

```bash
# enhance stage -- the MossFormerGAN service, port 8000
cd /home/akbar/craft/prod/mossformergan_serve && docker compose up -d
curl -s localhost:8000/healthz
```

MOSS is 0.9 B and wants ~6 GB VRAM, so on a single 8 GB card it cannot share the
GPU with the Sortformer or TitaNet stages. `sortformer` is the practical default.

---

## Adding a model

Five registries — `DIARIZERS`, `VADS`, `EMBEDDERS`, `GENDER`, `CLUSTERERS`.
A new backend is one file plus a decorator; no stage code changes:

```python
# dsd/backends/diarizers/pyannote.py
from ...registry import DIARIZERS

class PyannoteDiarizer:
    needs_audio = True
    def __init__(self, options): ...
    def diarize(self, call, audio_path, data, sr): ...   # -> list[Segment]

@DIARIZERS.register("pyannote")
def _build(options): return PyannoteDiarizer(options)
```

Import it in `dsd/backends/diarizers/__init__.py`, add an options block under
`diarize.options.pyannote` in `configs/default.yaml`, and select it with
`--set diarize.backend=pyannote`. Each family's `base.py` holds the `Protocol`
it must satisfy. Factories must stay lazy — registering must never load weights.

A new **stage** is the same idea: a module exposing `NAME`, `REQUIRES`,
`add_args(parser)` and `run(cfg, args)`, appended to `ORDER` in
`dsd/stages/__init__.py`.

---

## Status

Verified end to end on a 30-call slice: `diarize → filter → vad → embed →
cluster → select → build → verify`, all green, 14 calls and 337 chunks built,
`verify` passing. `tests/smoke.py` is green at 42/42.

Not yet exercised against a live service: the **`gender`** stage (needs the
Docker image re-pulled) and the **`moss_http`** diarizer (needs the server up).
Both have their payload parsing covered by the `http` checks, but neither has
been run against the real thing. A **full-corpus run** has not been done either.

`tutorials/explore.ipynb` is the original notebook this pipeline was derived
from, kept for reference.