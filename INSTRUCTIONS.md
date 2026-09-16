# How to run the pipeline

Operating guide for `dsd` — every command, what each stage consumes and produces,
and how to run only part of the chain.

`README.md` explains *why* the pipeline does what it does. This file is about
*running* it.

---

## 1. Before anything

### Environment

Everything runs in the existing `nemo` conda env. Nothing needs installing.

```bash
conda activate nemo
cd /home/akbar/craft/prod/dual_separation_dataset
python -m dsd --help
```

If you would rather not activate the env, call its interpreter directly — every
example below works the same way:

```bash
PY=/home/akbar/miniconda3/envs/nemo/bin/python
$PY -m dsd --help
```

### Run from the project root

This matters more than it looks. `--config` defaults to the *relative* path
`configs/default.yaml`, and when that file is not found the CLI **falls back to
built-in defaults instead of failing**, rooting every path at your current
directory. From `/tmp` you get this, silently:

```
"root": "/tmp",
"work_dir": "/tmp/work",
"audio_dirs": ["/tmp/Datasets/audios"]
```

No error, no audio found, no output. So either `cd` to the project root first, or
pass an absolute config path — the root is then derived from it correctly:

```bash
python -m dsd --config /home/akbar/craft/prod/dual_separation_dataset/configs/default.yaml config
```

Check what you are about to run against at any time:

```bash
python -m dsd config          # the fully resolved configuration
python -m dsd backends        # every registered backend
```

### Weights and services

| needed by | what | where |
|---|---|---|
| `diarize` (sortformer) | `diar_streaming_sortformer_4spk-v2.1.nemo` | set in `configs/default.yaml` |
| `embed` (titanet) | `titanet-l.nemo` | set in `configs/default.yaml` |
| `gender` (http_ecapa) | Docker service on **port 8001** | must be started, see §8 |
| `diarize` (moss_http) | vLLM/SGLang server on port 8000 | optional backend, see §8 |
| `enhance` (http_mossformergan) | MossFormerGAN service on **port 8000** | must be started, see §8 |

The default path (`sortformer` + `silero` + `titanet`) needs a GPU but no
network. `gender` and `enhance` each need their service running. Note that
`enhance` and the `moss_http` diarizer both default to port 8000 and cannot
both be up.

> **On `.env`:** the file contains `STREAMING_SORTFORMER_PATH=...`, but **nothing
> in the code reads it** — there is no `dotenv` dependency. It only takes effect
> if you export it yourself (`set -a && source .env && set +a`). Even then it is
> the *lowest* priority: `configs/default.yaml` already sets
> `diarize.options.sortformer.model_path`, and that wins. Precedence is
> `--set` › `configs/default.yaml` › `$STREAMING_SORTFORMER_PATH` › error.

---

## 2. Command anatomy

```
python -m dsd  [global flags]  <subcommand>  [stage flags]
```

**Global flags must come before the subcommand.** This is the single most common
mistake; argparse answers it with a usage error and exit code 2.

```bash
python -m dsd --set build.chunks=true build      # correct
python -m dsd build --set build.chunks=true      # WRONG - argparse error
```

| global flag | meaning |
|---|---|
| `--config PATH` | config file (default `configs/default.yaml`, relative to CWD) |
| `--root PATH` | project root; defaults to the config file's grandparent |
| `--set KEY=VALUE` | override any config key, repeatable. Values are parsed as YAML, so `true` is a bool and `[-2,2]` is a list |

Subcommands: the ten stages, plus `verify`, `run`, `backends`, `config`.

---

## 3. The stages

```
diarize → filter → vad → embed → cluster → gender → select → enhance → augment → build   (then verify)
```

| # | stage | reads | writes |
|---|---|---|---|
| 1 | `diarize` | `Datasets/audios/**` | `work/rttms/<call>.rttm` |
| 2 | `filter` | `work/rttms/`, audio | `work/suitable.json`, `work/unsuitable.json` |
| 3 | `vad` | `work/suitable.json`, audio | `work/vad/<call>.rttm` |
| 4 | `embed` | `work/suitable.json`, `work/vad/` | `work/embeddings.npz`, `work/embed_index.json` |
| 5 | `cluster` | `work/embeddings.npz` | `work/speakers.json` |
| 6 | `gender` | `work/suitable.json`, `work/vad/`, `work/speakers.json` | `work/gender.json` |
| 7 | `select` | `work/suitable.json`, `work/vad/`, `work/speakers.json`, `work/gender.json`\* | `work/selection.json` |
| 8 | `enhance` | `work/selection.json`, `work/vad/`, audio | `work/enhanced/<call>.flac` |
| 9 | `augment` | the impulse-response and noise corpora named in `augment.*` | `work/augment/{irs,rirs}/`, `noise.json`, `index.json` |
| 10 | `build` | `work/selection.json`, `work/vad/`, original audio (mixture) + enhanced or original audio (targets), `work/augment/` | `dataset/**`, `manifest.jsonl` + `<split>.jsonl`, `stats.json` |
| — | `verify` | `dataset/manifest.jsonl` | nothing; exit code only |

\* optional — see §5.

Anything that fails is recorded in `work/<stage>_failed.tsv` and the run
continues; one unreadable call out of 5613 must not cost the other 5612.

---

## 4. Running the whole thing

```bash
# everything, start to finish
python -m dsd run

# a cheap end-to-end trial first (recommended before committing to a full run)
python -m dsd run --stages diarize,filter,vad,embed,cluster,select,augment,build --limit 30 --chunks
python -m dsd verify
```

Budget roughly **2 hours** for `diarize` over the full corpus (5613 calls, both
channels, ~259x realtime on the 4060), plus embedding time.

`enhance` dominates everything else: throughput is **~3.2x realtime and does not
improve with concurrency**, so the 2513 selected calls take roughly **22 hours**.
It prints that estimate before it starts. Every stage is resumable, so it is safe
to interrupt and re-run the same command — and if you want a dataset now, run the
chain without `enhance` and add it later:

```bash
python -m dsd run --stages diarize,filter,vad,embed,cluster,gender,select,build
python -m dsd enhance                       # overnight
python -m dsd build --overwrite --chunks    # cheap; re-reads the cache
```

`--chunks` additionally writes fixed-length training windows under
`dataset/chunks/`. Without it you get full-call mixtures only.

---

## 5. Skipping stages

Three mechanisms, plus one thing worth understanding first.

### What actually enforces the order

**Not the stage list.** Each stage declares a `REQUIRES` tuple, but the CLI does
not check it — you can ask for any stages in any order and it will try. What
stops you is each stage checking for the *artifacts* it needs:

```console
$ python -m dsd run --stages select,build
[run] select -> build
======================================================================
== select
======================================================================
stage 'select' is missing input:
  suitable.json: /home/akbar/.../work/suitable.json
  speakers.json: /home/akbar/.../work/speakers.json
run the earlier stages first, or check paths.work_dir
```

So "skipping" a stage is fine whenever its output already exists on disk from an
earlier run. It fails cleanly, naming the missing file, when it does not.

> **One caveat:** the guard checks that a path *exists*, not that it has content.
> An empty `work/vad/` directory satisfies `embed`'s check, and `embed` will then
> report every call as `no_vad_rttm` and write zero embeddings. If a stage
> produces suspiciously little, check that its inputs are actually populated and
> not merely present.

### (a) Pick an explicit subset — `--stages`

```bash
python -m dsd run --stages vad,embed,cluster
```

Order is taken from your list. Unknown names are rejected up front.

### (b) Start partway and continue — `--from`

```bash
python -m dsd run --from cluster
# [run] cluster -> gender -> select -> enhance -> build
```

Expands to that stage and everything after it. `verify` is deliberately not part
of the chain — run it yourself.

If you pass both, `--stages` wins.

### (c) One stage on its own

```bash
python -m dsd cluster --report
python -m dsd build --chunks
```

This is the form to use when you want a stage's own flags (§6).

### Automatic skipping — how resume works

You rarely need to skip by hand, because every stage already refuses to redo
finished work. The two kinds behave differently:

| stage | granularity | skips when |
|---|---|---|
| `diarize` | per call | `work/rttms/<call>.rttm` exists |
| `vad` | per call | `work/vad/<call>.rttm` exists |
| `enhance` | per call | `work/enhanced/<call>.flac` exists |
| `build` | per call | `dataset/<split>/<call>/meta.json` exists |
| `embed` | whole artifact | `work/embeddings.npz` exists |
| `cluster` | whole artifact | `work/speakers.json` exists |
| `gender` | whole artifact | `work/gender.json` exists |
| `select` | whole artifact | `work/selection.json` exists |
| `filter` | never skips | always recomputes (it is seconds, not hours) |

A whole-artifact stage that skips says so and stops:

```
[cluster] speakers.json exists, nothing to do (use --overwrite to redo)
```

So **re-running `python -m dsd run` after an interruption resumes** rather than
starting over. To force work, pass `--overwrite` to that stage.

### Which stages are genuinely optional

**`gender`, `enhance` and `augment` are the optional stages.** Without `work/gender.json`, `select`
prints a warning and carries on with gender balancing disabled:

```
[select] gender.json not found -- balancing skipped
```

`enhance` is optional in the same sense: `build.use_enhanced` defaults to `auto`, so a
missing cache just means the dataset is built from raw audio. Set it to `always` if you
want `build` to refuse rather than quietly mix enhanced and raw calls in one dataset.

`augment` is optional too, but differently: it needs no service, only the impulse-response
and noise corpora that `augment.ir_dir` / `augment.noise_dir` point at. If you do not have
them, turn the degradation off rather than skipping the stage —

```bash
python -m dsd --set build.augment.variants=0 build --chunks
```

— because `build` refuses to start when it is asked for variants it has no assets for,
rather than quietly writing a dataset with no degradation in it.

That is the right way to run when the Docker services are not up:

```bash
python -m dsd run --stages diarize,filter,vad,embed,cluster,select,augment,build --chunks
```

Everything else is load-bearing. In particular **`vad` cannot be skipped** — its
masks are what `embed`, `gender` and `build` all use to decide what counts as
speech, and the diarizer's turn boundaries are not a substitute.

### Skipping diarization entirely

The corpus ships with 5610 RTTMs in `Datasets/rttms/`. To adopt those instead of
running a diarizer, switch the backend — `from_dir` just copies them in:

```bash
python -m dsd --set diarize.backend=from_dir run --stages diarize,filter
```

`diarize.options.from_dir.rttm_dir` already points at `Datasets/rttms`. Note
those files are *mono-mode* output whose channel field is a constant, so `filter`
will detect that and switch itself to `energy` mode to recover the mapping. The 3
`.opus` files with no matching RTTM are reported as failures, which is expected.

### What to re-run after changing something

Downstream stages skip when their artifact exists, so they need `--overwrite`
too. Read a row as "run these, in this order":

| you changed | re-run |
|---|---|
| audio corpus, or the diarizer/its options | `diarize --overwrite`, then everything after |
| `filter.*` thresholds | `filter`, `vad`, `embed --overwrite`, `cluster --overwrite`, `select --overwrite`, `build --overwrite` |
| `vad.*` parameters | `vad --overwrite`, `embed --overwrite`, `gender --overwrite`, `build --overwrite` |
| `embed.*`, or the embedder | `embed --overwrite`, `cluster --overwrite`, `select --overwrite`, `build --overwrite` |
| `cluster.threshold` | `cluster --overwrite`, `select --overwrite`, `build --overwrite` |
| the gender service or its options | `gender --overwrite`, `select --overwrite`, `build --overwrite` |
| `select.*` caps or splits | `select --overwrite`, `build --overwrite` |
| `enhance.*`, or the enhancement service | `enhance --overwrite`, `build --overwrite` |
| `augment.*` corpora or room settings | `augment --overwrite`, `build --overwrite` |
| `build.augment.*` (incl. `variants`) | `build` — it notices the change itself and rebuilds what it must |
| `build.*` mixing parameters (incl. `seam_guard_ms`) | `build --overwrite` |

Avoid `run --overwrite` unless you mean it: it propagates to *every* stage that
accepts the flag, including `diarize`, and re-diarizes the whole corpus.

### Worked examples

```bash
# Try a stricter speaker-merge threshold without recomputing any embeddings
python -m dsd cluster --overwrite --threshold 0.85 --report
python -m dsd select --overwrite
python -m dsd build  --overwrite --chunks
python -m dsd verify

# Rebuild the mixtures only, with more synthetic overlap
python -m dsd --set build.natural_frac=0.3 --set build.target_overlap='[0.2,0.7]' \
              build --overwrite --chunks

# Cap each speaker at 10 minutes of their own speech and re-split. On this corpus
# 10 min is where dev/test stop collapsing (393/23/22 vs 910/4/4 at 20 min).
python -m dsd select --overwrite --max-duration-per-speaker 600

# Either cap can be switched off with 0; both may be on at once
python -m dsd select --overwrite --max-duration-per-speaker 0 --max-calls-per-speaker 5
python -m dsd build --overwrite --chunks

# Everything except gender, because the Docker service is not running
python -m dsd run --stages diarize,filter,vad,embed,cluster,select,augment,build --chunks

# Resume a full run that was interrupted: identical command, finished work is skipped
python -m dsd run --chunks
```

---

## 6. Stage flags

`run` forwards `--limit`, `--overwrite` and `--chunks` to whichever stages accept
them, and ignores them for the rest. When you invoke a stage directly you also
get its own options.

| stage | `--limit` | `--overwrite` | `--chunks` | its own flags |
|---|:--:|:--:|:--:|---|
| `diarize` | ✓ | ✓ | – | `--backend` `--workers` |
| `filter` | ✓ | – | – | `--channel-mode {auto,channel,energy}` `--workers` |
| `vad` | ✓ | ✓ | – | `--backend` `--all-calls` |
| `embed` | ✓ | ✓ | – | `--backend` |
| `cluster` | – | ✓ | – | `--backend` `--threshold` `--report` |
| `gender` | ✓ | ✓ | – | `--backend` `--base-url` `--workers` |
| `select` | – | ✓ | – | `--max-duration-per-speaker SECONDS` `--max-calls-per-speaker` `--no-balance-gender` |
| `enhance` | ✓ | ✓ | – | `--backend` `--base-url` `--regions {speech,full}` `--workers` |
| `augment` | – | ✓ | – | `--simulated N` `--no-screen` |
| `build` | ✓ | ✓ | ✓ | `--shuffle` / `--no-shuffle` `--variants N` `--no-prune` |
| `verify` | ✓ | – | – | `--sample N` |

> `--limit N` is applied by each stage **independently**, to its own input. With
> `run --limit 30`, `filter` inspects 30 RTTMs while `embed` takes the first 30
> *suitable* calls — different sets. It is a smoke-test tool, not a way to
> process a consistent subset.

---

## 7. Changing settings without editing files

Every key in `configs/default.yaml` can be overridden per run. Values are parsed
as YAML, so types come out right:

```bash
python -m dsd --set build.chunks=true \
              --set select.max_calls_per_speaker=2 \
              --set cluster.threshold=0.85 \
              --set build.sir_db='[-3,3]' \
              run
```

### Keeping experiments apart

Point the outputs somewhere else and the main run is untouched:

```bash
python -m dsd --set paths.work_dir=work_experiment \
              --set paths.dataset_dir=dataset_experiment \
              run --chunks
```

Relative paths resolve against the project root, not your shell's CWD.

---

## 8. Services

Both default to port 8000, so they cannot both run at once.

```bash
# gender stage -- note BIND must move too, or the server still binds 8000 inside
# the container and the published port goes nowhere
docker pull akbarumirzokov/gender-detection:latest
docker run --name gender-detection-api \
  -e BIND=0.0.0.0:8001 -p 8001:8001 \
  akbarumirzokov/gender-detection:latest

python -m dsd gender --base-url http://localhost:8001
```

```bash
# moss_http diarizer (alternative to sortformer) -- see sandbox/moss_serving/
docker compose up -d moss-server                  # vLLM, port 8000
python -m dsd --set diarize.backend=moss_http diarize --workers 8
```

MOSS is 0.9 B and wants ~6 GB of VRAM, so on the 8 GB card it cannot share the
GPU with `sortformer` or `titanet`. Run those stages separately.

---

## 9. Checking the result

```bash
python -m dsd verify              # every call
python -m dsd verify --sample 200 # a random subset, for a large dataset
```

`verify` re-reads the finished dataset off disk and asserts: `mix == s1 + s2`
within 1e-4 (the PCM_16 quantization floor), equal sample counts, no global
speaker present in two splits, every manifest path resolving, the per-speaker cap
respected, and sources that are neither all-zero nor never-zero. It exits 1 and
lists the problems if anything fails.

Useful things to read afterwards:

```bash
# speaker-merge diagnostics
python -m dsd cluster --report

# why calls were rejected (the file is keyed by call, hence `.[]`)
jq -r '.[].reason' work/unsuitable.json | sort | uniq -c

# what selection dropped, and the resulting splits
jq '{dropped, counts, hours}' work/selection.json

# realized numbers next to the targets that produced them
jq 'del(.config)' dataset/stats.json

# per-split manifests, written beside the combined one
wc -l dataset/{train,dev,test}.jsonl dataset/chunks/{train,dev,test}.jsonl
```

### Tests

```bash
python tests/smoke.py              # 36 checks, ~2.5 s, no GPU/network/weights
python tests/smoke.py --real 10    # additionally: 10 real calls through the real models
python tests/smoke.py --only fade  # run checks matching a substring
```

Run the fast suite after changing anything in `dsd/`. It builds a synthetic
corpus and drives all eight stages with stub backends, so it catches broken
cross-stage contracts that `verify` alone would not.

---

## 10. Common problems

| symptom | cause |
|---|---|
| `dsd: error: unrecognized arguments: --set ...` | a global flag placed after the subcommand (§2) |
| No audio found; `work_dir` under a strange path | run from outside the project root without an absolute `--config` (§1) |
| `stage 'X' is missing input:` | an earlier stage has not produced its artifact — the listed path tells you which |
| `[cluster] speakers.json exists, nothing to do` | not an error; add `--overwrite` to redo |
| `no Sortformer weights configured` | neither `model_path` nor `$STREAMING_SORTFORMER_PATH` is set |
| `Sortformer weights not found: ...` | the path is set but wrong |
| A stage produces almost nothing | its input directory exists but is empty; the guard only checks existence (§5) |
| `gender` fails on every call | the Docker service is not up, or is on 8000 instead of 8001 (§8) |
| `[select] gender.json not found` | expected when `gender` was skipped; balancing is disabled |

Failures never abort a run. Check `work/<stage>_failed.tsv` for the per-item
reason.

---

## 11. Fresh start

`work/` is currently empty, so the next run begins at `diarize`. To reset
deliberately:

```bash
rm -rf work dataset          # start completely over
rm -rf work/vad              # redo VAD and everything downstream
rm work/speakers.json        # redo clustering onward
```

Deleting an artifact is equivalent to `--overwrite` for the stage that produces
it, and forces every later stage to be re-run as well.
