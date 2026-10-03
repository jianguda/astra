# ASTRA: analytic repair of code language models

Code and benchmarks of the paper on **ASTRA** (*Analytic Semantic Tailoring for Repair by Attribution*). Given one
repair pair (a prompt with the new API information and a reference answer), ASTRA

1. locates the failing tokens of the reference answer under teacher forcing,
2. targets carrier neurons, selected once by contrastive semantics,
3. tailors a joint patch for all failing tokens by solving one small linear system in closed form,
4. re-linearizes the repaired model for a few rounds.

Only columns of the feed-forward output projections (`mlp.down_proj`) change. No optimizer is used.

## Layout

| Path | Content |
| --- | --- |
| `gap/repair/astra.py` | ASTRA |
| `gap/repair/star.py`, `alphaedit/`, `lora.py` | baselines STAR, AlphaEdit, LowRank |
| `gap/repair/hparams/<method>/<model>.json` | hyperparameters (`default.json` for a model without its own file) |
| `gap/harness/` | evaluation: `evolve` (repair, generate, run tests), `proxy` (teacher-forced, patch geometry), `tokens` (side effects on HumanEval), `rq1` (gap statistics) |
| `data/PyEvo`, `data/RustEvo` | the two benchmarks (622 and 708 tasks) |
| `tools/mkjobs.py`, `tools/worker.sh` | job list of all experiments of one model, and a worker that runs it on one GPU |
| `tools/rust_setup.sh` | Rust toolchains for RustEvo |

## Setup

Python 3.12 or newer, PyTorch with CUDA, one GPU (the models run in bfloat16).

```bash
pip install -e .
bash tools/rust_setup.sh        # Rust toolchains 1.71.0 to 1.91.0 and nightly, needed for RustEvo
```

Models are downloaded from the Hugging Face hub on first use: `minicpm5-1b` (`openbmb/MiniCPM5-1B`),
`opencoder-8b-instruct` (`infly/OpenCoder-8B-Instruct`), `oxcoder-9b` (`OrionLLM/OxCoder-9B`). `--model` also accepts
a hub id or a local path. The PyEvo tests need `pytest`, `pytest-asyncio` and the third-party libraries that the
tasks import. HumanEval and WikiText (for AlphaEdit) are downloaded at run time.

## Usage

```bash
H="python -m gap.harness"; M=oxcoder-9b; T=pyevo
$H evolve --model $M --task $T --method ASTRA --dryrun            # 4 samples
$H evolve --model $M --task $T --method ASTRA                     # methods: ASTRA, STAR, AlphaEdit, LowRank, None
GAP_SHARD=0/12 $H evolve --model $M --task $T --method ASTRA      # one of twelve interleaved shards
```

`--set NAME=VALUE` overrides a hyperparameter. Results are written to
`outputs/<model>/harness_<command>_<GAP_TAG>/<task>/` as `<run>_results.json` (one record per sample) and
`<run>_summary.json` (Pass@1, outcome counts, hyperparameters).

## Experiments of the paper

```bash
python tools/mkjobs.py oxcoder-9b > jobs.txt      # all experiments of one model
bash tools/worker.sh jobs.txt 0                   # JOBS GPU; several workers can share one job list
```

The main comparison uses all twelve shards, the other experiments the first four (one third of each benchmark).

| Experiment | Command (per shard and task) | `GAP_TAG` |
| --- | --- | --- |
| main comparison | `evolve --method {None,AlphaEdit,STAR,LowRank,ASTRA}` | `main` |
| ablation | `evolve --method ASTRA` with `--set solve=sequential`, `--set margin=1`, `--set views=1`, `--set views=2 --set tokens_ratio=8` | `abl` |
| attributions | `evolve --method ASTRA --set attribution={ixg,path}` | `attr` |
| patch geometry | `GAP_GEOMETRY_LOG=<file> proxy --method ASTRA --set attribution={direct,ixg,path}` | `geom` |
| gap statistics | `rq1` | `rq1` |
| robustness | `evolve --method <method> --phrasing spec` | `rob` |
| side effects | `tokens --corpus humaneval --measure side-effects --method <method>` | `reg` |

The attributions are values of `attribution`: `direct` is contrastive semantics (default), `path` the contrastive
gradient, `ixg` Input x Gradient.

## Configuration of ASTRA

`gap/repair/hparams/ASTRA/*.json`, the same for all models and both benchmarks.

| Name | Value | Meaning |
| --- | --- | --- |
| `views` | 4 | renderings of the repair prompt that are fitted together |
| `margin` | 5 | lead in logits of the target token over the best other token |
| `tokens_ratio` | 16 | neuron budget per answer token (at least 16 neurons and at most 4 percent of the width per layer) |
| `layer_ratio` | 0.25 | share of the layers that are repaired |
| `max_rounds` | 10 | maximum number of re-linearization rounds |
| `solve` | `joint` | one linear system for all failing tokens (`sequential`: one token at a time) |
| `null_space` | `diagonal` | weights from the mean squared activation on reference prompts |

Baselines use their default settings: AlphaEdit edits layers 4 to 8, STAR repairs a quarter of the layers for 8
epochs, LowRank trains rank-8 adapters on `down_proj` for 10 steps.

Environment variables: `GAP_SHARD=k/n`, `GAP_TAG`, `GAP_SPLIT=dev|test` (every eighth sample or the rest),
`GAP_OUTPUT_ROOT`, `GAP_CACHE_ROOT`, `HF_HOME`.
