"""Write the job list of the experiments of one model: python tools/mkjobs.py MODEL [--blocks main,reg,...] > jobs.txt

A job is one line `NAME<TAB>NEEDS<TAB>COMMAND` for tools/worker.sh. The benchmarks are split into `--shards` interleaved
shards (GAP_SHARD=k/n). Jobs are ordered shard by shard, so that at any moment every method has covered the same
samples. The main comparison runs through all the shards; the analyses, the ablations and the robustness study use the
first `--aux` shards (one third of each benchmark with the defaults).

blocks   main   None, LowRank, STAR, AlphaEdit and ASTRA on PyEvo and RustEvo (executed tests)
         abl    ASTRA with one component changed: sequential solve, margin 1, one view, and two views with eight
                neurons per token
         attr   ASTRA with the neurons chosen by Input x Gradient (ixg) or by the contrastive gradient (path)
         rob    all methods with the test prompt in the `spec` phrasing
         rq1    the gap statistics
         geom   patch geometry of the three attributions (teacher-forced protocol, logs/geom/*.jsonl)
         reg    side effects on HumanEval (AlphaEdit, LowRank, STAR, ASTRA)
"""
import argparse

BLOCKS = ("main", "abl", "attr", "rob", "rq1", "geom", "reg")
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model")
p.add_argument("--shards", type=int, default=12, help="number of shards of a benchmark")
p.add_argument("--main-shards", type=int, default=None, help="shards of the main comparison (default: all)")
p.add_argument("--aux", type=int, default=4, help="first shards used by abl, attr, rob, rq1 and geom")
p.add_argument("--blocks", default=",".join(BLOCKS), help="comma-separated blocks (default: all)")
a = p.parse_args()
M, N = a.model, a.shards
blocks = a.blocks.split(",")
unknown = [b for b in blocks if b not in BLOCKS]
if unknown:
    p.error(f"unknown blocks {unknown}; available: {', '.join(BLOCKS)}")
TASKS = ("pyevo", "rustevo")
H = "python -m gap.harness"
ASTRA = "ASTRA"
METHODS = [ASTRA, "None", "LowRank", "STAR", "AlphaEdit"]
ABLATIONS = ("--set solve=sequential", "--set margin=1", "--set views=1",
             "--set views=2 --set tokens_ratio=8")


def job(name, needs, env, cmd):
    print(f"{name}\t{needs}\t{env} {H} {cmd}")


def needs(method):
    """AlphaEdit jobs wait for the covariance statistics of the model (job prep_alpha), computed once."""
    return "prep_alpha" if method == "AlphaEdit" else "-"


def evolve(tag, shard, task, method, sets="", extra_args=""):
    name = (f"{tag}_{task}_{method}{('_' + sets.replace(' ', '').replace('--set', '').replace('=', '')) if sets else ''}"
            f"{extra_args.replace(' ', '').replace('--', '_')}_s{shard}")
    job(name, needs(method), f"GAP_SHARD={shard}/{N} GAP_TAG={tag}",
        " ".join(f"evolve --model {M} --task {task} --method {method} {sets} {extra_args}".split()))


if any(b in blocks for b in ("main", "rob", "reg")):
    # one AlphaEdit run on a single sample: it computes and caches the null-space projection of the model
    job("prep_alpha", "-", "GAP_TAG=prep", f"evolve --model {M} --task pyevo --method AlphaEdit --limit 1")

for shard in range(N):
    if "main" in blocks and shard < (a.main_shards or N):
        for method in METHODS:
            for task in TASKS:
                evolve("main", shard, task, method)
    if shard != a.aux - 1:
        continue
    # analyses and ablations on the first shards
    for s in range(a.aux):
        for task in TASKS:
            if "abl" in blocks:
                for sets in ABLATIONS:
                    evolve("abl", s, task, ASTRA, sets)
            if "attr" in blocks:
                for attr in ("ixg", "path"):
                    evolve("attr", s, task, ASTRA, f"--set attribution={attr}")
            if "rob" in blocks:
                for method in METHODS:
                    evolve("rob", s, task, method, extra_args="--phrasing spec")
            if "rq1" in blocks:
                job(f"rq1_{task}_s{s}", "-", f"GAP_SHARD={s}/{N} GAP_TAG=rq1", f"rq1 --model {M} --task {task}")
            if "geom" in blocks:
                for attr in ("direct", "ixg", "path"):
                    job(f"geom_{task}_{attr}_s{s}", "-",
                        f"GAP_SHARD={s}/{N} GAP_TAG=geom GAP_GEOMETRY_LOG=logs/geom/{M}_{task}_{attr}_s{s}.jsonl",
                        f"proxy --model {M} --task {task} --method {ASTRA} --set attribution={attr}")
    if "reg" in blocks:
        for method in ("AlphaEdit", "LowRank", "STAR", ASTRA):
            job(f"reg_{method}", needs(method), "GAP_TAG=reg",
                f"tokens --model {M} --corpus humaneval --measure side-effects --method {method}")
