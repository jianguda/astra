"""`python -m gap.harness <command> ...`

evolve, tokens   the two evaluation protocols, for any method;
proxy            a fast teacher-forced stand-in for evolve (used for the patch-geometry logs);
rq1              the gap statistics.
Every command takes `--dryrun`: a few samples and small budgets through the same code path, with results kept
apart (`outputs/<model>/harness_<command>_dryrun/`).
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    from gap.repair import METHODS
    from gap.utils.models import MODEL_REGISTRY

    from .evo import TASKS
    from .tokens import CORPORA

    parser = argparse.ArgumentParser(prog="python -m gap.harness", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def shared(sub):
        sub.add_argument("--model", default="oxcoder-9b",
                         help="a registry key (" + ", ".join(sorted(MODEL_REGISTRY)) + "), a Hub id or a local path")
        sub.add_argument("--limit", type=int, default=None, help="N samples (code-evolution tasks: spread evenly over the selected samples)")
        sub.add_argument("--device", default=None)
        sub.add_argument("--dryrun", action="store_true", help="4 samples, small budgets, separate output folder")

    def method(sub):
        sub.add_argument("--method", required=True, choices=sorted(METHODS))
        sub.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", dest="overrides",
                         help="override a hyperparameter of the method (repeatable), e.g. --set neuron_num=64")
        sub.add_argument("--record_updates", action="store_true",
                         help="store, per sample, where the repair changed the model (top neurons by update norm)")

    evolve = commands.add_parser("evolve", help="code-evolution tasks, judged by executing tests")
    shared(evolve)
    method(evolve)
    evolve.add_argument("--task", required=True, choices=sorted(TASKS))
    evolve.add_argument("--phrasing", default="original", choices=["original", "spec"],
                        help="wording of the test prompt; 'spec' is a wording that no repair view uses")
    evolve.add_argument("--max_new_tokens", type=int, default=384)
    evolve.add_argument("--test_python", default=None, help="interpreter that runs the PyEvo tests")

    proxy = commands.add_parser("proxy", help="fast teacher-forced stand-in for evolve (no generation, no tests)")
    shared(proxy)
    method(proxy)
    proxy.add_argument("--task", required=True, choices=sorted(TASKS))

    tokens = commands.add_parser("tokens", help="zero-shot corpora, teacher-forced token predictions")
    shared(tokens)
    method(tokens)
    tokens.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    tokens.add_argument("--measure", default="side-effects", choices=["side-effects"])

    rq1 = commands.add_parser("rq1", help="gap statistics: where a failure points and where the patch lands")
    shared(rq1)
    rq1.add_argument("--task", required=True, choices=sorted(TASKS))

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "rq1":
        from .rq1 import run
        run(args.model, args.task, limit=args.limit, dryrun=args.dryrun, device=args.device)
        return

    from .models import parse_overrides
    common = dict(limit=args.limit, device=args.device, dryrun=args.dryrun,
                  overrides=parse_overrides(args.overrides))
    if args.command == "evolve":
        from .execution import run
        run(args.method, args.model, args.task, python_bin=args.test_python, record_updates=args.record_updates,
            max_new_tokens=args.max_new_tokens, phrasing=args.phrasing, **common)
        return
    if args.command == "proxy":
        from .proxy import run
        run(args.method, args.model, args.task, record_updates=args.record_updates, **common)
        return
    from . import tokens
    tokens.side_effects(args.method, args.model, args.corpus, **common)
