"""Protocol `proxy`: a fast, teacher-forced stand-in for `evolve` (no generation, no tests).

A repair sees the rephrased task and the reference code (as in `evolve`); the judge of `evolve` later asks the
*original* task without API knowledge. Here the reference code is scored on that original prompt, token by token,
before and after the repair, without generating and without running tests:

  acc            share of answer tokens predicted correctly on the test prompt (given the oracle prefix)
  know           tokens that the model gets right when the API knowledge is in the prompt and wrong without it: the
                 knowledge a repair has to supply; `know_fixed` is the share of them that is right after the repair
  nll            mean negative log-likelihood of the answer on the test prompt
  kl             mean KL(original || repaired) of the next-token distribution on reference prompts of the other task
"""
from __future__ import annotations

import json
from collections import defaultdict
from statistics import fmean
from time import time

import torch
from loguru import logger
from tqdm import tqdm

from gap.utils.numerics import stabilize

from .chat import format_prompt
from .evo import load_task, test_prompt
from .execution import reference_prompts
from .models import DRYRUN_SAMPLES, results_dir, run_tag, start, top_updates


@torch.no_grad()
def forced(model, tok, prompt: str, answer_ids: list):
    """(top-1 correct [n], nll [n]) of the answer tokens after `prompt` (oracle prefix)."""
    device = next(model.parameters()).device
    question = tok([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    answer = torch.tensor([answer_ids], device=device)
    logits = model(input_ids=torch.cat([question, answer], dim=-1)).logits[0, question.shape[1] - 1:-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(1, answer[0][:, None])[:, 0]
    return (logits.argmax(-1) == answer[0]).cpu(), nll.cpu()


@torch.no_grad()
def next_dists(model, tok, prompts: list):
    device = next(model.parameters()).device
    out = []
    for prompt in prompts:
        ids = tok([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        out.append(torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1).cpu())
    return torch.stack(out)


def run(method_name: str, model_key: str, task: str, limit: int = None,
        record_updates: bool = False, device: str = None, dryrun: bool = False, overrides: dict = None) -> dict:
    stabilize()
    if dryrun:
        limit = limit or DRYRUN_SAMPLES
    model, tok, model_name, method = start(method_name, model_key, overrides, dryrun, device)
    samples = load_task(task, model_name, limit)
    references = reference_prompts(task, model_name, dryrun)
    model = method.prepare(model, tok, reference_prompts=references)
    tag = run_tag(method)
    folder = results_dir(model_key, "proxy", task, dryrun)
    probe = references[:16]
    base_dist = next_dists(model, tok, probe)

    rows, t0 = [], time()
    try:
        for sample in tqdm(samples, desc=f"{tag} [{task} proxy]"):
            answer_ids = tok(sample["answer"], add_special_tokens=False)["input_ids"][:512]
            plain = format_prompt(model_name, test_prompt(sample)) or test_prompt(sample)
            known = format_prompt(model_name, test_prompt(sample, True)) or test_prompt(sample, True)
            ok0, nll0 = forced(model, tok, plain, answer_ids)
            okk, _ = forced(model, tok, known, answer_ids)
            ok_fit0, _ = forced(model, tok, sample["question"], answer_ids)
            state = method.apply(model, tok, sample)
            try:
                ok1, nll1 = forced(model, tok, plain, answer_ids)
                ok_fit1, _ = forced(model, tok, sample["question"], answer_ids)
                kl = float((base_dist.exp() * (base_dist - next_dists(model, tok, probe))).sum(-1).mean())
                updates = top_updates(method.update_mass(model, state)) if record_updates else None
            finally:
                method.restore(model, state)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            know = okk & ~ok0

            def run_length(ok):          # share of the answer before the first wrong token: how far greedy decoding follows it
                wrong = (~ok[1:]).nonzero()          # the first token is often a markdown fence on the test prompt
                return 1.0 if wrong.numel() == 0 else float(wrong[0, 0] + 1) / len(ok)

            row = {
                "id": sample["id"], "index": sample.get("index"), "change_type": sample.get("change_type"), "tokens": len(answer_ids),
                "acc0": float(ok0.float().mean()), "acc1": float(ok1.float().mean()),
                "nll0": float(nll0.mean()), "nll1": float(nll1.mean()),
                "fit0": float(ok_fit0.float().mean()), "fit1": float(ok_fit1.float().mean()),
                "know": int(know.sum()), "know_fixed": int((know & ok1).sum()),
                "wrong0": int((~ok0).sum()), "fixed": int((~ok0 & ok1).sum()), "broken": int((ok0 & ~ok1).sum()),
                "kl": kl, "run0": run_length(ok0), "run1": run_length(ok1), "exact0": float(bool(ok0[1:].all())),
                "exact1": float(bool(ok1[1:].all())),
            }
            if updates is not None:
                row["updates"] = updates
            rows.append(row)
    finally:
        summary = summarize(rows)
        summary.update({"method": method.name, "model": model_name, "task": task,
                        "hparams": vars(method.hparams), "time_sec": round(time() - t0, 1)})
        with open(folder / f"{tag}_results.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
        with open(folder / f"{tag}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"saved {folder / (tag + '_summary.json')}")
    logger.success(json.dumps({k: v for k, v in summary.items() if k not in ("hparams", "by_type")}, default=str))
    return summary


def summarize(rows: list) -> dict:
    if not rows:
        return {"n": 0}
    m = lambda key, rs=rows: round(fmean(r[key] for r in rs), 4)
    know, kfix = sum(r["know"] for r in rows), sum(r["know_fixed"] for r in rows)
    wrong, fixed = sum(r["wrong0"] for r in rows), sum(r["fixed"] for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r["change_type"]].append(r)
    return {
        "n": len(rows), "acc0": m("acc0"), "acc1": m("acc1"), "d_acc": round(m("acc1") - m("acc0"), 4),
        "nll0": m("nll0"), "nll1": m("nll1"), "fit0": m("fit0"), "fit1": m("fit1"),
        "know": know, "know_fixed": kfix, "know_fixed_rate": round(kfix / know, 4) if know else None,
        "wrong0": wrong, "fixed": fixed, "broken": sum(r["broken"] for r in rows),
        "run0": m("run0"), "run1": m("run1"), "exact0": m("exact0"), "exact1": m("exact1"), "kl": m("kl"), "kl_median": round(sorted(r["kl"] for r in rows)[len(rows) // 2], 4),
        "by_type": {k: {"n": len(v), "d_acc": round(m("acc1", v) - m("acc0", v), 4)} for k, v in by.items()},
    }
