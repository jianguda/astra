"""RQ1 -- the gap between where a failure points and where a patch lands.

For every sample the repair of ASTRA is analysed without being applied for good (see `astra.analyse_gap`): how far
the neurons that the failure attribution (contrastive semantics) points at are from the neurons that carry the patch of
the unrestricted solution, how different the patches of different neurons are (tailoring), and how much the
localisation moves once the patch is applied. Results are also split by the number of failing tokens.
"""
from __future__ import annotations

import json
import os
from time import time

import torch
from loguru import logger
from tqdm import tqdm

from gap.repair import build_method, restore_weights
from gap.repair.engine import RepairEngine
from gap.repair.astra import analyse_gap, repair_views
from gap.utils.numerics import stabilize

from .evo import load_task
from .execution import reference_prompts
from .measure import average
from .models import DRYRUN_SAMPLES, load, results_dir


def run(model_key: str, task: str, limit: int = None, dryrun: bool = False, device: str = None) -> dict:
    stabilize()
    if dryrun:
        limit = limit or DRYRUN_SAMPLES
    model, tok, model_name = load(model_key, device)
    samples = load_task(task, model_name, limit)
    method = build_method("ASTRA", model_key)
    if dryrun:
        method.dryrun()
    model = method.prepare(model, tok, reference_prompts=reference_prompts(task, model_name, dryrun))
    hparams = method.hparams
    folder = results_dir(model_key, "rq1", task, dryrun)

    rows, t0 = [], time()
    try:
        for sample in tqdm(samples, desc=f"RQ1 [{task}]"):
            engine = RepairEngine.from_hparams(model, tok, hparams)
            target_ids = tok(sample["answer"], add_special_tokens=False)["input_ids"][: hparams.max_focus_tokens]
            try:
                stats = analyse_gap(engine, hparams, [engine.encode(v) for v in repair_views(sample, hparams)], target_ids)
            finally:
                restore_weights(model, engine.weights_copy)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if stats is not None:
                rows.append({"id": sample["id"], **stats})
    finally:
        bins = {"tokens<=64": lambda r: r["tokens"] <= 64, "64<tokens<=200": lambda r: 64 < r["tokens"] <= 200,
                "tokens>200": lambda r: r["tokens"] > 200}
        summary = {
            "model": model_name, "task": task, "dryrun": dryrun, "samples": len(rows), "time_sec": round(time() - t0, 1),
            "overall": average(rows and [{k: v for k, v in r.items() if k != "id"} for r in rows]),
            "by_tokens": {name: {"n": sum(map(test, rows)), **average([{k: v for k, v in r.items() if k != "id"}
                                                                       for r in rows if test(r)])}
                          for name, test in bins.items() if any(map(test, rows))},
        }
        name = "gap" + (("-s%sof%s" % tuple(os.environ["GAP_SHARD"].split("/"))) if os.environ.get("GAP_SHARD") else "")
        with open(folder / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "samples": rows}, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"saved {folder / (name + '.json')}")
    logger.success(json.dumps({k: summary[k] for k in ("samples", "overall", "by_tokens")}, default=str))
    return summary
