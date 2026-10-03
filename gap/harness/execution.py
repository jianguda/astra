"""Protocol `evolve`: repair on one sample -> generate -> execute the tests of the sample -> restore.

Every sample starts from the original model. Metrics: Pass@k, API usage accuracy (AUA: neither the function
signature nor the API-usage check failed) and test-case coverage.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from time import time

import torch
from loguru import logger
from tqdm import tqdm

from gap.utils.numerics import stabilize

from .evo import TASKS, failure_class, judge, load_task, pass_at_k, preflight, test_prompt
from .chat import format_prompt
from .models import (
    DRYRUN_NEW_TOKENS, DRYRUN_SAMPLES, REFERENCE_PROMPTS, generate, results_dir, run_tag, start, top_updates,
)

TEST_WORKERS = 3

CLASSES = (
    "passed", "no_test", "compilation_failed", "test_failed", "timeout", "signature_error", "api_error",
    "extraction_error", "other_error",
)


def summarize(counts: Counter, cases_passed: int, cases_total: int, total: int) -> dict:
    tested = max(0, total - counts["no_test"])
    passed = counts["passed"]
    usage_errors = counts["signature_error"] + counts["api_error"]
    metrics = {f"pass@{k}": round(float(pass_at_k(tested, passed, k)) * 100.0, 2) for k in (1, 2, 5) if tested >= k}
    metrics["AUA"] = round((tested - usage_errors) / tested * 100.0, 2) if tested else 0.0
    metrics["Coverage"] = round(cases_passed / cases_total * 100.0, 2) if cases_total else 0.0
    return {
        "total_samples": total, "tested_samples": tested, "metrics": metrics,
        "test_cases_passed": cases_passed, "test_cases_total": cases_total,
        **{name: counts[name] for name in CLASSES},
    }


def reference_prompts(task: str, model_name: str, dryrun: bool = False):
    """Prompts whose behaviour a repair should preserve: test prompts of the *other* task, so that no
    evaluated sample is among them."""
    number = DRYRUN_SAMPLES if dryrun else REFERENCE_PROMPTS
    other = next(name for name in sorted(TASKS) if name != task)
    shard = os.environ.pop("GAP_SHARD", None)          # the control prompts do not depend on the shard being run
    try:
        prompts = [test_prompt(sample) for sample in load_task(other, model_name, number)]
    finally:
        if shard is not None:
            os.environ["GAP_SHARD"] = shard
    return [format_prompt(model_name, prompt) or prompt for prompt in prompts]


def run(
    method_name: str, model_key: str, task: str, limit: int = None,
    python_bin: str = None,
    record_updates: bool = False, max_new_tokens: int = 384, device: str = None,
    dryrun: bool = False, overrides: dict = None, phrasing: str = "original",
) -> dict:
    preflight(task, python_bin)
    stabilize()
    if dryrun:
        limit, max_new_tokens = limit or DRYRUN_SAMPLES, min(max_new_tokens, DRYRUN_NEW_TOKENS)
    model, tok, model_name, method = start(method_name, model_key, overrides, dryrun, device)
    samples = load_task(task, model_name, limit)
    model = method.prepare(model, tok, reference_prompts=reference_prompts(task, model_name, dryrun))

    tag = run_tag(method) + ("" if phrasing == "original" else f"-{phrasing}")
    folder = results_dir(model_key, "evolve", task, dryrun)
    logger.info(f"{tag} on {task} with {model_name}: {len(samples)} samples")

    counts, results = Counter(), []
    cases_passed = cases_total = 0
    t0 = time()
    # the tests of a sample run in worker threads while the model repairs and generates for the next samples
    pool, pending = ThreadPoolExecutor(max_workers=TEST_WORKERS), []

    def collect(block: bool) -> None:
        nonlocal cases_passed, cases_total
        while pending and (block or pending[0][0].done()):
            future, sample, updates, repair_sec, generate_sec = pending.pop(0)
            result = future.result()
            result["class"] = failure_class(result)
            result["repair_sec"], result["generate_sec"] = round(repair_sec, 2), round(generate_sec, 2)
            if updates is not None:
                result["updates"] = updates
            counts[result["class"]] += 1
            if result["class"] != "no_test":
                cases_passed += result["test_cases_passed"]
                cases_total += result["test_cases_total"]
            results.append({"index": sample.get("index"), "change_type": sample.get("change_type"), **result})
            if len(results) % 10 == 0:
                metrics = summarize(counts, cases_passed, cases_total, len(results))["metrics"]
                logger.info(f"{len(results)}/{len(samples)} passed={counts['passed']} {metrics}")

    try:
        for index, sample in enumerate(tqdm(samples, desc=f"{tag} [{task}]")):
            began = time()
            state = method.apply(model, tok, sample)
            repair_sec = time() - began
            try:
                updates = top_updates(method.update_mass(model, state)) if record_updates else None
                began = time()
                prediction = generate(model, tok, model_name, test_prompt(sample, phrasing=phrasing), max_new_tokens)
                generate_sec = time() - began
            finally:
                method.restore(model, state)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            pending.append((pool.submit(judge, sample, prediction, python_bin=python_bin),
                            sample, updates, repair_sec, generate_sec))
            collect(block=False)
        collect(block=True)
    finally:
        # also after an interruption: keep what has been evaluated
        collect(block=True)
        pool.shutdown()
        summary = {
            "method": method.name,
            "model": model_name, "task": task, "phrasing": phrasing, "protocol": "one repair, one generation, per sample", "dryrun": dryrun,
            "hparams": vars(method.hparams), "completed_samples": len(results), "time_sec": round(time() - t0, 1),
            **summarize(counts, cases_passed, cases_total, len(results)),
        }
        with open(folder / f"{tag}_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        with open(folder / f"{tag}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"saved {folder / (tag + '_summary.json')}")

    logger.success(json.dumps({k: summary[k] for k in ("tested_samples", "metrics", *CLASSES)}, default=str))
    return summary
