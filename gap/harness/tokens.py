"""Protocol `tokens`: teacher-forced next-token predictions of a reference solution, before and after a repair.

Corpora are zero-shot (HumanEval; BigCodeBench-Hard supplies the reference prompts): the prompt is the code prompt of
the corpus, as it is (no chat template), and the reference is its canonical solution. A repair sees one datum; every
datum starts from the original model.

side-effects  -- after each repair, the share of solution tokens predicted correctly on a held-out 20% of the corpus
                 (and exact match and BLEU of the predicted sequences), against the original model.
"""
from __future__ import annotations

import json
import random
from statistics import fmean, median, pstdev
from typing import List, Tuple

import torch
from loguru import logger
from tqdm import tqdm

from gap.utils.numerics import GLOBAL_SEED, stabilize

from .models import DRYRUN_SAMPLES, REFERENCE_PROMPTS, results_dir, run_tag, start

CORPORA = {
    "humaneval": "HumanEval (openai/openai_humaneval, test): function prompt -> canonical solution",
    "bigcodebench": "BigCodeBench-Hard (bigcode/bigcodebench-hard, v0.1.4): code prompt -> canonical solution",
}


def load_corpus(name: str, limit: int = None) -> List[Tuple[str, str]]:
    """(prompt, solution) pairs; the corpora are downloaded at run time."""
    from datasets import load_dataset

    if name == "humaneval":
        dataset, prompt_field = load_dataset("openai/openai_humaneval", split="test"), "prompt"
    elif name == "bigcodebench":
        dataset, prompt_field = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4"), "code_prompt"
    else:
        raise KeyError(f"unknown corpus {name!r}; available: {sorted(CORPORA)}")
    pairs = [(datum[prompt_field], datum["canonical_solution"].replace(" " * 4, "\t")) for datum in dataset]
    return pairs[:limit] if limit else pairs


def as_sample(tok, prompt: str, solution: str) -> dict:
    """The repair pair of a datum. The sequence start token is part of the question, since the methods
    encode a question without adding special tokens."""
    return {"question": (tok.bos_token or "") + prompt, "answer": solution}


@torch.no_grad()
def teacher_forced(model, tok, sample: dict) -> Tuple[List[int], List[int]]:
    """(target ids, predicted ids): the prediction at each position of the answer given the oracle prefix."""
    device = next(model.parameters()).device
    question = tok([sample["question"]], add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    targets = tok(sample["answer"], add_special_tokens=False)["input_ids"]
    if not targets:
        return [], []
    answer = torch.tensor([targets], device=device)
    logits = model(input_ids=torch.cat([question, answer], dim=-1)).logits
    # position i predicts the token at i + 1
    start = question.shape[1] - 1
    predicted = logits[0, start:start + len(targets)].argmax(dim=-1).tolist()
    return targets, predicted


def accuracy(pairs) -> float:
    """Share of correctly predicted tokens over (targets, predicted) pairs."""
    total = sum(len(targets) for targets, _ in pairs)
    correct = sum(t == p for targets, predicted in pairs for t, p in zip(targets, predicted))
    return correct / total if total else 0.0


def _words(token_ids, tok) -> List[str]:
    import re
    return re.findall(r"\w+|[^\w\s]", "".join(tok.decode([t]) for t in token_ids))


def corpus_bleu(pairs, tok) -> float:
    """Corpus BLEU-4 (uniform weights, brevity penalty) of the predicted token sequences against the reference ones."""
    from collections import Counter
    from math import exp, log
    matches, totals, hyp_len, ref_len = [0] * 4, [0] * 4, 0, 0
    for targets, predicted in pairs:
        ref, hyp = _words(targets, tok), _words(predicted, tok)
        hyp_len += len(hyp); ref_len += len(ref)
        for n in range(1, 5):
            h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
            r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
            matches[n - 1] += sum(min(c, r[g]) for g, c in h.items()); totals[n - 1] += sum(h.values())
    if hyp_len == 0 or min(matches) == 0:
        return 0.0
    precision = sum(log(m / t) for m, t in zip(matches, totals)) / 4
    return exp(precision) * (1.0 if hyp_len > ref_len else exp(1 - ref_len / hyp_len))


def exact_match(pairs) -> float:
    return sum(targets == predicted for targets, predicted in pairs) / max(1, len(pairs))


def _statistics(values: List[float]) -> dict:
    if not values:
        return {}
    return {
        "mean": round(fmean(values), 4), "std": round(pstdev(values), 4), "median": round(median(values), 4),
        "min": round(min(values), 4), "max": round(max(values), 4),
    }


def _setup(method_name, model_key, corpus, limit, device, dryrun, overrides):
    stabilize()
    if dryrun:
        limit = limit or DRYRUN_SAMPLES
    model, tok, model_name, method = start(method_name, model_key, overrides, dryrun, device)
    if not dryrun and hasattr(method.hparams, "max_focus_tokens"):
        # repair every failing token of the solution, not only the first ones
        method.hparams.max_focus_tokens = 10 ** 9
    data = [as_sample(tok, prompt, solution) for prompt, solution in load_corpus(corpus, limit)]
    # behaviour to preserve: prompts of the other corpus, so that no evaluated datum is among them
    number = DRYRUN_SAMPLES if dryrun else REFERENCE_PROMPTS
    other = next(name for name in sorted(CORPORA) if name != corpus)
    references = [(tok.bos_token or "") + prompt for prompt, _ in load_corpus(other, number)]
    model = method.prepare(model, tok, reference_prompts=references)
    return model, tok, model_name, method, data, results_dir(model_key, "tokens", corpus, dryrun)


def _save(folder, name: str, payload: dict) -> None:
    with open(folder / name, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"saved {folder / name}")


def side_effects(
    method_name: str, model_key: str, corpus: str, limit: int = None,
    device: str = None, dryrun: bool = False, overrides: dict = None,
) -> dict:
    model, tok, model_name, method, data, folder = _setup(
        method_name, model_key, corpus, limit, device, dryrun, overrides)
    tag = run_tag(method)

    held_out = set(random.Random(GLOBAL_SEED).sample(range(len(data)), int(len(data) * 0.2)))
    eval_data = [data[i] for i in sorted(held_out)]
    work_data = [datum for i, datum in enumerate(data) if i not in held_out]

    def held_out_accuracy():
        pairs = [teacher_forced(model, tok, sample) for sample in eval_data]
        return accuracy(pairs), exact_match(pairs), corpus_bleu(pairs, tok)

    original, original_em, original_bleu = held_out_accuracy()
    changes, em_changes, bleu_rpd = [], [], []
    try:
        for sample in tqdm(work_data, desc=f"{tag} side effects [{corpus}]"):
            state = method.apply(model, tok, sample)
            try:
                accuracy_now, em_now, bleu_now = held_out_accuracy()
                changes.append(accuracy_now - original)
                em_changes.append(em_now - original_em)
                bleu_rpd.append((bleu_now - original_bleu) / original_bleu if original_bleu else 0.0)
            finally:
                method.restore(model, state)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        summary = {
            "method": method.name, "model": model_name, "corpus": corpus,
            "hparams": vars(method.hparams), "held_out": len(eval_data), "repairs": len(changes),
            "held_out_token_accuracy_original": round(original, 4),
            # change of the held-out token accuracy after one repair (closer to zero is better)
            "change": _statistics(changes),
            "repairs_that_lowered_it": sum(change < 0 for change in changes),
            # overfitting risk as in STAR: exact-match change (absolute) and relative BLEU change on the held-out part
            "held_out_exact_match_original": round(original_em, 4), "held_out_bleu_original": round(original_bleu, 4),
            "em_apd": _statistics(em_changes), "bleu_rpd": _statistics(bleu_rpd),
        }
        _save(folder, f"{tag}_side_effects.json", {"summary": summary, "changes": changes})
    logger.success(json.dumps({k: v for k, v in summary.items() if k != "hparams"}, default=str))
    return summary
