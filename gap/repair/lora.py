"""LoRA: the optimization reference that does not touch the weights of the model.

Low-rank adapters are attached once (`attach`); every sample starts from freshly initialised adapters and
takes `hparams.steps` AdamW steps on the language-modeling loss of prompt + answer.
"""
from __future__ import annotations

import torch
from torch.optim import AdamW

from .hparams import LoRAHyperParams


def attach(model, hparams: LoRAHyperParams):
    """Wrap the frozen model with LoRA adapters; returns the wrapped model."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=hparams.r,
        lora_alpha=hparams.alpha,
        lora_dropout=hparams.dropout,
        target_modules=list(hparams.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def reset(model) -> None:
    """Back to the initial adapters: A re-drawn, B zero, i.e. the adapted model equals the base model."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name:
                torch.nn.init.kaiming_uniform_(param, a=5 ** 0.5)
            elif "lora_" in name:
                param.zero_()


def update(model, tok, hparams: LoRAHyperParams, sample: dict) -> None:
    """`hparams.steps` AdamW steps on the language-modeling loss of prompt + answer."""
    device = next(model.parameters()).device
    enc = tok([sample["question"] + sample["answer"]], return_tensors="pt", truncation=True,
              max_length=hparams.max_length, add_special_tokens=False)
    input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=hparams.lr, weight_decay=0.0)
    model.train()
    for _ in range(max(1, hparams.steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()


@torch.no_grad()
def weight_deltas(model) -> dict:
    """Effective update `scaling * B @ A` of every adapted module: {module name: tensor like its weight}."""
    deltas = {}
    for name, module in model.named_modules():
        lora_a, lora_b = getattr(module, "lora_A", None), getattr(module, "lora_B", None)
        if not lora_a or not lora_b:
            continue
        for adapter in lora_a:
            scaling = module.scaling[adapter]
            deltas[name] = (lora_b[adapter].weight.float() @ lora_a[adapter].weight.float()) * scaling
    return deltas
