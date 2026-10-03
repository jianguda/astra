"""Prompt formats of the supported chat models, in one place.

A method is trained on `format_prompt(question) + answer + answer_suffix`, and evaluated by generating after
`format_prompt(question)` until one of `stop_strings`.
"""
from typing import List, Optional


SYSTEM_OPENCODER = "You are OpenCoder, created by OpenCoder Team."


# tokenizers of loaded models without a hand-written format below, by checkpoint name
_TOKENIZERS = {}


def register_tokenizer(model_name: str, tok) -> None:
    """Let a model outside the paper suite be prompted through the chat template of its own tokenizer."""
    _TOKENIZERS[model_name] = tok


def _template_tokenizer(model_name: str):
    tok = _TOKENIZERS.get(model_name)
    return tok if tok is not None and getattr(tok, "chat_template", None) else None


def chat_family(model_name: str) -> Optional[str]:
    name = (model_name or "").lower()
    if "oxcoder" in name:
        return "oxcoder"
    if "opencoder" in name:
        return "opencoder"
    return None


def format_prompt(model_name: str, question: str) -> Optional[str]:
    """The user turn followed by the assistant header; None if the model has no known chat format."""
    family = chat_family(model_name)
    if family == "oxcoder":
        # ChatML with reasoning switched off: the model's chat template closes an empty <think> block
        # when `enable_thinking` is false, so that the answer starts right after the prompt
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if family == "opencoder":
        # ChatML with the default system turn of the model's chat template
        return (f"<|im_start|>system\n{SYSTEM_OPENCODER}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")
    tok = _template_tokenizer(model_name)
    if tok is not None:
        # reasoning is switched off, as for OxCoder: the answer starts right after the prompt
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(
                [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    return None


def answer_suffix(model_name: str) -> str:
    """The end-of-turn token appended to a reference answer."""
    if chat_family(model_name) is not None:
        return "<|im_end|>"
    tok = _template_tokenizer(model_name)
    return _end_of_turn(tok) if tok is not None else ""


def _end_of_turn(tok) -> str:
    """The token that ends an assistant turn in the chat template of `tok` (ChatML models: `<|im_end|>`)."""
    if "<|im_end|>" in tok.get_vocab():
        return "<|im_end|>"
    return tok.eos_token or ""


def stop_strings(model_name: str) -> List[str]:
    """Tokens that end a generated answer; empty means the tokenizer's own EOS."""
    if chat_family(model_name) == "oxcoder":
        return ["<|im_end|>", "<|endoftext|>"]
    if chat_family(model_name) == "opencoder":
        return ["<|im_end|>"]
    tok = _template_tokenizer(model_name)
    if tok is not None:
        return list(dict.fromkeys([_end_of_turn(tok), tok.eos_token or ""]))
    return []
