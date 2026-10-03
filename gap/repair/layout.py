"""Module-path resolution that does not depend on the model's name.

Hyperparameter files address modules by templates such as `model.layers.{}.mlp.down_proj`. Checkpoints that
wrap the text decoder (e.g. vision-language models, whose decoder sits under `model.language_model`) use a
different prefix. `resolve_hparams_modules` rewrites the templates to the paths that exist in the loaded model.
"""
from __future__ import annotations

from loguru import logger

MODULE_FIELDS = (
    "rewrite_module_tmp", "layer_module_tmp", "mlp_module_tmp", "attn_module_tmp", "ln_f_module", "lm_head_module",
)


def resolve_module_path(module_names: set, template: str) -> str:
    """Return `template` if it names a module of the model, otherwise the same path under the right prefix.

    A template may contain `{}` for the layer index. A path that cannot be resolved is returned unchanged.
    """
    if not template:
        return template
    probe = template.format(0)
    if probe in module_names:
        return template
    # try the path relative to wherever the decoder lives: drop the leading wrapper names one by one
    parts = template.split(".")
    for drop in range(0, len(parts) - 1):
        tail_template = ".".join(parts[drop:])
        tail = tail_template.format(0)
        matches = sorted((name for name in module_names if name == tail or name.endswith("." + tail)), key=len)
        if matches:
            prefix = matches[0][: len(matches[0]) - len(tail)]
            return prefix + tail_template
    return template


def resolve_hparams_modules(model, hparams):
    """Rewrite, in place, the module fields of `hparams` to paths that exist in `model`.

    The layer template decides where the decoder lives (e.g. `model` -> `model.language_model`); the other
    fields follow the same root, so that `model.norm` becomes the decoder's norm and not some other one.
    """
    module_names = {name for name, _ in model.named_modules()}
    old_root = new_root = None
    layer_template = getattr(hparams, "layer_module_tmp", None)
    if isinstance(layer_template, str) and ".layers.{}" in layer_template:
        resolved = resolve_module_path(module_names, layer_template)
        old_root, new_root = layer_template.split(".layers.{}")[0], resolved.split(".layers.{}")[0]

    for field in MODULE_FIELDS:
        template = getattr(hparams, field, None)
        if not isinstance(template, str) or not template or template.format(0) in module_names:
            continue
        resolved = template
        if old_root is not None and new_root != old_root and (template == old_root or template.startswith(old_root + ".")):
            candidate = new_root + template[len(old_root):]
            if candidate.format(0) in module_names:
                resolved = candidate
        if resolved == template:
            resolved = resolve_module_path(module_names, template)
        if resolved != template:
            logger.info(f"{field}: {template} -> {resolved}")
            setattr(hparams, field, resolved)
        else:
            logger.warning(f"{field}: no module matches {template}")
    return hparams


def fit_layer_indices(model, hparams):
    """Keep the layer indices of `hparams` inside the depth of `model` (for models smaller than the defaults assume).

    `layers` (the layers an editing method changes) is cut to the existing ones; `v_loss_layer` is the last layer.
    """
    template = getattr(hparams, "layer_module_tmp", None)
    if not isinstance(template, str) or ".{}" not in template:
        return hparams
    container = model
    for part in template.split(".{}")[0].split("."):
        container = getattr(container, part)
    depth = len(container)
    layers = getattr(hparams, "layers", None)
    if layers is not None:
        kept = [layer for layer in layers if layer < depth] or [max(0, depth // 4)]
        if kept != list(layers):
            logger.info(f"layers: {list(layers)} -> {kept} (the model has {depth} layers)")
            hparams.layers = kept
    if hasattr(hparams, "v_loss_layer") and hparams.v_loss_layer != depth - 1:
        logger.info(f"v_loss_layer: {hparams.v_loss_layer} -> {depth - 1}")
        hparams.v_loss_layer = depth - 1
    return hparams
