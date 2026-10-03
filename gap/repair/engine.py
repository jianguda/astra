from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch
from loguru import logger

def _get_module(model, name: str):
    module = model
    for attr in name.split("."):
        module = module[int(attr)] if attr.isdigit() else getattr(module, attr)
    return module


class RepairEngine:
    """A thin, model-agnostic view of a causal LM for neuron-level repair.

    The model is described by the module templates of the hyperparameter file, e.g.
    `rewrite_module_tmp = "model.layers.{}.mlp.down_proj"`. A *neuron* is `(layer, position)`,
    and its write vector is column `position` of that layer's FFN output weight ([hidden, neurons]).
    """

    def __init__(self, model, tok, rewrite_module_tmp: str, layer_module_tmp: str, lm_head_module: str = "lm_head"):
        self.model = model
        self.tok = tok
        self.rewrite_module_tmp = rewrite_module_tmp
        self.mlp_module_tmp = rewrite_module_tmp.rsplit(".", 1)[0]
        self.layer_module_tmp = layer_module_tmp
        self.lm_head_module = lm_head_module
        self.n_layers = len(_get_module(model, layer_module_tmp.split(".{}")[0]))
        self.device = next(model.parameters()).device

        # original weights of every parameter this repair touched: the value `apply_*_to_model` returns
        self.weights_copy: Dict[str, torch.Tensor] = {}
        self._grad_flags: Optional[Dict[str, bool]] = None
        self.report = {"points": [], "skipped": 0, "layers": set(), "neurons": []}

    @classmethod
    def from_hparams(cls, model, tok, hparams) -> "RepairEngine":
        from .layout import resolve_hparams_modules
        resolve_hparams_modules(model, hparams)
        return cls(model, tok, hparams.rewrite_module_tmp, hparams.layer_module_tmp, hparams.lm_head_module)

    # ------------------------------------------------------------------ modules and weights
    def ff_out(self, layer: int):
        return _get_module(self.model, self.rewrite_module_tmp.format(layer))

    def mlp(self, layer: int):
        return _get_module(self.model, self.mlp_module_tmp.format(layer))

    def weight_name(self, layer: int) -> str:
        return self.rewrite_module_tmp.format(layer) + ".weight"

    def weight(self, layer: int) -> torch.Tensor:
        return self.ff_out(layer).weight

    def backup(self, layer: int) -> None:
        """Keep the original weight of a layer before its first change (lazy: untouched layers are not copied)."""
        name = self.weight_name(layer)
        if name not in self.weights_copy:
            self.weights_copy[name] = self.weight(layer).detach().clone()
            self.report["layers"].add(layer)

    def set_trainable(self, layers: List[int]) -> List[torch.nn.Parameter]:
        """Make exactly the FFN output weights of `layers` trainable; `finish` restores the previous flags."""
        if self._grad_flags is None:
            self._grad_flags = {name: p.requires_grad for name, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad = False
        params = []
        for layer in layers:
            self.backup(layer)
            p = self.weight(layer)
            p.requires_grad = True
            params.append(p)
        return params

    @torch.no_grad()
    def rollback(self) -> None:
        """Undo every change of this repair (used when a repair fails half-way)."""
        logger.warning("repair failed half-way; restoring the original weights")
        for name, original in self.weights_copy.items():
            _get_module(self.model, name.rsplit(".", 1)[0]).weight[...] = original

    def finish(self) -> Dict[str, torch.Tensor]:
        """Leave the model as the harness expects it and return the weights needed to undo the repair."""
        if self._grad_flags is not None:
            for name, p in self.model.named_parameters():
                p.requires_grad = self._grad_flags[name]
            self._grad_flags = None
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        logger.info(
            f"repaired {len(self.report['points'])} failing token(s), skipped {self.report['skipped']}, "
            f"changed layers {sorted(self.report['layers'])}"
        )
        return self.weights_copy

    # ------------------------------------------------------------------ semantics
    def _output_gram_factor(self) -> torch.Tensor:
        """Cholesky factor of the Gram of the decoder atoms, G = W_g^T W_g (W_g: LM-head rows after the final-norm
        gain): the metric under which the lift of a contrast changes the logits of all other tokens least. It is
        computed once per model; neither the LM head nor the final norm is changed by a repair."""
        factor = getattr(self.model, "_output_gram_factor", None)
        if factor is None:
            self.contrast(0, 0)                                     # resolves the final-norm gain
            head = _get_module(self.model, self.lm_head_module).weight
            gain = self.model._decoder_gain[1].to(head.device)
            gram = torch.zeros(head.shape[1], head.shape[1], device=head.device)
            for chunk in head.split(8192):
                atoms = chunk.detach().float() * gain
                gram += atoms.T @ atoms
            gram += 1e-6 * gram.diagonal().mean() * torch.eye(gram.shape[0], device=gram.device)
            factor = self.model._output_gram_factor = torch.linalg.cholesky(gram)
        return factor

    def semantic_delta(self, target_label: int, argmax_label: int) -> torch.Tensor:
        """Direction in hidden space from the produced token to the expected token ([hidden], on device): the lift
        G^{-1} c of the contrast covector c, which raises the contrast while moving every other logit least."""
        factor = self._output_gram_factor()
        contrast = self.contrast(target_label, argmax_label).to(factor.device)
        return torch.cholesky_solve(contrast[:, None], factor)[:, 0].to(self.device)

    def contrasts(self, targets: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
        """Contrast covectors of many (target, produced) pairs at once ([pairs, hidden], on device)."""
        self.contrast(0, 0)
        head = _get_module(self.model, self.lm_head_module).weight
        gain = self.model._decoder_gain[1].to(head.device)
        return ((head[targets].detach().float() - head[others].detach().float()) * gain).to(self.device)

    def semantic_deltas(self, targets: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
        """The lifts of many contrasts with a single triangular solve ([pairs, hidden])."""
        factor = self._output_gram_factor()
        contrast = self.contrasts(targets, others).to(factor.device)
        return torch.cholesky_solve(contrast.T.contiguous(), factor).T.to(self.device)

    def contrast(self, target_label: int, argmax_label: int) -> torch.Tensor:
        """The covector that reads the logit contrast of the two tokens off a normalised final State ([hidden]).

        It is the difference of the two decoder atoms (LM-head rows after the final-norm gain); the common-logit
        centring of the atoms cancels in a difference. Pairing it with a State vector needs no metric.
        """
        frames = getattr(self.model, "_decoder_gain", None)
        if frames is None:
            from gap.utils.runtime import discover_model_layout, fixed_norm_gain

            layout = discover_model_layout(self.model)
            frames = (layout.final_norm, fixed_norm_gain(layout.final_norm).float())
            self.model._decoder_gain = frames
        head = _get_module(self.model, self.lm_head_module).weight
        rows = head[[target_label, argmax_label]].detach().float()
        return ((rows[0] - rows[1]) * frames[1].to(rows.device)).to(self.device)

    @torch.no_grad()
    def observe(self, input_ids: torch.Tensor, layers: List[int]):
        """One forward pass at the last position: (keys, scale, logits).

        keys[layer]  -- the neuron activations that the FFN output projection of `layer` receives ([neurons]),
                        i.e. the coefficients of its write vectors (for a gated FFN: act_fn(gate) * up);
        scale        -- the factor by which the final normalisation divides the residual State (its RMS);
        logits       -- next-token logits ([vocab], float32).
        A vector `v` added to the residual stream at the last position changes the logit contrast of two
        tokens by `contrast @ v / scale`, up to the change of `scale` and of later layers.
        """
        self.contrast(0, 0)                                         # resolves the final normalisation
        final_norm = self.model._decoder_gain[0]
        keys, state = {}, {}

        def key_hook(layer):
            def hook(_module, inputs):
                keys[layer] = inputs[0][0, -1].detach().float()
            return hook

        def norm_hook(_module, inputs):
            state["scale"] = inputs[0][0, -1].detach().float().square().mean().sqrt()

        handles = [self.ff_out(layer).register_forward_pre_hook(key_hook(layer)) for layer in layers]
        handles.append(final_norm.register_forward_pre_hook(norm_hook))
        try:
            logits = self.model(input_ids=input_ids).logits[0, -1].float()
        finally:
            for handle in handles:
                handle.remove()
        return keys, state["scale"], logits

    @torch.no_grad()
    def observe_answer(self, source_ids: torch.Tensor, answer_ids: torch.Tensor, layers: List[int]):
        """One teacher-forced pass over `source + answer`: (keys, scale, logits) at the T positions that predict the
        answer tokens. keys[layer] is [T, neurons], scale [T], logits [T, vocab] (float32)."""
        self.contrast(0, 0)
        final_norm = self.model._decoder_gain[0]
        start, count = source_ids.shape[1] - 1, answer_ids.shape[-1]
        keys, state = {}, {}

        def key_hook(layer):
            def hook(_module, inputs):
                keys[layer] = inputs[0][0, start:start + count].detach().float()
            return hook

        def norm_hook(_module, inputs):
            state["scale"] = inputs[0][0, start:start + count].detach().float().square().mean(-1).sqrt()

        handles = [self.ff_out(layer).register_forward_pre_hook(key_hook(layer)) for layer in layers]
        handles.append(final_norm.register_forward_pre_hook(norm_hook))
        try:
            ids = torch.cat([source_ids, answer_ids.reshape(1, -1).to(source_ids.dtype)], dim=-1)
            try:
                # the vocabulary projection is only needed at the positions that predict the answer
                logits = self.model(input_ids=ids, logits_to_keep=count + 1).logits[0, :count].float()
            except TypeError:
                logits = self.model(input_ids=ids).logits[0, start:start + count].float()
        finally:
            for handle in handles:
                handle.remove()
        return keys, state["scale"], logits

    # ------------------------------------------------------------------ inference
    def encode(self, prompt: str) -> torch.Tensor:
        enc = self.tok([prompt], add_special_tokens=False, truncation=True, return_tensors="pt")
        return enc["input_ids"].to(self.device)

    @torch.no_grad()
    def probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Next-token distribution after `input_ids` ([vocab], float32)."""
        logits = self.model(input_ids=input_ids).logits[0, -1]
        return torch.softmax(logits.to(torch.float), dim=-1)

    def failing_points(
        self, source: str, target_ids: List[int]
    ) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """Teacher-forced walk over the target: yields (input_ids, target_label, argmax_label) where they differ.

        The prediction is taken when a point is reached, so it reflects the repairs of earlier points.
        """
        source_ids = self.encode(source)
        target_ids = [int(label) for label in target_ids]
        for index, target_label in enumerate(target_ids):
            # the oracle prefix is appended as token ids, so the positions are exactly those of the target
            prefix = torch.tensor([target_ids[:index]], dtype=source_ids.dtype, device=self.device)
            input_ids = torch.cat([source_ids, prefix], dim=-1)
            argmax_label = int(self.probs(input_ids).argmax().item())
            if argmax_label != target_label:
                yield input_ids, target_label, argmax_label
