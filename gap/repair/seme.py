from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.optim.optimizer import Optimizer


def min_norm_coefficients(acts: torch.Tensor, steers: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
    """Coefficients C ([points, hidden]) of the minimum-norm W = (acts * weights)^T @ C that solves acts @ W = steers.

    acts is [points, neurons], steers is [points, hidden]. Without `weights` the result equals
    `torch.linalg.lstsq(acts, steers)` for these under-determined systems, but only the small [points, points] Gram
    matrix is inverted. With `weights` ([neurons], the preservation weights of the null space) the norm of the change
    of a neuron is weighted by 1 / weights, so neurons that unrelated text rarely activates carry most of it.
    """
    gram = (acts if weights is None else acts * weights) @ acts.T
    return torch.linalg.pinv(gram) @ steers


class Seme(Optimizer):
    """The STAR optimizer: a prior-guided update.

    For every parameter, the *magnitude* of the step comes from a prior delta (the analytical solution that
    steers the latent state toward the expected token), and the *sign* comes from the gradient of the loss:
    entries of the prior that point against the descent direction are flipped.

    A prior has the size of a weight matrix, so it is passed in factored form (activations and coefficients)
    and materialized for one parameter at a time inside `step`, then released.
    """

    def __init__(self, params, lr: float = 1e-2):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr))
        self._factors: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def set_priors(self, factors: Dict[torch.nn.Parameter, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """`factors[param] = (acts, coefficients)`: the prior of `param` ([hidden, neurons]) is coefficients^T @ acts.

        Priors are keyed by the parameter itself, so the pairing cannot depend on any ordering.
        """
        self._factors = {id(param): factor for param, factor in factors.items()}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if id(param) not in self._factors:
                    raise RuntimeError("Seme.set_priors must provide a prior for every trainable parameter")
                acts, coefficients = self._factors[id(param)]
                prior = coefficients.T @ acts                                  # [hidden, neurons], float32
                if prior.shape != param.shape:
                    raise ValueError(f"prior shape {tuple(prior.shape)} does not match parameter {tuple(param.shape)}")
                # flip, in place, the entries whose sign is not the descent direction
                against_descent = prior.sign() != -param.grad.sign()
                prior[against_descent] *= -1
                param.add_(prior.to(param.dtype), alpha=group["lr"])
                del prior, against_descent
        self._factors = {}
        return loss
