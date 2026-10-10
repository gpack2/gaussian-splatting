"""P4 / P8: adaptive density control (clone, split, prune), shared by 2D and 3D.

Works on a dict of parameter tensors with an "mu" (N, D) position, a "log_s" (N, D)
log-scale, an "op_raw" (N,) opacity logit, and any other per-Gaussian tensors
(rotation, color), which are copied along. The optimizer must have one param group
per tensor, tagged with group["name"] equal to its key in the dict.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class DensifyConfig:
    every: int = 200               # run a pass every `every` optimization steps ...
    start: int = 200               # ... starting at this step ...
    stop: int = 1500               # ... until this step (the remaining steps only refine)
    grad_threshold: float = 2e-4   # densify Gaussian i if g_i > grad_threshold ...
    top_frac: float = None         # ... or, if set, densify the top fraction of Gaussians by g_i
    size_threshold: float = 0.02   # clone if max scale <= size_threshold * extent, else split
    split_scale: float = 1.6       # each split child gets (parent scale / split_scale)
    prune_opacity: float = 0.005   # remove Gaussian i if its opacity < this
    reset_at: tuple = ()           # steps at which to cap opacities (3DGS opacity reset)
    reset_opacity: float = 0.01    # ... at this value

    def is_pass(self, step):
        return self.start <= step <= self.stop and step % self.every == 0


class GradStats:
    """Accumulates g_i = (1/M) sum_k ||dL_k / dmu_i||_2 over the M steps since the last pass.

    `grad_scale` converts the gradient to the units the threshold is defined in.
    """

    def __init__(self, n, device, grad_scale=1.0):
        self.grad_scale = grad_scale
        self.reset(n, device)

    def reset(self, n, device):
        self.total = torch.zeros(n, device=device)
        self.steps = 0

    @torch.no_grad()
    def add(self, mu_grad):
        self.total += mu_grad.norm(dim=-1) * self.grad_scale
        self.steps += 1

    def score(self):
        return self.total / max(self.steps, 1)


@torch.no_grad()
def reset_opacity(params, opt, max_opacity, mask=None):
    """3DGS opacity reset: cap the opacity of the Gaussians in `mask` (default: all) at
    max_opacity and clear their Adam moments. Gaussians the views need regain opacity;
    the rest (floaters) stay faint and the next passes prune them."""
    mask = torch.ones_like(params["op_raw"], dtype=torch.bool) if mask is None else mask
    cap = math.log(max_opacity / (1 - max_opacity))
    op = params["op_raw"]
    op[mask] = op[mask].clamp(max=cap)
    state = opt.state.get(op)
    if state:
        state["exp_avg"][mask] = 0
        state["exp_avg_sq"][mask] = 0


def replace_params(params, opt, keep, new, perm=None):
    """Keep the rows of every parameter selected by the bool mask `keep`, append the rows in
    `new`, then reorder the result by `perm` if given. The Adam state is updated to match:
    surviving Gaussians keep their moments, new ones start from zero."""
    assert {g["name"] for g in opt.param_groups} == set(params), "one param group per tensor"

    def rebuild(kept, added):
        out = torch.cat([kept, added])
        return out if perm is None else out[perm]

    for group in opt.param_groups:
        name = group["name"]
        old = group["params"][0]
        p = rebuild(old.detach()[keep], new[name]).requires_grad_()
        state = opt.state.pop(old, None)
        if state:
            for k in ("exp_avg", "exp_avg_sq"):
                state[k] = rebuild(state[k][keep], torch.zeros_like(new[name]))
            opt.state[p] = state
        group["params"][0] = p
        params[name] = p


@torch.no_grad()
def densify_and_prune(params, opt, score, cfg, budget, extent, rotation_fn):
    """One densification pass. Modifies `params` and `opt` in place.

    score:       (N,) g_i from GradStats
    budget:      maximum number of Gaussians after the pass
    extent:      size the scale threshold is relative to (image width in 2D, scene size in 3D)
    rotation_fn: maps a params dict to rotation matrices (N, D, D), for sampling split offsets
    Returns a dict of counts for logging.
    """
    n = params["mu"].shape[0]
    opacity = params["op_raw"].sigmoid()
    max_scale = params["log_s"].exp().max(dim=-1).values

    prune = opacity < cfg.prune_opacity
    if cfg.top_frac is None:
        dense = score > cfg.grad_threshold
    else:
        dense = score >= score.topk(max(1, int(cfg.top_frac * n))).values[-1]
    dense &= ~prune
    # Clone and split each add one Gaussian net. If there is not enough room under the
    # budget, densify only the highest-scoring candidates.
    room = max(0, budget - (n - int(prune.sum())))
    if int(dense.sum()) > room:
        top = torch.where(dense, score, torch.full_like(score, -math.inf)).topk(room).indices
        dense = torch.zeros_like(dense)
        dense[top] = True
    clone = dense & (max_scale <= cfg.size_threshold * extent)
    split = dense & ~clone

    # Clones are exact copies (they separate as the optimizer moves them).
    cloned = {k: v[clone] for k, v in params.items()}
    # Each split parent becomes two children, placed at samples from the parent Gaussian
    # and shrunk by split_scale.
    children = {k: v[split].repeat_interleave(2, dim=0) for k, v in params.items()}
    s = children["log_s"].exp()
    offset = (rotation_fn(children) @ (s * torch.randn_like(s))[..., None])[..., 0]
    children["mu"] = children["mu"] + offset
    children["log_s"] = children["log_s"] - math.log(cfg.split_scale)

    new = {k: torch.cat([cloned[k], children[k]]) for k in params}
    # Put each new Gaussian at its source's place in the list rather than at the end. Order
    # is irrelevant in 3D (we sort by depth), but in 2D the list order is the compositing
    # order, and moving a split parent's children to the very back changes the image a lot.
    keep = ~(prune | split)
    idx = torch.arange(n, device=score.device)
    src = torch.cat([2 * idx[keep],                                   # survivors
                     2 * idx[clone] + 1,                              # clone right behind original
                     2 * idx[split].repeat_interleave(2) + 1])        # children where the parent was
    replace_params(params, opt, keep, new, perm=torch.argsort(src))
    return {"cloned": int(clone.sum()), "split": int(split.sum()), "pruned": int(prune.sum()),
            "total": params["mu"].shape[0]}
