"""Checks for the densification pass (P4 / P8).

Run with:  python -m tests.test_densify
"""
import math

import torch

from gs.densify import DensifyConfig, densify_and_prune
from gs.gaussian2d import rotation_2d


def _setup():
    # 5 Gaussians: 0 small + high score (clone), 1 large + high score (split),
    # 2 transparent (prune), 3 and 4 low score (untouched).
    params = {
        "mu": torch.tensor([[1.0, 1], [2, 2], [3, 3], [4, 4], [5, 5]]),
        "log_s": torch.log(torch.tensor([[0.5, 0.5], [10, 2], [1, 1], [1, 1], [1, 1]])),
        "theta": torch.zeros(5),
        "color": torch.arange(15, dtype=torch.float32).reshape(5, 3),
        "op_raw": torch.tensor([0.0, 0.0, -10.0, 0.0, 0.0]),
    }
    params = {k: v.requires_grad_() for k, v in params.items()}
    opt = torch.optim.Adam([{"params": [v], "name": k} for k, v in params.items()], lr=0.1)
    loss = sum((v ** 2).sum() for v in params.values())
    loss.backward()
    opt.step()
    score = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
    cfg = DensifyConfig(grad_threshold=0.5, size_threshold=0.02, split_scale=1.6, prune_opacity=0.005)
    return params, opt, score, cfg


def _rot(p):
    return rotation_2d(p["theta"])


def test_clone_split_prune_and_order():
    params, opt, score, cfg = _setup()
    old_m = opt.state[params["color"]]["exp_avg"].clone()
    old_color = params["color"].detach().clone()
    old_scale = params["log_s"].exp().detach().clone()
    info = densify_and_prune(params, opt, score, cfg, budget=100, extent=100.0, rotation_fn=_rot)
    assert info == {"cloned": 1, "split": 1, "pruned": 1, "total": 6}

    # Expected order: 0, clone of 0, child of 1, child of 1, 3, 4 (2 pruned, 1 replaced).
    color = params["color"].detach()
    assert torch.equal(color, old_color[[0, 0, 1, 1, 3, 4]])
    scale = params["log_s"].exp().detach()
    assert torch.allclose(scale[2], old_scale[1] / 1.6)
    assert torch.allclose(scale[3], old_scale[1] / 1.6)

    # Adam moments: survivors keep theirs, new Gaussians start at zero.
    m = opt.state[params["color"]]["exp_avg"]
    assert torch.equal(m[0], old_m[0]) and torch.equal(m[4], old_m[3]) and torch.equal(m[5], old_m[4])
    assert torch.equal(m[[1, 2, 3]], torch.zeros(3, 3))
    # The optimizer now holds the new tensors.
    assert all(g["params"][0] is params[g["name"]] for g in opt.param_groups)
    assert all(v.requires_grad and v.is_leaf for v in params.values())


def test_budget_caps_growth_to_top_scores():
    params, opt, score, cfg = _setup()
    score = torch.tensor([0.9, 1.0, 1.0, 0.0, 0.0])
    # 5 - 1 pruned = 4, so a budget of 5 leaves room for one densification: the higher score (1).
    info = densify_and_prune(params, opt, score, cfg, budget=5, extent=100.0, rotation_fn=_rot)
    assert info == {"cloned": 0, "split": 1, "pruned": 1, "total": 5}


def test_top_frac_mode():
    params, opt, score, cfg = _setup()
    cfg.top_frac = 0.2           # 1 of 5
    score = torch.tensor([0.1, 0.3, 0.2, 0.0, 0.0])
    info = densify_and_prune(params, opt, score, cfg, budget=100, extent=100.0, rotation_fn=_rot)
    assert info["split"] == 1 and info["cloned"] == 0


def test_training_continues_after_densify():
    params, opt, score, cfg = _setup()
    densify_and_prune(params, opt, score, cfg, budget=100, extent=100.0, rotation_fn=_rot)
    loss = sum((v ** 2).sum() for v in params.values())
    opt.zero_grad()
    loss.backward()
    before = params["mu"].detach().clone()
    opt.step()
    assert not torch.equal(before, params["mu"].detach())
    assert all(math.isfinite(v.detach().abs().max().item()) for v in params.values())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
