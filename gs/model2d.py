"""P3: a set of 2D Gaussians as optimizable parameters."""
import torch

from .gaussian2d import covariance_2d
from .render import render

PARAM_NAMES = ("mu", "log_s", "theta", "color", "op_raw")


def init_gaussians_2d(N, H, W, device="cpu", seed=0):
    """The handout's spread-out init. Returns a dict of leaf tensors (requires_grad=True).

    mu      (N, 2)  uniform over the image, in pixels (x, y)
    log_s   (N, 2)  log of the per-axis std dev, 2% of the image size
    theta   (N,)    rotation, radians
    color   (N, 3)  logits, sigmoid -> 0.5 gray
    op_raw  (N,)    logit, sigmoid -> ~0.12 opacity
    """
    g = torch.Generator().manual_seed(seed)
    params = {
        "mu": torch.rand(N, 2, generator=g) * torch.tensor([W, H], dtype=torch.float32),
        "log_s": torch.log(0.02 * max(H, W) * torch.ones(N, 2)),
        "theta": torch.zeros(N),
        "color": torch.zeros(N, 3),
        "op_raw": torch.full((N,), -2.0),
    }
    return {k: v.to(device).requires_grad_() for k, v in params.items()}


def render_gaussians_2d(params, H, W):
    """Render with a fixed front-to-back order (index order); there is no depth in 2D."""
    Sigma = covariance_2d(params["log_s"].exp(), params["theta"])
    order = torch.arange(params["mu"].shape[0], device=params["mu"].device)
    return render(params["mu"], Sigma, params["color"].sigmoid(), params["op_raw"].sigmoid(),
                  order, H, W)
