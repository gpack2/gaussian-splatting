"""P7: a set of 3D Gaussians as optimizable parameters, rendered through a camera."""
import math

import torch

from .gaussian3d import covariance_3d, project_gaussian
from .render import render

# Gaussians closer to the camera than this (or behind it) are skipped.
NEAR = 0.2
# Added to the diagonal of every projected 2D covariance (pixels^2), as in the 3DGS paper:
# a small screen-space blur so no splat is thinner than about a pixel. It keeps edge-on,
# flat Gaussians from producing singular 2D covariances.
DILATION = 0.3


def fibonacci_sphere(n):
    """n roughly evenly spaced unit vectors -> (n, 3)."""
    i = torch.arange(n, dtype=torch.float32) + 0.5
    y = 1 - 2 * i / n
    r = torch.sqrt(1 - y * y)
    phi = math.pi * (3 - math.sqrt(5)) * i
    return torch.stack([r * torch.cos(phi), y, r * torch.sin(phi)], dim=-1)


def init_gaussians_3d(N, device="cpu", seed=0, half_extent=1.5, scale=0.08, n_bg=0, bg_radius=1000.0):
    """The handout's init: a uniform random cloud in [-half_extent, half_extent]^3.

    mu      (N, 3)  world positions
    log_s   (N, 3)  log of per-axis std dev (world units)
    quat    (N, 4)  rotation as (w, x, y, z), identity
    color   (N, 3)  logits, sigmoid -> 0.5 gray
    op_raw  (N,)    logit, sigmoid -> ~0.12 opacity

    Optionally n_bg extra "background" Gaussians on a sphere of radius bg_radius, evenly
    spaced and wide enough to overlap, starting nearly opaque. Being far outside the camera
    orbit, they act like an environment map at infinity (almost no parallax between views),
    so they can paint the backdrop consistently for every camera, including new ones.
    """
    g = torch.Generator().manual_seed(seed)
    quat = torch.zeros(N + n_bg, 4)
    quat[:, 0] = 1.0
    mu = (torch.rand(N, 3, generator=g) * 2 - 1) * half_extent
    log_s = torch.full((N, 3), math.log(scale))
    op_raw = torch.full((N,), -2.0)
    if n_bg:
        spacing = bg_radius * math.sqrt(4 * math.pi / n_bg)     # distance between neighbors
        mu = torch.cat([mu, bg_radius * fibonacci_sphere(n_bg)])
        log_s = torch.cat([log_s, torch.full((n_bg, 3), math.log(0.7 * spacing))])
        op_raw = torch.cat([op_raw, torch.full((n_bg,), 2.0)])  # sigmoid -> ~0.88
    params = {
        "mu": mu,
        "log_s": log_s,
        "quat": quat,
        "color": torch.zeros(N + n_bg, 3),
        "op_raw": op_raw,
    }
    return {k: v.to(device).requires_grad_() for k, v in params.items()}


def render_gaussians_3d(params, cam):
    """Project every Gaussian into `cam`, sort by camera depth, and composite. -> (H, W, 3)"""
    Sig3 = covariance_3d(params["log_s"].exp(), params["quat"])
    mu2, Sig2, depth = project_gaussian(params["mu"], Sig3, cam.R, cam.t, cam.K, cam.W, cam.H)
    Sig2 = Sig2 + DILATION * torch.eye(2, device=Sig2.device)
    # Front to back: nearest (smallest z_c) first, leaving out anything behind the near plane.
    visible = (depth > NEAR).nonzero().squeeze(1)
    order = visible[torch.argsort(depth[visible])]
    return render(mu2, Sig2, params["color"].sigmoid(), params["op_raw"].sigmoid(), order,
                  cam.H, cam.W)
