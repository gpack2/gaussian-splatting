"""P2: differentiable rasterizer (front-to-back alpha compositing).

Three implementations of the same image formula, C = sum_i c_i alpha_i T_i:
  render_loop   the handout's loop, one Gaussian at a time (slow, reference)
  render_dense  vectorized over all (pixel, Gaussian) pairs (reference)
  render        tiled: each 16x16 tile only composites the Gaussians overlapping it

All three truncate each Gaussian at 3 sigma (the "local window" from the handout),
so they agree to floating-point precision and the tests check that.
"""
import math

import torch
from torch.utils.checkpoint import checkpoint

from .gaussian2d import gaussian_weight, max_radius

# Alpha is clamped below 1 (as in the 3DGS paper) so the transmittance never
# hits exactly zero, which keeps the cumulative product and its gradient stable.
MAX_ALPHA = 0.99
# Each Gaussian only touches pixels inside its 3-sigma ellipse, i.e. where the
# weight is at least exp(-4.5) ~= 0.011.
CUTOFF_SIGMA = 3.0
MIN_WEIGHT = math.exp(-0.5 * CUTOFF_SIGMA ** 2)


def pixel_grid(H, W, device="cpu"):
    """Pixel-center coordinates (x, y) = (col + 0.5, row + 0.5), row-major -> (H*W, 2)."""
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32) + 0.5,
                            torch.arange(W, device=device, dtype=torch.float32) + 0.5,
                            indexing="ij")
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2)


def _alpha(w, opacity):
    """Per-(pixel, Gaussian) alpha from weights; opacity broadcasts along the last dim."""
    return (opacity * w * (w >= MIN_WEIGHT)).clamp(max=MAX_ALPHA)


def render_loop(mu, Sigma, color, opacity, order, H, W):
    """Reference rasterizer: the handout's loop, one Gaussian at a time."""
    xy = pixel_grid(H, W, mu.device)                  # (H*W, 2)
    w = gaussian_weight(xy, mu, Sigma)                # (P, N)
    alpha = _alpha(w, opacity[None, :])               # (P, N)
    C = torch.zeros(H * W, 3, device=mu.device)
    T = torch.ones(H * W, device=mu.device)
    for i in order:                                   # front to back
        a = alpha[:, i]
        C = C + (a * T)[:, None] * color[i]           # add color weighted by alpha * transmittance
        T = T * (1 - a)                               # light remaining for the Gaussians behind
    return C.reshape(H, W, 3)


def _composite(xy, mu, Sigma, color, opacity):
    """Composite Gaussians that are already sorted front to back along the Gaussian axis.

    xy: (..., P, 2), mu: (..., N, 2), Sigma: (..., N, 2, 2), color: (..., N, 3),
    opacity: (..., 1, N). Returns (..., P, 3).
    """
    alpha = _alpha(gaussian_weight(xy, mu, Sigma), opacity)          # (..., P, N)
    # T_i = prod_{j<i} (1 - alpha_j): an exclusive cumulative product along the sorted axis.
    T = torch.cumprod(1 - alpha, dim=-1)
    T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
    # C = sum_i c_i alpha_i T_i, written as a (P, N) x (N, 3) matmul.
    return (alpha * T) @ color


def _composite_in_chunks(n_elems, max_elems, split_gaussians, xy, *gaussians):
    """Run `_composite` over chunks of dim 0 of xy (and of the Gaussian tensors too if
    `split_gaussians`, for per-tile batches). When the whole problem has more than
    `max_elems` (pixel, Gaussian) pairs, each chunk is gradient-checkpointed so backward
    recomputes its intermediates instead of holding all of them in memory at once."""
    if n_elems <= max_elems:
        return _composite(xy, *gaussians)
    chunk = max(1, xy.shape[0] * max_elems // n_elems)
    needs_grad = torch.is_grad_enabled() and any(g.requires_grad for g in gaussians)
    out = []
    for s in range(0, xy.shape[0], chunk):
        args = [xy[s:s + chunk]] + [g[s:s + chunk] if split_gaussians else g for g in gaussians]
        if needs_grad:
            out.append(checkpoint(_composite, *args, use_reentrant=False))
        else:
            out.append(_composite(*args))
    return torch.cat(out)


def render_dense(mu, Sigma, color, opacity, order, H, W, max_elems=2**24):
    """Vectorized rasterizer over every (pixel, Gaussian) pair. Same output as `render_loop`."""
    mu, Sigma, color, opacity = mu[order], Sigma[order], color[order], opacity[order]
    xy = pixel_grid(H, W, mu.device)
    C = _composite_in_chunks(xy.shape[0] * mu.shape[0], max_elems, False,
                             xy, mu, Sigma, color, opacity[None, :])
    return C.reshape(H, W, 3)


@torch.no_grad()
def _bin_gaussians(mu, Sigma, H, W, tile):
    """Assign Gaussians (already sorted front to back) to the screen tiles their 3-sigma
    circle overlaps.

    Returns idx, valid of shape (n_tiles, K): row t lists the Gaussians touching tile t
    in front-to-back order, padded to the busiest tile's count K (valid marks real entries).
    """
    N, dev = mu.shape[0], mu.device
    TH, TW = -(-H // tile), -(-W // tile)
    n_tiles = TH * TW

    # Tile-space bounding box of each Gaussian's 3-sigma circle (empty if fully off screen).
    r = max_radius(Sigma, CUTOFF_SIGMA).nan_to_num(0.0).clamp(max=2.0 * (H + W))
    mu = mu.nan_to_num(-1e6).clamp(-1e6, 1e6)
    x0 = torch.floor((mu[:, 0] - r) / tile).clamp(min=0).long()
    x1 = torch.floor((mu[:, 0] + r) / tile).clamp(max=TW - 1).long()
    y0 = torch.floor((mu[:, 1] - r) / tile).clamp(min=0).long()
    y1 = torch.floor((mu[:, 1] + r) / tile).clamp(max=TH - 1).long()
    nx, ny = (x1 - x0 + 1).clamp(min=0), (y1 - y0 + 1).clamp(min=0)

    # One (tile, Gaussian) pair for every tile in every box.
    counts = nx * ny
    gid = torch.repeat_interleave(torch.arange(N, device=dev), counts)
    k = torch.arange(gid.numel(), device=dev) - (torch.cumsum(counts, 0) - counts)[gid]
    tid = (y0[gid] + k // nx[gid]) * TW + (x0[gid] + k % nx[gid])

    # Group pairs by tile; inside a tile keep Gaussian (= depth) order.
    key, _ = torch.sort(tid * N + gid)
    tid, gid = key // N, key % N
    tiles = torch.arange(n_tiles, device=dev)
    first = torch.searchsorted(tid, tiles)
    per_tile = torch.searchsorted(tid, tiles, right=True) - first
    # Round K up so its value (and the kernel shapes) changes less often between steps.
    K = max(1, -(-int(per_tile.max()) // 32) * 32) if gid.numel() else 1

    slot = torch.arange(gid.numel(), device=dev) - first[tid]
    idx = torch.zeros(n_tiles, K, dtype=torch.long, device=dev)
    valid = torch.zeros(n_tiles, K, dtype=torch.bool, device=dev)
    idx[tid, slot] = gid
    valid[tid, slot] = True
    return idx, valid


def _tile_pixels(TH, TW, tile, device):
    """Pixel centers grouped by tile -> (TH*TW, tile*tile, 2)."""
    xy = pixel_grid(TH * tile, TW * tile, device).reshape(TH, tile, TW, tile, 2)
    return xy.permute(0, 2, 1, 3, 4).reshape(TH * TW, tile * tile, 2)


def render(mu, Sigma, color, opacity, order, H, W, tile=16, max_elems=2**24):
    """Tiled rasterizer: same output as `render_dense`, much less work.

    mu: (N, 2), Sigma: (N, 2, 2), color: (N, 3) in [0, 1], opacity: (N,) in [0, 1],
    order: (N,) indices sorted front -> back. Returns (H, W, 3) over a black background.
    """
    mu, Sigma, color, opacity = mu[order], Sigma[order], color[order], opacity[order]
    TH, TW = -(-H // tile), -(-W // tile)
    idx, valid = _bin_gaussians(mu.detach(), Sigma.detach(), H, W, tile)   # (n_tiles, K)
    xy = _tile_pixels(TH, TW, tile, mu.device)                              # (n_tiles, tile^2, 2)

    # Gather each tile's Gaussians; padding slots get zero opacity, so they add nothing.
    op = (opacity[idx] * valid)[:, None, :]                                 # (n_tiles, 1, K)
    C = _composite_in_chunks(xy.shape[0] * xy.shape[1] * idx.shape[1], max_elems, True,
                             xy, mu[idx], Sigma[idx], color[idx], op)       # (n_tiles, tile^2, 3)

    C = C.reshape(TH, TW, tile, tile, 3).permute(0, 2, 1, 3, 4)
    return C.reshape(TH * tile, TW * tile, 3)[:H, :W]
