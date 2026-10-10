"""Sanity checks for P1 (covariance / weights) and P2 (rasterizer).

Run with:  python -m tests.test_p1_p2   (or pytest, if installed)
"""
import math

import torch

from gs.gaussian2d import covariance_2d, gaussian_weight
from gs.render import pixel_grid, render, render_dense, render_loop


def test_covariance_axis_aligned():
    scale = torch.tensor([[2.0, 0.5]])
    Sigma = covariance_2d(scale, torch.zeros(1))
    assert torch.allclose(Sigma[0], torch.diag(torch.tensor([4.0, 0.25])))


def test_covariance_rotated_is_valid():
    torch.manual_seed(0)
    scale = torch.rand(64, 2) * 5 + 0.1
    theta = torch.rand(64) * 2 * math.pi
    Sigma = covariance_2d(scale, theta)
    assert torch.allclose(Sigma, Sigma.transpose(-1, -2), atol=1e-5)
    # Eigenvalues are the squared scales, independent of the rotation.
    eig = torch.linalg.eigvalsh(Sigma)
    assert torch.allclose(eig, (scale ** 2).sort(dim=1).values, rtol=1e-4, atol=1e-4)


def test_covariance_rotation_direction():
    # Rotating a long-in-x Gaussian by 90 degrees should make it long in y.
    Sigma = covariance_2d(torch.tensor([[3.0, 1.0]]), torch.tensor([math.pi / 2]))
    assert torch.allclose(Sigma[0], torch.diag(torch.tensor([1.0, 9.0])), atol=1e-5)


def test_weight_matches_explicit_inverse():
    torch.manual_seed(0)
    N, P = 16, 100
    mu = torch.rand(N, 2) * 20
    Sigma = covariance_2d(torch.rand(N, 2) * 4 + 0.5, torch.rand(N) * 3)
    xy = torch.rand(P, 2) * 20
    w = gaussian_weight(xy, mu, Sigma)
    d = xy[:, None, :] - mu[None]                                  # (P, N, 2)
    q = torch.einsum("pni,nij,pnj->pn", d, torch.linalg.inv(Sigma), d)
    assert w.shape == (P, N)
    assert torch.allclose(w, torch.exp(-0.5 * q), atol=1e-5)
    # Weight is exactly 1 at the center.
    assert torch.allclose(gaussian_weight(mu, mu, Sigma).diagonal(), torch.ones(N))


def test_pixel_grid_layout():
    xy = pixel_grid(2, 3)
    assert xy.shape == (6, 2)
    assert torch.equal(xy[1], torch.tensor([1.5, 0.5]))            # row 0, col 1
    assert torch.equal(xy[3], torch.tensor([0.5, 1.5]))            # row 1, col 0


def _random_scene(N, H, W, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    mu = (torch.rand(N, 2, generator=g) * torch.tensor([W, H])).to(dtype)
    scale = (torch.rand(N, 2, generator=g) * 3 + 1).to(dtype)
    theta = (torch.rand(N, generator=g) * 3).to(dtype)
    color = torch.rand(N, 3, generator=g).to(dtype)
    opacity = (torch.rand(N, generator=g) * 0.9 + 0.05).to(dtype)
    return [t.requires_grad_() for t in (mu, scale, theta, color, opacity)]


def _render_and_grads(fn, params, order, H, W, **kw):
    mu, scale, theta, color, opacity = params
    img = fn(mu, covariance_2d(scale, theta), color, opacity, order, H, W, **kw)
    loss = (img * torch.linspace(0, 1, H * W * 3).reshape(H, W, 3)).sum()
    return img.detach(), torch.autograd.grad(loss, params)


def test_renderers_agree_forward_and_backward():
    # Image size not a multiple of the tile size, a few Gaussians partly or fully off
    # screen, and small max_elems so the chunked + checkpointed paths run too.
    N, H, W = 60, 21, 37
    params = _random_scene(N, H, W)
    with torch.no_grad():
        params[0][:4] = torch.tensor([[-3.0, 5.0], [40.0, 10.0], [-50.0, -50.0], [10.0, 25.0]])
    order = torch.randperm(N, generator=torch.Generator().manual_seed(1))

    ref_img, ref_grads = _render_and_grads(render_loop, params, order, H, W)
    for fn, kw in [(render_dense, {}), (render_dense, {"max_elems": 37 * N}),
                   (render, {"tile": 8}), (render, {"tile": 8, "max_elems": 3 * 64 * 32}),
                   (render, {"tile": 16})]:
        img, grads = _render_and_grads(fn, params, order, H, W, **kw)
        assert torch.allclose(ref_img, img, atol=1e-5), (fn.__name__, kw)
        for g_ref, g in zip(ref_grads, grads):
            assert torch.allclose(g_ref, g, rtol=1e-4, atol=1e-5), (fn.__name__, kw)


def test_render_no_visible_gaussians():
    params = _random_scene(3, 8, 8)
    with torch.no_grad():
        params[0][:] = -100.0
    img = render(params[0], covariance_2d(params[1], params[2]), params[3], params[4],
                 torch.arange(3), 8, 8)
    assert torch.equal(img, torch.zeros(8, 8, 3))


def test_render_order_matters_and_is_respected():
    # Two coincident opaque Gaussians: the front one should dominate the color.
    mu = torch.tensor([[4.0, 4.0], [4.0, 4.0]])
    Sigma = covariance_2d(torch.full((2, 2), 3.0), torch.zeros(2))
    color = torch.tensor([[1.0, 0, 0], [0, 0, 1.0]])
    opacity = torch.tensor([0.99, 0.99])
    red_front = render(mu, Sigma, color, opacity, torch.tensor([0, 1]), 8, 8)
    blue_front = render(mu, Sigma, color, opacity, torch.tensor([1, 0]), 8, 8)
    assert red_front[4, 4, 0] > 0.9 and red_front[4, 4, 2] < 0.1
    assert blue_front[4, 4, 2] > 0.9 and blue_front[4, 4, 0] < 0.1


def test_render_gradcheck():
    N, H, W = 5, 5, 6
    params = _random_scene(N, H, W, dtype=torch.float64)
    order = torch.arange(N)

    def f(mu, scale, theta, color, opacity):
        return render(mu, covariance_2d(scale, theta), color, opacity, order, H, W)

    assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
