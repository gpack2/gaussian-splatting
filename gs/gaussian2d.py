"""P1: 2D Gaussian primitive."""
import torch


def rotation_2d(theta):
    # theta: (N,) radians -> (N, 2, 2)
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], dim=-1),
                        torch.stack([s, c], dim=-1)], dim=-2)


def covariance_2d(scale, theta):
    # scale: (N, 2) positive,  theta: (N,) radians
    # Sigma = R S S^T R^T with S = diag(scale). R S just scales the columns of R.
    RS = rotation_2d(theta) * scale[:, None, :]       # (N, 2, 2)
    return RS @ RS.transpose(-1, -2)                  # (N, 2, 2)


def gaussian_weight(xy, mu, Sigma):
    # xy: (P, 2) pixel coords,  mu: (N, 2),  Sigma: (N, 2, 2)
    # w[p, n] = exp(-0.5 (xy_p - mu_n)^T Sigma_n^-1 (xy_p - mu_n))     -> (P, N)
    # Leading batch dims are allowed too: (..., P, 2), (..., N, 2), (..., N, 2, 2) -> (..., P, N).
    # Sigma is symmetric 2x2 [[a, b], [b, c]], so invert it in closed form.
    a, b, c = Sigma[..., 0, 0], Sigma[..., 0, 1], Sigma[..., 1, 1]
    det = a * c - b * b
    ia, ib, ic = [(v / det)[..., None, :] for v in (c, -b, a)]   # entries of Sigma^-1, (..., 1, N)

    dx = xy[..., :, None, 0] - mu[..., None, :, 0]    # (..., P, N)
    dy = xy[..., :, None, 1] - mu[..., None, :, 1]    # (..., P, N)
    q = ia * dx * dx + 2 * ib * dx * dy + ic * dy * dy
    return torch.exp(-0.5 * q)                        # (..., P, N)


def max_radius(Sigma, n_sigma=3.0):
    """Radius of a circle containing each Gaussian's n-sigma ellipse -> (N,)."""
    a, b, c = Sigma[..., 0, 0], Sigma[..., 0, 1], Sigma[..., 1, 1]
    lam_max = 0.5 * (a + c) + torch.sqrt((0.5 * (a - c)) ** 2 + b * b)   # largest eigenvalue
    return n_sigma * torch.sqrt(lam_max)
