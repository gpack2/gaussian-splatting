"""P6: 3D Gaussians and their projection into a pinhole camera."""
import torch


def quaternion_to_rotation(q):
    # q: (N, 4) as (w, x, y, z), any nonzero length -> (N, 3, 3)
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1),
    ], dim=-2)


def covariance_3d(scale, quat):
    # scale: (N, 3) positive,  quat: (N, 4)
    # Sigma = R S S^T R^T with S = diag(scale); R S scales the columns of R.
    RS = quaternion_to_rotation(quat) * scale[:, None, :]   # (N, 3, 3)
    return RS @ RS.transpose(-1, -2)                        # (N, 3, 3)


def project_gaussian(mu3, Sigma3, R_wc, t, K, W=None, H=None):
    # mu3: (N, 3) world means,  Sigma3: (N, 3, 3) world covariances
    # R_wc: (3, 3), t: (3,) world -> camera,  K: (3, 3) intrinsics
    # Returns 2D means (N, 2) in pixels, 2D covariances (N, 2, 2), camera-space depth z_c (N,).
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mu_cam = mu3 @ R_wc.T + t                               # world -> camera
    x, y, z = mu_cam.unbind(-1)

    mu2 = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)     # pinhole projection

    # Jacobian of (u, v) w.r.t. (x_c, y_c, z_c) at the mean. As in the 3DGS reference code,
    # x/z and y/z are clamped to a bit outside the field of view when building J, so
    # Gaussians far off screen don't get huge, nearly degenerate footprints. This only
    # changes Gaussians that are already well outside the image.
    if W is not None and H is not None:
        lim_x, lim_y = 1.3 * (W / 2) / fx, 1.3 * (H / 2) / fy
        x = (x / z).clamp(-lim_x, lim_x) * z
        y = (y / z).clamp(-lim_y, lim_y) * z
    zero = torch.zeros_like(z)
    J = torch.stack([torch.stack([fx / z, zero, -fx * x / z ** 2], dim=-1),
                     torch.stack([zero, fy / z, -fy * y / z ** 2], dim=-1)], dim=-2)   # (N, 2, 3)

    Scam = R_wc @ Sigma3 @ R_wc.T                           # (N, 3, 3) camera-space covariance
    Sig2 = J @ Scam @ J.transpose(-1, -2)                   # (N, 2, 2)
    depth = mu_cam[:, 2]
    return mu2, Sig2, depth
