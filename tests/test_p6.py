"""Sanity checks for P6 (3D Gaussians and projection) and the camera helpers.

Run with:  python -m tests.test_p6
"""
import json
import math
from pathlib import Path

import torch

from gs.gaussian3d import covariance_3d, project_gaussian, quaternion_to_rotation
from gs.scene import look_at_camera

K = torch.tensor([[200.0, 0, 80], [0, 210.0, 75], [0, 0, 1]])


def test_quaternion_rotation_is_valid():
    torch.manual_seed(0)
    q = torch.randn(32, 4) * 3                       # deliberately not unit length
    R = quaternion_to_rotation(q)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(32, 3, 3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(32), atol=1e-5)
    assert torch.allclose(quaternion_to_rotation(torch.tensor([[2.0, 0, 0, 0]]))[0], torch.eye(3))


def test_quaternion_rotation_direction():
    # 90 degrees about +z maps +x to +y.
    q = torch.tensor([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])
    assert torch.allclose(quaternion_to_rotation(q)[0] @ torch.tensor([1.0, 0, 0]),
                          torch.tensor([0.0, 1, 0]), atol=1e-6)


def test_covariance_3d_eigenvalues():
    torch.manual_seed(0)
    scale = torch.rand(16, 3) + 0.1
    Sigma = covariance_3d(scale, torch.randn(16, 4))
    eig = torch.linalg.eigvalsh(Sigma)
    assert torch.allclose(eig, (scale ** 2).sort(dim=1).values, rtol=1e-4, atol=1e-6)


def _random_pose(seed):
    g = torch.Generator().manual_seed(seed)
    R = quaternion_to_rotation(torch.randn(1, 4, generator=g))[0]
    return R, torch.tensor([0.1, -0.2, 4.0])


def test_projection_mean_and_jacobian():
    R, t = _random_pose(1)
    torch.manual_seed(0)
    mu3 = torch.randn(8, 3) * 0.5
    mu2, _, depth = project_gaussian(mu3, torch.eye(3).expand(8, 3, 3), R, t, K)
    Xc = mu3 @ R.T + t
    assert torch.allclose(depth, Xc[:, 2])
    assert torch.allclose(mu2, (Xc @ K.T)[:, :2] / Xc[:, 2:], atol=1e-4)

    # J from project_gaussian equals the autograd Jacobian of the pixel projection:
    # with world covariance R^T R = I, the camera covariance is I and Sig2 = J J^T.
    def proj(xc):
        return torch.stack([K[0, 0] * xc[0] / xc[2] + K[0, 2], K[1, 1] * xc[1] / xc[2] + K[1, 2]])
    _, Sig2, _ = project_gaussian(mu3, torch.eye(3).expand(8, 3, 3), R, t, K)
    for i in range(8):
        J = torch.autograd.functional.jacobian(proj, Xc[i])
        assert torch.allclose(Sig2[i], J @ J.T, rtol=1e-4, atol=1e-4)


def test_projected_covariance_matches_samples():
    # For a small Gaussian the linearized projection matches the true spread of projected samples.
    R, t = _random_pose(2)
    mu3 = torch.tensor([[0.3, -0.2, 0.1]])
    Sigma3 = covariance_3d(torch.tensor([[0.05, 0.02, 0.01]]), torch.tensor([[0.9, 0.2, -0.3, 0.1]]))
    mu2, Sig2, _ = project_gaussian(mu3, Sigma3, R, t, K)

    torch.manual_seed(0)
    L = torch.linalg.cholesky(Sigma3[0].double())
    X = mu3[0].double() + torch.randn(200_000, 3, dtype=torch.float64) @ L.T
    Xc = X @ R.double().T + t.double()
    uv = (Xc @ K.double().T)[:, :2] / Xc[:, 2:]
    emp = torch.cov(uv.T)
    assert torch.allclose(uv.mean(0).float(), mu2[0], atol=0.05)
    assert torch.allclose(emp.float(), Sig2[0], rtol=0.03, atol=0.02)


def test_look_at_reproduces_dataset_cameras():
    path = Path("data/spheres/cameras.json")
    if not path.exists():
        print("  (skipped: data/spheres not downloaded)")
        return
    meta = json.loads(path.read_text())
    Kd = torch.tensor(meta["K"])
    for f in meta["frames"] + meta["val_frames"]:
        R, t = torch.tensor(f["R_wc"]), torch.tensor(f["t"])
        C = -R.T @ t
        r = C.norm().item()
        az = math.degrees(math.atan2(C[0], C[2]))
        el = math.degrees(math.asin(C[1] / r))
        cam = look_at_camera(az, el, r, Kd, 160, 160)
        assert torch.allclose(cam.R, R, atol=1e-5), f["file"]
        assert torch.allclose(cam.t, t, atol=1e-5), f["file"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
