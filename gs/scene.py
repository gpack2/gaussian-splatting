"""Posed cameras for the 3D tasks (P7-P9)."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .utils import load_image


@dataclass
class Camera:
    name: str
    R: torch.Tensor        # (3, 3) world -> camera rotation (OpenCV: x right, y down, z forward)
    t: torch.Tensor        # (3,)   world -> camera translation, X_cam = R X_world + t
    K: torch.Tensor        # (3, 3) intrinsics
    H: int
    W: int
    image: torch.Tensor = None   # (H, W, 3) in [0, 1], if this is a captured view

    @property
    def center(self):
        return -self.R.T @ self.t


def load_scene(root, device="cpu"):
    """Read cameras.json + images. Returns (train_cameras, val_cameras)."""
    root = Path(root)
    meta = json.loads((root / "cameras.json").read_text())
    K = torch.tensor(meta["K"], dtype=torch.float32, device=device)

    def make(frames):
        return [Camera(name=f["file"],
                       R=torch.tensor(f["R_wc"], dtype=torch.float32, device=device),
                       t=torch.tensor(f["t"], dtype=torch.float32, device=device),
                       K=K, H=meta["height"], W=meta["width"],
                       image=load_image(root / f["file"], device))
                for f in frames]

    return make(meta["frames"]), make(meta["val_frames"])


def look_at_camera(azimuth_deg, elevation_deg, radius, K, H, W, name="orbit"):
    """A camera on a sphere around the origin looking at it, in the dataset's convention
    (world y is up; azimuth 0 is on +z, increasing toward +x)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    center = radius * torch.tensor([math.cos(el) * math.sin(az), math.sin(el),
                                    math.cos(el) * math.cos(az)])
    forward = -center / center.norm()                                  # camera +z
    right = torch.linalg.cross(forward, torch.tensor([0.0, 1.0, 0.0]))
    right = right / right.norm()                                       # camera +x
    down = torch.linalg.cross(forward, right)                          # camera +y
    R = torch.stack([right, down, forward]).to(K.device)
    return Camera(name=name, R=R, t=-R @ center.to(K.device), K=K, H=H, W=W)
