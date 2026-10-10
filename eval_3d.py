"""P9: novel-view evaluation of a fitted 3D run.

    python eval_3d.py --run outputs/p7/plain

Writes to <run>/eval/:
  val_psnr.json        held-out PSNR per view, vs. training PSNR
  psnr_vs_elevation.png  per-view PSNR against camera elevation (train vs. held-out)
  val_error.png        ground truth | render | error map for each held-out view
  orbit.gif, orbit/    a 360-degree orbit at an elevation between the training rings
  orbit_sheet.png      a few orbit frames side by side
  scene_only.png       held-out views rendered without the background Gaussians (if any),
                       which shows floaters and fog in the scene volume
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from gs.model3d import render_gaussians_3d
from gs.scene import load_scene, look_at_camera
from gs.utils import get_device, psnr, save_image

BG_RADIUS_MIN = 100.0   # Gaussians farther than this from the origin are the background shell


def elevation(cam):
    c = cam.center
    return math.degrees(math.asin((c[1] / c.norm()).item()))


def error_map(render, gt, vmax=0.25):
    """Per-pixel RGB error magnitude as a heat map (H, W, 3)."""
    err = ((render - gt) ** 2).sum(-1).sqrt().clamp(max=vmax) / vmax
    return torch.from_numpy(plt.cm.inferno(err.cpu().numpy())[..., :3]).float().to(gt.device)


def to_uint8(img):
    return (img.clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="output directory of fit_3d.py")
    ap.add_argument("--scene", default="data/spheres")
    ap.add_argument("--orbit-elev", type=float, default=25.0)
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    run = Path(args.run)
    out = run / "eval"
    out.mkdir(exist_ok=True)
    params = {k: v.to(device) for k, v in torch.load(run / "gaussians.pt").items()}
    train, val = load_scene(args.scene, device)

    # Held-out vs. training PSNR, and how both vary with camera elevation.
    rows = {}
    for split, cams in [("train", train), ("val", val)]:
        rows[split] = []
        for cam in cams:
            r = render_gaussians_3d(params, cam)
            rows[split].append({"view": cam.name, "elevation": round(elevation(cam), 1),
                                "psnr": psnr(r, cam.image)})
    summary = {s: sum(r["psnr"] for r in rows[s]) / len(rows[s]) for s in rows}
    (out / "val_psnr.json").write_text(json.dumps({"mean": summary, "views": rows}, indent=2))
    print(f"train PSNR {summary['train']:.2f} dB   held-out PSNR {summary['val']:.2f} dB   "
          f"gap {summary['train'] - summary['val']:.2f} dB")

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for split, marker in [("train", "o"), ("val", "^")]:
        ax.scatter([r["elevation"] for r in rows[split]], [r["psnr"] for r in rows[split]],
                   marker=marker, label=f"{split} (mean {summary[split]:.2f} dB)", alpha=0.8)
    ax.set_xlabel("camera elevation (degrees)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "psnr_vs_elevation.png", dpi=130)
    plt.close(fig)

    # Ground truth | render | error, one row per held-out view.
    grid = []
    for cam in val:
        r = render_gaussians_3d(params, cam)
        grid.append(torch.cat([cam.image, r, error_map(r, cam.image)], dim=1))
    save_image(torch.cat(grid, dim=0), out / "val_error.png")

    # Held-out views with only the scene Gaussians (background shell removed).
    is_bg = params["mu"].norm(dim=-1) > BG_RADIUS_MIN
    if is_bg.any():
        scene_only = {k: v[~is_bg] for k, v in params.items()}
        cells = [torch.cat([cam.image, render_gaussians_3d(scene_only, cam)], dim=1) for cam in val[:6]]
        save_image(torch.cat(cells, dim=0), out / "scene_only.png")

    # Orbit at an elevation the training rings (-5, 15, 35, 55) never used.
    radius = train[0].center.norm().item()
    (out / "orbit").mkdir(exist_ok=True)
    frames = []
    for i in range(args.frames):
        cam = look_at_camera(360.0 * i / args.frames, args.orbit_elev, radius, train[0].K,
                             train[0].H, train[0].W)
        img = render_gaussians_3d(params, cam)
        save_image(img, out / "orbit" / f"{i:03d}.png")
        frames.append(Image.fromarray(to_uint8(img)))
    frames[0].save(out / "orbit.gif", save_all=True, append_images=frames[1:], duration=60, loop=0)
    picks = np.linspace(0, args.frames, 7)[:-1].astype(int)
    sheet = np.concatenate([np.asarray(frames[i]) for i in picks], axis=1)
    Image.fromarray(sheet).save(out / "orbit_sheet.png")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
