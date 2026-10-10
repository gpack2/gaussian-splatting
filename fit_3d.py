"""P7 / P8: reconstruct the spheres scene from posed views with 3D Gaussians.

Examples:
    python fit_3d.py --out outputs/p7/plain                 # P7: fixed cloud of 4000
    python fit_3d.py --densify --out outputs/p8/densified   # P8: grow from 1000 to 4000
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gs.densify import DensifyConfig, GradStats, densify_and_prune, reset_opacity
from gs.gaussian3d import quaternion_to_rotation
from gs.model3d import init_gaussians_3d, render_gaussians_3d
from gs.scene import load_scene
from gs.utils import get_device, mse_to_psnr, psnr, save_image

SCENE_EXTENT = 3.0   # the scene fits in a [-1.5, 1.5]^3 cube
BG_RADIUS_MIN = 100.0  # Gaussians farther than this from the origin are the background shell


def densify_config_3d(every=200, stop=1000, top_frac=0.35, prune_opacity=0.005,
                      size_threshold=0.02, reset_at=()):
    """P8 settings: same rule as 2D; the size threshold is a fraction of SCENE_EXTENT
    (0.02 * 3 = 0.06 world units). reset_at = steps of a 3DGS opacity reset."""
    return DensifyConfig(every=every, start=every, stop=stop, top_frac=top_frac,
                         prune_opacity=prune_opacity, size_threshold=size_threshold,
                         reset_at=tuple(reset_at))


def make_optimizer(params, lr, lr_mu=None):
    return torch.optim.Adam([{"params": [params[k]], "name": k,
                              "lr": lr_mu if (k == "mu" and lr_mu is not None) else lr}
                             for k in params])


def fit(params, train, iters, lr, lr_mu=None, densify=None, budget=None, seed=0, log_every=100):
    """Optimize params in place against the training cameras, one random view per step.
    Returns per-step training PSNR and Gaussian-count histories."""
    opt = make_optimizer(params, lr, lr_mu)
    device = params["mu"].device
    stats = GradStats(params["mu"].shape[0], device)
    g = torch.Generator().manual_seed(seed)
    queue, history, counts = [], [], []
    t0 = time.time()
    for step in range(iters):
        if not queue:                                   # visit views in shuffled passes
            queue = torch.randperm(len(train), generator=g).tolist()
        cam = train[queue.pop()]
        img = render_gaussians_3d(params, cam)
        loss = ((img - cam.image) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if densify:
            stats.add(params["mu"].grad)
        opt.step()
        history.append(mse_to_psnr(loss.item()))
        counts.append(params["mu"].shape[0])

        if densify and densify.is_pass(step):
            info = densify_and_prune(params, opt, stats.score(), densify, budget,
                                     extent=SCENE_EXTENT,
                                     rotation_fn=lambda p: quaternion_to_rotation(p["quat"]))
            stats.reset(info["total"], device)
            print(f"  densify @ {step}: +{info['cloned']} cloned, {info['split']} split, "
                  f"-{info['pruned']} pruned -> {info['total']} Gaussians", flush=True)
        if densify and step in densify.reset_at:
            # Scene Gaussians only: the background shell holds no floaters.
            reset_opacity(params, opt, densify.reset_opacity,
                          mask=params["mu"].detach().norm(dim=-1) < BG_RADIUS_MIN)
            print(f"  opacity reset @ {step}", flush=True)
        if log_every and (step % log_every == 0 or step == iters - 1):
            recent = sum(history[-log_every:]) / len(history[-log_every:])
            print(f"step {step:5d}  train psnr (avg of recent steps) {recent:6.2f} dB  "
                  f"N {counts[-1]:5d}  ({time.time() - t0:6.1f}s)", flush=True)
    return history, counts


@torch.no_grad()
def evaluate(params, cams):
    """Render every camera; returns (renders, per-view PSNRs)."""
    renders = [render_gaussians_3d(params, cam) for cam in cams]
    return renders, [psnr(r, cam.image) for r, cam in zip(renders, cams)]


def comparison_grid(cams, renders, cols):
    """Image grid: each cell is ground truth | render, side by side."""
    cells = [torch.cat([c.image, r], dim=1) for c, r in zip(cams, renders)]
    pad = torch.ones_like(cells[0])
    cells += [pad] * (-len(cells) % cols)
    rows = [torch.cat(cells[i:i + cols], dim=1) for i in range(0, len(cells), cols)]
    return torch.cat(rows, dim=0)


def save_results(out, params, train, val, history, counts, meta, show_train=(0, 12, 33)):
    out.mkdir(parents=True, exist_ok=True)
    train_renders, train_psnr = evaluate(params, train)
    val_renders, val_psnr = evaluate(params, val)

    shown = [i for i in show_train if i < len(train)]
    save_image(comparison_grid([train[i] for i in shown], [train_renders[i] for i in shown], 1),
               out / "train_compare.png")
    save_image(comparison_grid(val, val_renders, 3), out / "val_compare.png")
    (out / "val").mkdir(exist_ok=True)
    for cam, r in zip(val, val_renders):
        save_image(r, out / "val" / Path(cam.name).name)
    torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "gaussians.pt")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(history, lw=0.6, alpha=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel("train PSNR of the step's view (dB)")
    ax.grid(alpha=0.3)
    if meta["densify"]:
        ax2 = ax.twinx()
        ax2.plot(counts, color="gray", lw=1, ls="--")
        ax2.set_ylabel("# Gaussians", color="gray")
    fig.tight_layout()
    fig.savefig(out / "psnr_curve.png", dpi=120)
    plt.close(fig)

    metrics = {**meta, "final_n": params["mu"].shape[0],
               "train_psnr_mean": sum(train_psnr) / len(train_psnr),
               "val_psnr_mean": sum(val_psnr) / len(val_psnr),
               "train_psnr": {c.name: p for c, p in zip(train, train_psnr)},
               "val_psnr": {c.name: p for c, p in zip(val, val_psnr)}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run(out, n=4000, iters=1500, lr=1e-2, lr_mu=None, densify=None, n_init=None, n_bg=0,
        seed=0, device=None, scene="data/spheres"):
    """Fit and save to `out`. n is the number of scene Gaussians (the budget with densify);
    n_bg background Gaussians at infinity are added on top. Returns the metrics dict."""
    device = device or get_device()
    torch.manual_seed(seed)
    train, val = load_scene(scene, device)
    n_init = (n_init or n // 4) if densify else n
    params = init_gaussians_3d(n_init, device, seed=seed, n_bg=n_bg)

    t0 = time.time()
    history, counts = fit(params, train, iters, lr, lr_mu, densify, budget=n + n_bg, seed=seed)
    seconds = time.time() - t0

    init = "uniform in [-1.5, 1.5]^3, scale 0.08, identity rotation, gray, opacity 0.12"
    if n_bg:
        init += f"; plus {n_bg} background Gaussians on a radius-1000 sphere, opacity 0.88"
    meta = {"n": n, "n_bg": n_bg, "densify": bool(densify), "n_init": n_init, "iters": iters,
            "lr": lr, "lr_mu": lr_mu if lr_mu is not None else lr, "seed": seed,
            "device": device, "seconds": round(seconds, 1), "init": init}
    if densify:
        meta["densify_config"] = vars(densify)
    m = save_results(Path(out), params, train, val, history, counts, meta)
    print(f"N={m['final_n']}  train PSNR {m['train_psnr_mean']:.2f} dB  "
          f"val PSNR {m['val_psnr_mean']:.2f} dB  ({seconds:.0f}s)  -> {out}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=4000, help="number of Gaussians (budget with --densify)")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lr-mu", type=float, default=None, help="lr for positions (default: --lr)")
    ap.add_argument("--n-bg", type=int, default=0, help="extra background Gaussians at infinity")
    ap.add_argument("--densify", action="store_true", help="P8: start from --n-init and grow to --n")
    ap.add_argument("--n-init", type=int, default=None, help="starting count with --densify (default n/4)")
    ap.add_argument("--densify-every", type=int, default=200)
    ap.add_argument("--densify-stop", type=int, default=1000)
    ap.add_argument("--top-frac", type=float, default=0.35)
    ap.add_argument("--prune-opacity", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    densify = None
    if args.densify:
        densify = densify_config_3d(args.densify_every, args.densify_stop, args.top_frac,
                                    args.prune_opacity)
    run(args.out, args.n, args.iters, args.lr, args.lr_mu, densify, args.n_init, args.n_bg,
        args.seed, args.device)


if __name__ == "__main__":
    main()
