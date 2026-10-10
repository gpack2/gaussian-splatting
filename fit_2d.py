"""P3 / P4: fit a 2D image with N Gaussians, optionally with densification.

Examples:
    python fit_2d.py --image data/2d/coffee.png --n 1024 --out outputs/p3/coffee_1024
    python fit_2d.py --image data/2d/coffee.png --n 1024 --densify --out outputs/p4/coffee_1024
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from gs.densify import DensifyConfig, GradStats, densify_and_prune
from gs.gaussian2d import rotation_2d
from gs.model2d import init_gaussians_2d, render_gaussians_2d
from gs.utils import get_device, load_image, mse_to_psnr, psnr, save_image


def densify_config_2d(every=100, stop=1500, top_frac=0.2, grad_threshold=2e-4):
    """P4 settings: a pass every 100 steps until step 1500, each densifying the top 20% of
    Gaussians by g_i (a fixed threshold does not transfer across N, since g_i ~ 1/N)."""
    return DensifyConfig(every=every, start=every, stop=stop, grad_threshold=grad_threshold,
                         top_frac=top_frac)


def make_optimizer(params, lr, lr_mu=None):
    groups = [{"params": [params["mu"]], "lr": lr_mu if lr_mu is not None else lr, "name": "mu"}]
    groups += [{"params": [params[k]], "lr": lr, "name": k}
               for k in ("log_s", "theta", "color", "op_raw")]
    return torch.optim.Adam(groups)


def fit(target, params, steps, lr, lr_mu=None, log_every=100, densify=None, budget=None):
    """Optimize params (in place) to reproduce target (H, W, 3).

    densify: a DensifyConfig to grow/prune the set up to `budget` Gaussians, or None
    for a fixed count. Returns per-step PSNR and Gaussian-count histories.
    """
    H, W, _ = target.shape
    opt = make_optimizer(params, lr, lr_mu)
    # Score position gradients in normalized image units ([-1, 1] across the image, like
    # the 3DGS paper's screen-space gradients), so its threshold carries over.
    stats = GradStats(params["mu"].shape[0], target.device, grad_scale=max(H, W) / 2)
    history, counts = [], []
    t0 = time.time()
    for step in range(steps):
        img = render_gaussians_2d(params, H, W)
        loss = ((img - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if densify:
            stats.add(params["mu"].grad)
        opt.step()
        history.append(mse_to_psnr(loss.item()))
        counts.append(params["mu"].shape[0])

        if densify and densify.is_pass(step):
            info = densify_and_prune(params, opt, stats.score(), densify, budget,
                                     extent=W, rotation_fn=lambda p: rotation_2d(p["theta"]))
            stats.reset(info["total"], target.device)
            print(f"  densify @ {step}: +{info['cloned']} cloned, {info['split']} split, "
                  f"-{info['pruned']} pruned -> {info['total']} Gaussians", flush=True)
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:5d}  psnr {history[-1]:6.2f} dB  N {counts[-1]:5d}  "
                  f"({time.time() - t0:6.1f}s)", flush=True)
    return history, counts


def save_results(out, target, params, history, counts, meta):
    out.mkdir(parents=True, exist_ok=True)
    H, W, _ = target.shape
    with torch.no_grad():
        img = render_gaussians_2d(params, H, W)
    final = psnr(img, target)

    save_image(img, out / "render.png")
    save_image(torch.cat([target, img], dim=1), out / "compare.png")   # target | render
    torch.save({k: v.detach().cpu() for k, v in params.items()}, out / "gaussians.pt")

    n_final = params["mu"].shape[0]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(history)
    ax.set_xlabel("step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{meta['image']}, N={n_final}{' (densified)' if meta['densify'] else ''}: {final:.2f} dB")
    ax.grid(alpha=0.3)
    if meta["densify"]:
        ax2 = ax.twinx()
        ax2.plot(counts, color="gray", lw=1, ls="--")
        ax2.set_ylabel("# Gaussians", color="gray")
    fig.tight_layout()
    fig.savefig(out / "psnr_curve.png", dpi=120)
    plt.close(fig)

    metrics = {**meta, "final_psnr": final, "final_n": n_final, "num_values": n_final * 9}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run(image, n, out, steps=2000, lr=1e-2, lr_mu=0.1, densify=None, n_init=None, seed=0,
        device=None, size=None):
    """Fit one image and save everything to `out`. With a DensifyConfig, start from
    `n_init` (default n/8) Gaussians and grow to the budget `n`. Returns the metrics dict."""
    device = device or get_device()
    torch.manual_seed(seed)
    target = load_image(image, device, size)
    H, W, _ = target.shape
    n_init = (n_init or max(1, n // 8)) if densify else n
    params = init_gaussians_2d(n_init, H, W, device, seed=seed)

    t0 = time.time()
    history, counts = fit(target, params, steps, lr, lr_mu, densify=densify, budget=n)
    seconds = time.time() - t0

    meta = {"image": Path(image).stem, "size": H, "n": n, "densify": bool(densify), "n_init": n_init,
            "steps": steps, "lr": lr, "lr_mu": lr_mu, "seed": seed, "device": device,
            "seconds": round(seconds, 1)}
    if densify:
        meta["densify_config"] = vars(densify)
    metrics = save_results(Path(out), target, params, history, counts, meta)
    print(f"final PSNR {metrics['final_psnr']:.2f} dB  ({seconds:.0f}s)  -> {out}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--n", type=int, default=1024, help="number of Gaussians (the budget with --densify)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-2, help="Adam lr for every parameter")
    ap.add_argument("--lr-mu", type=float, default=0.1,
                    help="lr for positions, in pixels (the handout's 1e-2 moves them too slowly)")
    ap.add_argument("--densify", action="store_true", help="P4: start small and grow to --n")
    ap.add_argument("--n-init", type=int, default=None, help="starting count with --densify (default n/8)")
    ap.add_argument("--densify-every", type=int, default=100)
    ap.add_argument("--densify-stop", type=int, default=1500)
    ap.add_argument("--grad-threshold", type=float, default=DensifyConfig.grad_threshold)
    ap.add_argument("--top-frac", type=float, default=0.2,
                    help="densify the top fraction by g_i each pass (0 = use --grad-threshold)")
    ap.add_argument("--size", type=int, default=None, help="resize the target to size x size (e.g. 128)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    densify = None
    if args.densify:
        densify = densify_config_2d(args.densify_every, args.densify_stop, args.top_frac or None,
                                    args.grad_threshold)
    run(args.image, args.n, args.out, args.steps, args.lr, args.lr_mu, densify, args.n_init,
        args.seed, args.device, args.size)


if __name__ == "__main__":
    main()
