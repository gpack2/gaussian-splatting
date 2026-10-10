"""P7 + P8 (+ P9 evals): the four 3D runs and their comparison.

    python sweep_3d.py             # run whatever is missing, evaluate, and compare
    python sweep_3d.py --compare-only

Runs (scene Gaussians, plus 1000 background Gaussians at infinity for the *_bg runs):
  outputs/p7/cloud        handout init, fixed 4000 Gaussians
  outputs/p7/cloud_bg     + background shell
  outputs/p8/densify      start from 1000, densify/prune up to 4000
  outputs/p8/densify_bg   + background shell
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from fit_3d import densify_config_3d, run
from gs.model3d import render_gaussians_3d
from gs.scene import load_scene
from gs.utils import get_device, save_image

N, N_BG = 4000, 1000
LR, LR_MU = 2e-2, 3e-3   # best held-out PSNR among the lr settings tried (30.5 dB with the shell)
RUNS = {
    "p7/cloud": dict(densify=False, n_bg=0),
    "p7/cloud_bg": dict(densify=False, n_bg=N_BG),
    "p8/densify": dict(densify=True, n_bg=0),
    "p8/densify_bg": dict(densify=True, n_bg=N_BG),
}
OUT = Path("outputs")
SHOW_VAL = [1, 3, 9]      # held-out views for the side-by-side figure


def metrics(name):
    path = OUT / name / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


@torch.no_grad()
def compare():
    lines = ["| run | background shell | densify | scene Gaussians (final) | train PSNR | held-out PSNR | gap | time |",
             "|---|---|---|---|---|---|---|---|"]
    for name, cfg in RUNS.items():
        m = metrics(name)
        if m:
            lines.append(f"| {name} | {'yes' if cfg['n_bg'] else 'no'} | {'yes' if cfg['densify'] else 'no'} | "
                         f"{m['final_n'] - m.get('n_bg', 0)} | {m['train_psnr_mean']:.2f} | "
                         f"{m['val_psnr_mean']:.2f} | {m['train_psnr_mean'] - m['val_psnr_mean']:.2f} | "
                         f"{m['seconds']:.0f}s |")
    (OUT / "p8" / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Same held-out views for every run: GT | cloud | densify, then the same with the shell.
    device = get_device()
    _, val = load_scene("data/spheres", device)
    rows = []
    for i in SHOW_VAL:
        cells = [val[i].image]
        for name in RUNS:
            p = OUT / name / "gaussians.pt"
            if p.exists():
                params = {k: v.to(device) for k, v in torch.load(p).items()}
                cells.append(render_gaussians_3d(params, val[i]))
        rows.append(torch.cat(cells, dim=1))
    save_image(torch.cat(rows, dim=0), OUT / "p8" / "val_side_by_side.png")
    print("columns: ground truth | " + " | ".join(RUNS) + f"   (held-out views {SHOW_VAL})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()

    if not args.compare_only:
        for name, cfg in RUNS.items():
            if metrics(name):
                continue
            print(f"=== {name}", flush=True)
            run(OUT / name, n=N, lr=LR, lr_mu=LR_MU, n_bg=cfg["n_bg"],
                densify=densify_config_3d() if cfg["densify"] else None)
            subprocess.run([sys.executable, "eval_3d.py", "--run", str(OUT / name)], check=True)
    (OUT / "p8").mkdir(parents=True, exist_ok=True)
    compare()


if __name__ == "__main__":
    main()
