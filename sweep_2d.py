"""P4 + P5: fit every 2D target at several Gaussian counts, with and without densification,
then plot PSNR vs. N and write a results table.

    python sweep_2d.py                 # run whatever is missing, then plot
    python sweep_2d.py --plot-only

Fixed-count fits go to outputs/p5/<image>_<N>/, densified ones to outputs/p4/<image>_<N>/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_2d import densify_config_2d, run

IMAGES = ["coffee", "astronaut", "cat"]
COUNTS = [256, 1024, 4096]
SIZE = 128     # targets downsampled from 256x256 (allowed per the course clarification)
OUT = Path("outputs")


def run_dir(image, n, densify):
    return OUT / ("p4" if densify else "p5") / f"{image}_{n}"


def load_metrics(image, n, densify):
    path = run_dir(image, n, densify) / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def plot_and_table():
    rows, fig = [], plt.figure(figsize=(6, 4))
    ax = fig.gca()
    for image, color in zip(IMAGES, ["tab:brown", "tab:blue", "tab:orange"]):
        for densify, style in [(False, "o-"), (True, "s--")]:
            ms = [m for n in COUNTS if (m := load_metrics(image, n, densify))]
            if ms:
                ax.plot([m["final_n"] for m in ms], [m["final_psnr"] for m in ms], style,
                        color=color, label=f"{image}{' (densified)' if densify else ''}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(COUNTS, [str(n) for n in COUNTS])
    ax.set_xlabel("number of Gaussians N  (model size ≈ 9N values)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"2D fits at {SIZE}x{SIZE}: quality vs. Gaussian count (2000 steps)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (OUT / "p5").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "p5" / "psnr_vs_n.png", dpi=150)

    lines = ["| image | N | fixed PSNR (dB) | densified PSNR (dB) | Δ (dB) |", "|---|---|---|---|---|"]
    for image in IMAGES:
        for n in COUNTS:
            f, d = load_metrics(image, n, False), load_metrics(image, n, True)
            fs = f"{f['final_psnr']:.2f}" if f else "—"
            ds = f"{d['final_psnr']:.2f}" if d else "—"
            delta = f"{d['final_psnr'] - f['final_psnr']:+.2f}" if f and d else "—"
            lines.append(f"| {image} | {n} | {fs} | {ds} | {delta} |")
            rows.append({"image": image, "n": n, "fixed": f and f["final_psnr"],
                         "densified": d and d["final_psnr"]})
    (OUT / "p5" / "results.md").write_text("\n".join(lines) + "\n")
    (OUT / "p5" / "results.json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-densify", action="store_true", help="only the fixed-count (P5) runs")
    args = ap.parse_args()

    if not args.plot_only:
        for densify in [False] if args.no_densify else [False, True]:
            for n in COUNTS:
                for image in IMAGES:
                    if load_metrics(image, n, densify):
                        continue
                    print(f"=== {image} N={n} {'densified' if densify else 'fixed'}", flush=True)
                    run(f"data/2d/{image}.png", n, run_dir(image, n, densify),
                        densify=densify_config_2d() if densify else None, size=SIZE)
    plot_and_table()


if __name__ == "__main__":
    main()
