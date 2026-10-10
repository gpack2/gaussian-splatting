"""Assemble the writeup figures from finished runs into outputs/figures/.

    python make_figures.py

Needs the outputs of sweep_2d.py and sweep_3d.py. Each figure is a labeled grid of
renders; images are upscaled with nearest-neighbor so pixels stay visible.
"""
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from gs.model3d import render_gaussians_3d
from gs.scene import load_scene, look_at_camera
from gs.utils import get_device, load_image, psnr

OUT = Path("outputs")
FIG = OUT / "figures"
IMAGES = ["coffee", "astronaut", "cat"]
COUNTS = [256, 1024, 4096]
RUNS_3D = {"p7/cloud": "cloud", "p7/cloud_bg": "cloud + shell",
           "p8/densify": "densified", "p8/densify_bg": "densified + shell"}


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


def to_pil(img, scale):
    arr = (img.detach().clamp(0, 1).cpu().numpy() * 255 + 0.5).astype("uint8")
    im = Image.fromarray(arr)
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def grid(cells, col_titles, row_titles=None, pad=6, title_h=22, label_w=0):
    """cells: rows of (PIL image, caption under it) pairs."""
    w, h = cells[0][0][0].size
    cap_h = 20
    label_w = label_w if row_titles else 0
    W = label_w + len(col_titles) * (w + pad) + pad
    H = title_h + len(cells) * (h + cap_h + pad) + pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    f, fs = font(15), font(13)
    for j, t in enumerate(col_titles):
        draw.text((label_w + pad + j * (w + pad) + w // 2, title_h // 2), t, fill="black", font=f, anchor="mm")
    for i, row in enumerate(cells):
        y = title_h + i * (h + cap_h + pad)
        if row_titles:
            draw.text((label_w // 2, y + h // 2), row_titles[i], fill="black", font=f, anchor="mm")
        for j, (im, cap) in enumerate(row):
            x = label_w + pad + j * (w + pad)
            canvas.paste(im, (x, y))
            if cap:
                draw.text((x + w // 2, y + h + cap_h // 2), cap, fill="black", font=fs, anchor="mm")
    return canvas


def metrics(path):
    p = Path(path) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def render_2d(run_dir):
    from gs.model2d import render_gaussians_2d
    params = torch.load(Path(run_dir) / "gaussians.pt")
    m = metrics(run_dir)
    return render_gaussians_2d(params, m["size"], m["size"]), m


@torch.no_grad()
def fig_2d():
    # P4: target | fixed | densified at each N, one row per image.
    for n in COUNTS:
        rows = []
        for image in IMAGES:
            fixed, mf = render_2d(OUT / "p5" / f"{image}_{n}")
            dens, md = render_2d(OUT / "p4" / f"{image}_{n}")
            target = load_image(f"data/2d/{image}.png", size=mf["size"])
            rows.append([(to_pil(target, 2), image),
                         (to_pil(fixed, 2), f"{mf['final_psnr']:.2f} dB"),
                         (to_pil(dens, 2), f"{md['final_psnr']:.2f} dB")])
        grid(rows, ["target", f"fixed N={n}", f"densified N={n}"]).save(FIG / f"p4_densify_{n}.png")

    # P5: target and fixed fits at every N.
    rows = []
    for image in IMAGES:
        row = [(to_pil(load_image(f"data/2d/{image}.png", size=128), 2), image)]
        for n in COUNTS:
            img, m = render_2d(OUT / "p5" / f"{image}_{n}")
            row.append((to_pil(img, 2), f"{m['final_psnr']:.2f} dB"))
        rows.append(row)
    grid(rows, ["target"] + [f"N={n}" for n in COUNTS]).save(FIG / "p5_renders.png")


@torch.no_grad()
def fig_3d():
    device = get_device()
    train, val = load_scene("data/spheres", device)
    runs = {name: {k: v.to(device) for k, v in torch.load(OUT / name / "gaussians.pt").items()}
            for name in RUNS_3D if (OUT / name / "gaussians.pt").exists()}

    def compare(cams, names, path, show_psnr=True):
        rows = []
        for cam in cams:
            row = [(to_pil(cam.image, 2), Path(cam.name).as_posix())]
            for name in names:
                r = render_gaussians_3d(runs[name], cam)
                row.append((to_pil(r, 2), f"{psnr(r, cam.image):.2f} dB" if show_psnr else ""))
            rows.append(row)
        grid(rows, ["ground truth"] + [RUNS_3D[n] for n in names]).save(path)

    p7 = [n for n in ("p7/cloud", "p7/cloud_bg") if n in runs]
    compare([train[i] for i in (0, 12, 33)], p7, FIG / "p7_train.png")
    compare([val[i] for i in (1, 3, 9)], list(runs), FIG / "p8_val.png")

    # Floaters: the scene Gaussians alone (background shell removed) on held-out views.
    for name in ("p7/cloud_bg", "p8/densify_bg"):
        if name in runs:
            p = runs[name]
            keep = p["mu"].norm(dim=-1) < 100
            runs[name + "/scene"] = {k: v[keep] for k, v in p.items()}
    names = [n for n in ("p7/cloud_bg/scene", "p8/densify_bg/scene") if n in runs]
    if names:
        RUNS_3D.update({"p7/cloud_bg/scene": "cloud + shell, scene only",
                        "p8/densify_bg/scene": "densified + shell, scene only"})
        compare([val[i] for i in (1, 3, 9)], names, FIG / "p8_scene_only.png", show_psnr=False)

    # P9: views outside the training elevations (-5 to 55 degrees), no ground truth.
    best = runs.get("p8/densify_bg")
    if best is not None:
        K, H, W = train[0].K, train[0].H, train[0].W
        radius = train[0].center.norm().item()
        row = []
        for el in (-35, -15, 25, 70, 85):
            cam = look_at_camera(30.0, el, radius, K, H, W)
            row.append((to_pil(render_gaussians_3d(best, cam), 2), f"elevation {el}°"))
        grid([row], ["below training", "", "inside training range", "above training", ""]).save(
            FIG / "p9_extrapolate.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_2d()
    fig_3d()
    print(f"-> {FIG}")
