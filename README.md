# A2: Gaussian Splatting (15-474)

Differentiable 2D and 3D Gaussian splatting in PyTorch.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The 2D targets are in `data/2d/`. For 3D, unzip
[spheres.zip](https://graphics.cs.cmu.edu/courses/15474/f26/assets/scenes/spheres.zip) into `data/`.

## Reproducing the results

```bash
python sweep_2d.py      # P4 + P5: 2D fits at 128x128 (~15 min)  -> outputs/p4, outputs/p5
python sweep_3d.py      # P7 - P9: 3D fits + evaluation (~10 min) -> outputs/p7, outputs/p8
python make_figures.py  # writeup figures                         -> outputs/figures
```

Single runs: `python fit_2d.py --image data/2d/coffee.png --n 1024 --size 128 --out <dir>`
(add `--densify` for P4) and `python fit_3d.py --lr 2e-2 --lr-mu 3e-3 --n-bg 1000 --out <dir>`
(add `--densify` for P8).

Tests: `python -m tests.test_p1_p2 && python -m tests.test_densify && python -m tests.test_p6`

## Code

- `gs/gaussian2d.py`, `gs/render.py`, `gs/model2d.py`: P1–P3 (covariance, tiled rasterizer, 2D model)
- `gs/densify.py`: P4/P8 clone, split and prune, shared by 2D and 3D
- `gs/gaussian3d.py`, `gs/model3d.py`, `gs/scene.py`: P6/P7 (projection, 3D model, cameras)
- `eval_3d.py`: P9 held-out PSNR, error maps and orbit GIF
