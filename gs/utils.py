import numpy as np
import torch
from PIL import Image


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_image(path, device="cpu", size=None):
    """Load an image as a float tensor (H, W, 3) in [0, 1], optionally resized to size x size."""
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)


def save_image(img, path):
    """Save a float tensor (H, W, 3) in [0, 1] as an 8-bit PNG."""
    arr = (img.detach().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def mse_to_psnr(mse):
    """PSNR in dB for images in [0, 1] (MAX = 1)."""
    return -10.0 * np.log10(float(mse))


def psnr(img, target):
    return mse_to_psnr(((img.detach() - target) ** 2).mean().item())
