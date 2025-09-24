import numpy as np
from PIL import Image

def _chi2_pairs_stat(bytes_1d: np.ndarray) -> float:
    """Westfeld chi-square statistic over 256-bin histogram, paired (0,1), (2,3), ..."""
    # fast bincount to 256 bins
    h = np.bincount(bytes_1d.astype(np.uint8), minlength=256)
    pairs = h.reshape(128, 2)
    s = pairs.sum(axis=1).astype(np.float64)
    d = (pairs[:, 0] - pairs[:, 1]).astype(np.float64)
    # avoid div by 0
    s[s == 0] = 1.0
    return np.sum((d * d) / s)

def chi_square_heatmap(
    cover_flat: np.ndarray,
    stego_flat: np.ndarray,
    shape: tuple,                  # (H, W)
    *,
    block: int = 16,               # block size (px)
    channel: int | None = None,    # None=RGB (all), 0/1/2 for R/G/B
    mode: str = "delta",           # "delta" (cover χ² − stego χ²) or "stego"
    alpha: float = 0.6,            # overlay strength (0..1)
    color=(255, 0, 0)              # heat color (red)
) -> Image.Image:
    """
    Returns a PIL.Image of the STEGO image with a chi-square heat overlay.
    - 'delta' mode: brighter blocks where stego χ² << cover χ² (suspicious).
    - 'stego' mode: shows raw stego χ² (low = suspicious). We invert for display.
    """
    H, W = shape
    C = 3
    cover = cover_flat.reshape(H, W, C).astype(np.uint8)
    stego = stego_flat.reshape(H, W, C).astype(np.uint8)

    # choose channels
    if channel is None:
        # average χ² over channels by concatenating bytes
        def block_bytes(img_blk):
            return img_blk.reshape(-1, C)
    else:
        def block_bytes(img_blk):
            return img_blk.reshape(-1, C)[:, [channel]]

    Hb = (H + block - 1) // block
    Wb = (W + block - 1) // block
    heat = np.zeros((Hb, Wb), dtype=np.float64)

    for by in range(Hb):
        y1 = by * block
        y2 = min(H, y1 + block)
        for bx in range(Wb):
            x1 = bx * block
            x2 = min(W, x1 + block)

            cov_blk = cover[y1:y2, x1:x2, :]
            stg_blk = stego[y1:y2, x1:x2, :]

            cov_bytes = block_bytes(cov_blk).reshape(-1)
            stg_bytes = block_bytes(stg_blk).reshape(-1)

            chi_cov = _chi2_pairs_stat(cov_bytes)
            chi_stg = _chi2_pairs_stat(stg_bytes)

            if mode == "delta":
                # larger positive value => pairs more equalized in stego => suspicious
                val = max(0.0, chi_cov - chi_stg)
            else:  # "stego"
                # smaller chi-square => more suspicious; invert for display
                val = 1.0 / (chi_stg + 1e-9)

            heat[by, bx] = val

    # robust normalize (percentiles) → 0..1
    lo, hi = np.percentile(heat, [5, 95])
    if hi <= lo:
        lo, hi = float(heat.min()), float(heat.max() + 1e-9)
    heat_norm = np.clip((heat - lo) / (hi - lo), 0.0, 1.0)

    # upsample block grid to image size (nearest keeps sharp block edges)
    heat_img = (heat_norm * 255.0).astype(np.uint8)
    heat_img = Image.fromarray(heat_img, mode="L").resize((W, H), resample=Image.NEAREST)
    heat_arr = np.array(heat_img, dtype=np.float32) / 255.0  # 0..1

    # colorize as single-hue heat (red) and alpha-blend over STEGO
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[..., 0] = int(color[0])  # R
    overlay[..., 1] = int(color[1])  # G
    overlay[..., 2] = int(color[2])  # B

    stg = stego.astype(np.float32)
    ov  = overlay.astype(np.float32)
    a = (alpha * heat_arr)[..., None]  # per-pixel alpha 0..alpha
    out = (stg * (1.0 - a) + ov * a).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")