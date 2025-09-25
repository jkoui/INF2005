import numpy as np
from PIL import Image, ImageDraw

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


def decode_mono(flat_bytes: np.ndarray, wav_params: dict) -> np.ndarray:
    """
    Convert interleaved PCM bytes → mono float32 in [-1, 1].
    Supports 8-bit unsigned PCM and 16-bit signed PCM (little-endian).
    """
    nch = wav_params["nchannels"]
    sw  = wav_params["sampwidth"]   # bytes per sample
    if sw == 1:
        # 8-bit PCM is unsigned in WAV: [0..255] with 128 as zero
        arr = np.frombuffer(flat_bytes, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sw == 2:
        # 16-bit PCM signed little-endian
        arr = np.frombuffer(flat_bytes, dtype="<i2").astype(np.float32)
        arr = arr / 32768.0
    else:
        raise ValueError(f"Unsupported sampwidth={sw*8} bit")

    if nch > 1:
        arr = arr.reshape(-1, nch).mean(axis=1)  # simple mono mixdown
    return np.clip(arr, -1.0, 1.0)

import numpy as np
from PIL import Image, ImageDraw

def _peak_envelope(x: np.ndarray, width_px: int):
    n = x.size
    if n <= 0:
        return np.zeros(width_px), np.zeros(width_px)
    step = max(1, n // width_px)
    mins = np.empty(width_px, dtype=np.float32)
    maxs = np.empty(width_px, dtype=np.float32)
    for i in range(width_px):
        s = i * step
        e = min(n, s + step)
        if s >= n:
            mins[i] = 0.0; maxs[i] = 0.0
        else:
            seg = x[s:e]
            mins[i] = seg.min(); maxs[i] = seg.max()
    return mins, maxs

def _draw_band(drw, y0, y1, x0, x1, fill):
    drw.rectangle((x0, y0, x1, y1), fill=fill)

def _draw_env(drw, mins, maxs, W, y_top, y_bottom, color, alpha=170, fill=True, width=2):
    mid = (y_top + y_bottom) // 2
    s = (y_bottom - y_top) * 0.45
    if fill:
        pts = []
        for x in range(W):
            y = int(mid - maxs[x]*s); pts.append((x, y))
        for x in range(W-1, -1, -1):
            y = int(mid - mins[x]*s); pts.append((x, y))
        drw.polygon(pts, fill=(*color, alpha))
    else:
        rgba = (*color, alpha)
        for x in range(W):
            y1 = int(mid - maxs[x]*s)
            y2 = int(mid - mins[x]*s)
            drw.line((x, y1, x, y2), fill=rgba, width=width)

def waveform_compare_stacked(
    mono_cover: np.ndarray,
    mono_stego: np.ndarray,
    *,
    framerate: int,
    start_sample: int = 0,
    num_samples: int | None = None,
    width: int = 1100,
    lane_h: int = 180,       # height of each lane
    lane_gap: int = 28,      # gap between lanes
    show_diff: bool = True,
    diff_gain: float | str = "auto",   # "auto" rescales residual to ~40% lane height
    highlight_ms: tuple[float, float] | None = None,
    bg=(255,255,255),
) -> Image.Image:
    """Three synchronized lanes: Cover (gray), Stego (red), Residual (blue)."""
    N = min(mono_cover.size, mono_stego.size)
    start_sample = max(0, min(N, start_sample))
    num_samples = N - start_sample if (num_samples is None) else max(1, min(N - start_sample, num_samples))

    c = mono_cover[start_sample:start_sample + num_samples]
    s = mono_stego[start_sample:start_sample + num_samples]
    d = (s - c) if show_diff else None

    # auto diff gain so residual occupies ~40% of lane height
    if show_diff and isinstance(diff_gain, str) and diff_gain == "auto":
        m = np.percentile(np.abs(d), 95) if d.size else 0.0
        diff_gain = 0.4 / max(1e-6, float(m))
    if show_diff and isinstance(diff_gain, (int, float)):
        d = np.clip(d * float(diff_gain), -1.0, 1.0)

    W = width
    H = lane_h*3 + lane_gap*2
    img = Image.new("RGBA", (W, H), (*bg, 255))
    drw = ImageDraw.Draw(img, "RGBA")

    # lanes’ y ranges
    lanes = [
        ("Cover", (0, lane_h-1), (120,120,120)),      # gray
        ("Stego", (lane_h+lane_gap, 2*lane_h+lane_gap-1), (230,60,60)),  # red
        ("Residual", (2*(lane_h+lane_gap), 3*lane_h+2*lane_gap-1), (50,90,220)),  # blue
    ]

    # highlight band (ROI)
    if highlight_ms is not None:
        ms0, ms1 = highlight_ms
        s0 = int(ms0 * framerate / 1000.0)
        s1 = int(ms1 * framerate / 1000.0)
        def samp_to_x(n): return int((n - start_sample) / max(1, num_samples) * W)
        x0, x1 = sorted((samp_to_x(s0), samp_to_x(s1)))
        for _, (y0, y1), _ in lanes:
            _draw_band(drw, y0, y1, x0, x1, fill=(255,215,0,60))

    # grid + midlines
    for _, (y0, y1), _ in lanes:
        mid = (y0 + y1)//2
        drw.line((0, mid, W, mid), fill=(210,210,210,255))
        for gx in range(0, W, 120):
            drw.line((gx, y0, gx, y1), fill=(240,240,240,255))

    # envelopes
    cmin, cmax = _peak_envelope(c, W)
    smin, smax = _peak_envelope(s, W)
    if show_diff:
        dmin, dmax = _peak_envelope(d, W)

    # draw each lane
    _draw_env(drw, cmin, cmax, W, *lanes[0][1], color=lanes[0][2], fill=True)
    _draw_env(drw, smin, smax, W, *lanes[1][1], color=lanes[1][2], fill=True)
    if show_diff:
        _draw_env(drw, dmin, dmax, W, *lanes[2][1], color=lanes[2][2], fill=False, width=3)

    # labels (left titles)
    for name, (y0, y1), col in lanes:
        drw.text((8, y0+6), name, fill=(*col, 255))

    # footer time label
    ms0 = int(1000 * start_sample / max(1, framerate))
    dur = int(1000 * num_samples  / max(1, framerate))
    drw.text((8, H-18), f"{ms0}–{ms0+dur} ms  (window {dur} ms)", fill=(0,0,0,200))

    return img

