# image + wav I/O, capacity, previews

import base64
import io
import wave
from typing import Tuple
import cv2
import tempfile, shutil
import os
import numpy as np
from PIL import Image

# -------- Images --------
def load_image_from_file(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    arr = np.array(img, dtype=np.uint8)   # H,W,3
    h, w, _ = arr.shape
    flat = arr.reshape(-1)
    return flat, (h, w)

def save_image_to_bytes(flat: np.ndarray, shape: Tuple[int,int]) -> bytes:
    h, w = shape
    arr = flat.reshape(h, w, 3).astype(np.uint8)
    out = Image.fromarray(arr, mode="RGB")
    bio = io.BytesIO()
    out.save(bio, format="PNG")
    bio.seek(0)
    return bio.read()

def image_capacity_bits(flat_len: int, n_lsb: int) -> int:
    return flat_len * n_lsb

def image_lsb_plane(flat: np.ndarray, n_lsb: int, shape: Tuple[int,int]) -> Image.Image:
    mask = (1 << n_lsb) - 1
    vals = (flat & mask).astype(np.uint16)
    scaled = (vals * (255 // mask if mask else 1)).astype(np.uint8)
    h, w = shape
    img = scaled.reshape(h, w, 3)
    gray = np.mean(img, axis=2).astype(np.uint8)
    return Image.fromarray(gray, mode="L")

def image_diff(original_flat: np.ndarray, stego_flat: np.ndarray, shape: Tuple[int,int]) -> Image.Image:
    diff = np.abs(stego_flat.astype(np.int16) - original_flat.astype(np.int16)).astype(np.uint8)
    h, w = shape
    img = diff.reshape(h, w, 3)
    img = np.clip(img * 64, 0, 255).astype(np.uint8)  # emphasize changes
    return Image.fromarray(img, mode="RGB")

def img_to_data_url(pil_img: Image.Image) -> str:
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    return "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode("ascii")

def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Very simple 3x3 dilation without extra deps.
    mask: HxW boolean
    """
    if iterations <= 0:
        return mask
    m = mask
    H, W = m.shape
    for _ in range(iterations):
        # pad to avoid border checks
        p = np.pad(m, 1, mode='constant', constant_values=False)
        # 3x3 max filter via neighbor shifts
        d = (
            p[0:H,   0:W]   | p[0:H,   1:W+1] | p[0:H,   2:W+2] |
            p[1:H+1, 0:W]   | p[1:H+1, 1:W+1] | p[1:H+1, 2:W+2] |
            p[2:H+2, 0:W]   | p[2:H+2, 1:W+1] | p[2:H+2, 2:W+2]
        )
        m = d
    return m

def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    # Erode = dilate of the inverse, then invert
    if iterations <= 0: 
        return mask
    inv = ~mask
    inv_dil = dilate_mask(inv, iterations=iterations)
    return ~inv_dil

def make_change_overlay(
    cover_flat: np.ndarray,
    stego_flat: np.ndarray,
    shape: tuple,
    *,
    n_lsb: int = 1,
    color=(255, 32, 32),
    alpha: float = 0.6,
    dilate_px: int = 1,
    outline_only: bool = False,
    roi: tuple | None = None   # (x1,y1,x2,y2) to draw border only (optional)
) -> Image.Image:
    """
    Returns a PIL Image of the COVER with a colored overlay where LSBs changed.
    - cover_flat / stego_flat: flattened uint8 arrays length H*W*3
    - shape: (H, W)
    """
    H, W = shape
    C = 3
    cover = cover_flat.reshape(H, W, C).astype(np.uint8)
    stego = stego_flat.reshape(H, W, C).astype(np.uint8)

    # 1) Change mask for the LSB region
    diff = cover ^ stego
    lsb_mask_per_channel = (diff & ((1 << n_lsb) - 1)) != 0   # HxWxC bool
    mask = lsb_mask_per_channel.any(axis=2)                    # HxW bool

    # 2) Morphology: thicken or outline
    if dilate_px > 0:
        mask_d = dilate_mask(mask, iterations=dilate_px)
    else:
        mask_d = mask

    if outline_only:
        # Outline = dilated - eroded (or mask_d - erode(mask))
        inner = erode_mask(mask_d, iterations=max(1, dilate_px))
        mask_final = mask_d & (~inner)
    else:
        mask_final = mask_d

    # 3) Blend overlay color on COVER (not STEGO) so the artifacts pop
    out = cover.copy()
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]

    # alpha blend only where mask_final == True
    m = mask_final[:, :, None]  # HxWx1
    # out = (1 - alpha)*base + alpha*color on masked pixels
    out = np.where(m, 
                   (out.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8),
                   out)

    img = Image.fromarray(out, mode="RGB")

    return img

# -------- WAV (byte-level) --------
def load_wav_from_file(file_or_path):
    if hasattr(file_or_path, "stream"):
        wav = wave.open(file_or_path.stream, "rb")
    elif isinstance(file_or_path, (str, bytes, os.PathLike)):
        wav = wave.open(file_or_path, "rb")
    else:
        wav = wave.open(file_or_path, "rb")

    params = {
        "nchannels": wav.getnchannels(),
        "sampwidth": wav.getsampwidth(),
        "framerate": wav.getframerate(),
        "nframes": wav.getnframes(),
    }
    raw = wav.readframes(params["nframes"])
    wav.close()

    # store as bytes (carrier space!)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr, params, raw

def save_wav_to_bytes(flat: np.ndarray, params: dict) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(params["nchannels"])
        wf.setsampwidth(params["sampwidth"])
        wf.setframerate(params["framerate"])
        wf.writeframes(flat.tobytes())
    bio.seek(0)
    return bio.read()

def wav_capacity_bits(flat_len: int, n_lsb: int) -> int:
    return flat_len * n_lsb

def audio_to_data_url(wav_bytes: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")

def mean_abs_byte_delta(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16))))

# ---- single-bit bit-plane preview (0..7) ----
def image_bit_plane(flat: np.ndarray, shape: Tuple[int, int], bit: int = 0, channel: int | None = None) -> Image.Image:
    """
    bit: 0..7 (0 = LSB)
    channel: None -> average across RGB; 0/1/2 -> R/G/B only
    Returns grayscale PIL image with 0/255 values.
    """
    h, w = shape
    arr = flat.reshape(h, w, 3).astype(np.uint8)
    if channel is None:
        plane = ((arr >> bit) & 1).mean(axis=2) >= 0.5
        plane = (plane.astype(np.uint8) * 255)
    else:
        plane = (((arr[..., channel] >> bit) & 1) * 255).astype(np.uint8)
    return Image.fromarray(plane, mode="L")

# ---- LSB-change mask between cover and stego for a specific bit ----
def image_lsb_change_mask(original_flat: np.ndarray, stego_flat: np.ndarray, shape: Tuple[int,int], bit: int = 0) -> Image.Image:
    """
    Highlights pixels where any channel changed at 'bit'.
    Returns a binary (0/255) mask in grayscale.
    """
    h, w = shape
    a = original_flat.reshape(h, w, 3).astype(np.uint8)
    b = stego_flat.reshape(h, w, 3).astype(np.uint8)
    changed = (((a ^ b) >> bit) & 1).any(axis=2)  # True where selected bit differs
    mask = (changed.astype(np.uint8) * 255)
    return Image.fromarray(mask, mode="L")


# -------- MP4 (frame-level) --------
def load_video_from_file(file_or_path):
    """
    Reads video frames into flat bytes.
    Accepts either a Flask FileStorage or a path string.
    """
    # Case 1: Flask FileStorage
    if hasattr(file_or_path, "stream"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file_or_path.stream, tmp)
            tmp_path = tmp.name
    else:
        # Case 2: path string
        tmp_path = file_or_path

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()

    if hasattr(file_or_path, "stream"):
        os.remove(tmp_path)  # only delete if we created a temp file

    if not frames:
        raise ValueError("No frames loaded from video.")

    h, w, _ = frames[0].shape
    flat = np.concatenate([f.reshape(-1) for f in frames])
    meta = {"width": w, "height": h, "frames": len(frames), "fps": fps, "shape": (h, w)}
    return flat, meta, frames



def save_video_to_bytes(flat: np.ndarray, meta: dict) -> bytes:
    import tempfile, cv2, os

    h, w = meta["shape"]
    fps = int(meta["fps"])
    frames = []
    total = h * w * 3
    for i in range(meta["frames"]):
        start = i * total
        end = start + total
        frame = flat[start:end].reshape(h, w, 3).astype(np.uint8)
        frames.append(frame)

    # ---- use AVI with FFV1 codec (lossless) ----
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    tmp_path = tempfile.mktemp(suffix=".avi")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data



def video_capacity_bits(flat_len: int, n_lsb: int) -> int:
    return flat_len * n_lsb

