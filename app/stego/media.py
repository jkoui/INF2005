# image + wav I/O, capacity, previews

import base64
import io
import wave
from typing import Tuple

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

# -------- WAV (byte-level) --------
def load_wav_from_file(file_storage):
    """
    Returns:
      flat_bytes (np.uint8 1-D),
      params (dict with nchannels, sampwidth, framerate, nframes, comptype, compname),
      raw_frames (bytes)
    """
    with wave.open(file_storage.stream, "rb") as wf:
        params = dict(
            nchannels=wf.getnchannels(),
            sampwidth=wf.getsampwidth(),
            framerate=wf.getframerate(),
            nframes=wf.getnframes(),
            comptype=wf.getcomptype(),
            compname=wf.getcompname(),
        )
        if params["comptype"] != "NONE":
            raise ValueError("Only uncompressed PCM WAV is supported.")
        if params["sampwidth"] not in (1, 2):
            raise ValueError("Only 8-bit or 16-bit PCM WAV is supported.")
        frames = wf.readframes(params["nframes"])
    flat = np.frombuffer(frames, dtype=np.uint8)  # byte-level view
    return flat.copy(), params, frames

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
    return flat_len * n_lsb  # byte-level capacity

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
