# MAGIC, bits↔bytes, header, PRNG, download MIME helper

import os
import struct
from typing import Tuple, Callable

import numpy as np
from werkzeug.utils import secure_filename

MAGIC = b"ACW1"

# ---- bits/bytes ----
def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        bits = np.pad(bits, (0, 8 - (bits.size % 8)))
    return np.packbits(bits.astype(np.uint8)).tobytes()

# ---- header ----
def build_header(payload: bytes, filename: str) -> bytes:
    """
    MAGIC(4) + PAYLOAD_LEN(4, BE) + NAME_LEN(1) + NAME(NAME_LEN, utf-8)
    NAME is the original filename, e.g. 'report_v3.pdf'
    """
    name = os.path.basename(filename or "recovered.bin")
    name_bytes = name.encode("utf-8")
    if len(name_bytes) > 255:
        raise ValueError("Filename too long (max 255 bytes in utf-8).")
    return MAGIC + struct.pack(">I", len(payload)) + struct.pack("B", len(name_bytes)) + name_bytes

def parse_header(bit_reader: Callable[[int], np.ndarray]) -> Tuple[int, str]:
    # fixed 9 bytes = 72 bits
    fixed_bits = bit_reader(72)
    fixed = bits_to_bytes(fixed_bits)
    if fixed[:4] != MAGIC:
        raise ValueError("Invalid magic / wrong key.")
    payload_len = struct.unpack(">I", fixed[4:8])[0]
    name_len = fixed[8]
    filename = ""
    if name_len:
        name_bits = bit_reader(name_len * 8)
        filename = bits_to_bytes(name_bits).decode("utf-8", errors="ignore")
    return payload_len, filename

# ---- PRNG ----
def prng_perm(total_slots: int, key: int) -> np.ndarray:
    rng = np.random.default_rng(seed=int(key) & 0xFFFFFFFF)
    return rng.permutation(total_slots)

# ---- download helper ----
def safe_download_name_and_mime(ext: str) -> Tuple[str, str]:
    mime = "application/octet-stream"
    name = f"recovered{ext or '.bin'}"
    name = secure_filename(name)
    table = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".zip": "application/zip",
        ".json": "application/json",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
    }
    mime = table.get(ext.lower(), mime)
    return name, mime
