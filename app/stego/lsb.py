# generic byte-wise LSB embed/extract

import numpy as np
import struct
from .core import prng_perm

def embed_bits_into_bytes(flat: np.ndarray, data_bits: np.ndarray, n_lsb: int, key: int) -> np.ndarray:
    flat = flat.copy()
    total_slots = flat.size * n_lsb
    if data_bits.size > total_slots:
        raise ValueError("Payload too large for selected LSBs / cover.")

    perm = prng_perm(total_slots, key)
    tgt = perm[:data_bits.size]
    byte_idx = tgt // n_lsb
    bit_pos  = (tgt % n_lsb).astype(np.uint8)

    # Accumulate per-byte clear/set masks
    clear_mask = np.full(flat.size, 0xFF, dtype=np.uint8)
    set_mask   = np.zeros(flat.size,     dtype=np.uint8)

    # For each occurrence: clear that bit in the byte, and OR in the desired value
    np.bitwise_and.at(clear_mask, byte_idx, np.uint8(~(1 << bit_pos)))
    np.bitwise_or.at( set_mask,   byte_idx, np.uint8(((data_bits & 1).astype(np.uint8)) << bit_pos))

    # Apply masks once per byte
    flat = (flat & clear_mask) | set_mask
    return flat

def extract_bits_from_bytes(flat: np.ndarray, n_lsb: int, key: int, num_bits: int) -> np.ndarray:
    total_slots = flat.size * n_lsb
    if num_bits > total_slots:
        raise ValueError("Requested bits exceed capacity.")
    perm = prng_perm(total_slots, key)
    tgt = perm[:num_bits]
    byte_idx = tgt // n_lsb
    bit_pos  = (tgt % n_lsb).astype(np.uint8)
    vals = flat[byte_idx]
    return ((vals >> bit_pos) & 1).astype(np.uint8)

# Small, plain locator written sequentially at file start.
LOC_MAGIC = b'ASTG'   # 4 bytes
LOC_VER   = 1
LOC_FMT   = "<4sBII"  # magic, ver, start_byte, end_byte
LOC_LEN   = struct.calcsize(LOC_FMT)  # 13 bytes

def build_locator(start_byte: int, end_byte: int) -> bytes:
    return struct.pack(LOC_FMT, LOC_MAGIC, LOC_VER, start_byte, end_byte)

def parse_locator(b: bytes) -> tuple[int, int]:
    m, ver, s, e = struct.unpack(LOC_FMT, b)
    if m != LOC_MAGIC or ver != LOC_VER:
        raise ValueError("No locator prefix found.")
    return s, e

# Sequential bit I/O (no PRNG) just for the tiny locator prefix
def write_bits_sequential(flat: np.ndarray, n_lsb: int, bits: np.ndarray, start_byte: int = 0):
    for i in range(bits.size):
        b_ix = start_byte + (i // n_lsb)
        bitpos = i % n_lsb
        v = int(bits[i]) & 1
        cur = int(flat[b_ix])
        cur = (cur & ~(1 << bitpos)) | (v << bitpos)
        flat[b_ix] = cur & 0xFF

def read_bits_sequential(flat: np.ndarray, n_lsb: int, num_bits: int, start_byte: int = 0) -> np.ndarray:
    out = np.empty(num_bits, dtype=np.uint8)
    for i in range(num_bits):
        b_ix = start_byte + (i // n_lsb)
        bitpos = i % n_lsb
        out[i] = (flat[b_ix] >> bitpos) & 1
    return out

# --- sequential embedding/extraction (diagnostic) ---
def embed_bits_into_bytes_seq(flat: np.ndarray, data_bits: np.ndarray, n_lsb: int) -> np.ndarray:
    flat = flat.copy()
    need_bytes = (data_bits.size + n_lsb - 1) // n_lsb
    if need_bytes > flat.size:
        raise ValueError("Payload too large for selected LSBs / cover.")
    vals = flat[:need_bytes].astype(np.uint16)
    for i in range(data_bits.size):
        b_ix = i // n_lsb
        bitpos = i % n_lsb
        v = int(data_bits[i]) & 1
        cur = int(vals[b_ix])
        cur = (cur & ~(1 << bitpos)) | (v << bitpos)
        vals[b_ix] = cur
    flat[:need_bytes] = vals.astype(np.uint8)
    return flat

def extract_bits_from_bytes_seq(flat: np.ndarray, n_lsb: int, num_bits: int) -> np.ndarray:
    need_bytes = (num_bits + n_lsb - 1) // n_lsb
    if need_bytes > flat.size:
        raise ValueError("Requested bits exceed capacity.")
    out = np.empty(num_bits, dtype=np.uint8)
    for i in range(num_bits):
        b_ix = i // n_lsb
        bitpos = i % n_lsb
        out[i] = (flat[b_ix] >> bitpos) & 1
    return out
