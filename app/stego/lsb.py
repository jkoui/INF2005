# generic byte-wise LSB embed/extract

import numpy as np
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

    vals = flat[byte_idx].astype(np.uint16)
    mask_clear = ~(1 << bit_pos)
    vals = vals & mask_clear
    vals = vals | ((data_bits.astype(np.uint16) & 1) << bit_pos)
    flat[byte_idx] = vals.astype(np.uint8)
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
