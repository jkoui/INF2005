#stego header code

# [ HEADER ] || [ CIPHERTEXT + TAG ]

import os, struct
from dataclasses import dataclass
from typing import Tuple, Callable
import numpy as np
from werkzeug.utils import secure_filename

MAGIC = b"ACW1" 
VER   = 1

def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        bits = np.pad(bits, (0, 8 - (bits.size % 8)))
    return np.packbits(bits.astype(np.uint8)).tobytes()

def prng_perm(total_slots: int, key: int) -> np.ndarray:
    """
    Deterministic permutation for a given length and integer key.
    - No global RNG state
    - Stable across processes/platforms
    - Works for any positive/negative key
    """
    # Normalize the key into an unsigned 64-bit seed (avoid sign/overflow quirks)
    seed = np.uint64(key & 0xFFFFFFFFFFFFFFFF)
    rng = np.random.default_rng(seed)      # local generator (doesn't touch global state)
    return rng.permutation(total_slots).astype(np.int64)

def safe_download_name_and_mime(ext: str):
    mime = "application/octet-stream"
    name = secure_filename(f"recovered{ext or '.bin'}")
    table = {
        ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".gif": "image/gif", ".bmp": "image/bmp",
        ".txt": "text/plain", ".csv": "text/csv", ".zip": "application/zip",
        ".json": "application/json", ".wav": "audio/wav", ".mp3": "audio/mpeg",
    }
    return name, table.get(ext.lower(), mime)

# -------- Secure header layout --------
# MAGIC(4) unique identifier for extracting the right stego object, like a signature
# VER(1) | version
# FLAGS(1) | for other purposes
# LSB(1) | use how LSB to embed
# START_OFFSET(8) | for advance embedding since not always start at position 0
# PLAIN_LEN(4) | length of payload
# NAME_LEN(1) | length of filename
# NONCE(12) | vector for AES-GCM encryption
# SALT(16) | used in KDF (key derivation func) to derive AES key
# SHA256(32) | integrity check -> hash of plaintext
# NAME(NAME_LEN, utf-8) | filename


FIXED_LEN = 96   # bytes (without NAME)

@dataclass
class StegoHeader:
    ver: int
    flags: int
    lsb_used: int
    start_offset: int
    plain_len: int
    name: str
    nonce: bytes
    salt: bytes
    sha256: bytes
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int

    def to_bytes(self) -> bytes:
        name_bytes = os.path.basename(self.name or "recovered.bin").encode("utf-8")
        if len(name_bytes) > 255:
            raise ValueError("Filename too long (max 255 bytes in utf-8).")
        fixed = (
            MAGIC +
            struct.pack("B", self.ver) +
            struct.pack("B", self.flags) +
            struct.pack("B", self.lsb_used) +
            self.start_offset.to_bytes(8, "big", signed=False) +
            self.plain_len.to_bytes(4, "big", signed=False) +
            struct.pack("B", len(name_bytes)) +
            self.nonce +
            self.salt +
            self.sha256 +
            int(self.roi_x1).to_bytes(4, "big", signed=False) +
            int(self.roi_y1).to_bytes(4, "big", signed=False) +
            int(self.roi_x2).to_bytes(4, "big", signed=False) +
            int(self.roi_y2).to_bytes(4, "big", signed=False)
        )
        return fixed + name_bytes

def build_secure_header(*, plain_len: int, filename: str, lsb_used: int,
                        start_offset: int, nonce: bytes, salt: bytes, sha256_bytes: bytes,
                        roi_x1: int = 0, roi_y1: int = 0, roi_x2: int = 0, roi_y2: int = 0) -> bytes:
    return StegoHeader(
        ver=VER, flags=0, lsb_used=int(lsb_used), start_offset=int(start_offset), 
        plain_len=int(plain_len), name=filename, nonce=nonce, salt=salt, sha256=sha256_bytes,
        roi_x1=int(roi_x1), roi_y1=int(roi_y1), roi_x2=int(roi_x2), roi_y2=int(roi_y2)
    ).to_bytes()

def parse_secure_header(bit_reader: Callable[[int], np.ndarray]) -> Tuple[StegoHeader, int]:
    """
    Works with a stateless bit_reader (each call starts at bit 0).
    First read the fixed part, then read (fixed + name_len) and slice the name.
    Returns (header_obj, total_header_bits).
    """
    # 1) Read fixed part
    fixed_bits = bit_reader(FIXED_LEN * 8)
    fixed = bits_to_bytes(fixed_bits)
    if fixed[:4] != MAGIC:
        raise ValueError("Invalid magic / wrong key.")

    ver   = fixed[4]
    flags = fixed[5]
    lsb   = fixed[6]
    start = int.from_bytes(fixed[7:15], "big")
    plen  = int.from_bytes(fixed[15:19], "big")
    nlen  = fixed[19]
    nonce = fixed[20:32]
    salt  = fixed[32:48]
    sha   = fixed[48:80]

    roi_x1 = int.from_bytes(fixed[80:84], "big")
    roi_y1 = int.from_bytes(fixed[84:88], "big")
    roi_x2 = int.from_bytes(fixed[88:92], "big")
    roi_y2 = int.from_bytes(fixed[92:96], "big")

    # Optional sanity checks
    if ver != VER:
        raise ValueError(f"Unsupported header version: {ver}")
    if lsb < 1 or lsb > 8:
        raise ValueError(f"Invalid LSB count in header: {lsb}")
    if nlen > 255:
        raise ValueError("Header filename too long.")
    if len(nonce) != 12 or len(salt) != 16 or len(sha) != 32:
        raise ValueError("Corrupt header crypto fields.")

    # 2) If there is a name, re-read (fixed + name) and slice the tail
    name = ""
    if nlen:
        full_bits = bit_reader((FIXED_LEN + nlen) * 8)
        full_bytes = bits_to_bytes(full_bits)
        name = full_bytes[FIXED_LEN:FIXED_LEN + nlen].decode("utf-8", errors="ignore")

    hdr = StegoHeader(
        ver=ver, flags=flags, lsb_used=lsb, start_offset=start,
        plain_len=plen, name=name, nonce=nonce, salt=salt, sha256=sha,
        roi_x1=roi_x1, roi_y1=roi_y1, roi_x2=roi_x2, roi_y2=roi_y2
    )
    total_header_bits = (FIXED_LEN + nlen) * 8
    return hdr, total_header_bits

