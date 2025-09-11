import os, hashlib
from typing import Optional, Tuple
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PBKDF2_ITERS = 200_000 # resists brute force
KEY_LEN      = 32      # derived AES-256 bit key
NONCE_LEN    = 12      # AES-GCM standard
SALT_LEN     = 16      # recommended length for PBKDF2
TAG_LEN      = 16      # AES-GCM tag is 16 bytes

# converts an integer into a fixed 8 byte big-endian representation
def _int_to_8bytes(i: int) -> bytes:
    return int(i).to_bytes(8, "big", signed=False)

# builds a PBKDF2-HMAC-SHA256 KDF that outputs 32 bytes (AES-256 key)
def derive_key(user_key_int: int, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=KEY_LEN, salt=salt, iterations=PBKDF2_ITERS)
    return kdf.derive(_int_to_8bytes(user_key_int))

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

# creates a fresh random nonce for the message
def encrypt_aes_gcm(key: bytes, plaintext: bytes, aad: Optional[bytes] = None) -> Tuple[bytes, bytes]:
    nonce = os.urandom(NONCE_LEN)
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)   # ciphertext || tag
    return nonce, ct

def decrypt_aes_gcm(key: bytes, nonce: bytes, ciphertext_with_tag: bytes, aad: Optional[bytes] = None) -> bytes:
    return AESGCM(key).decrypt(nonce, ciphertext_with_tag, aad)
