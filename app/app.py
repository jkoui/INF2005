import io
import os
from typing import Tuple
import os as _os

from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from PIL import Image
from werkzeug.utils import secure_filename


from stego.crypto import (
    derive_key, encrypt_aes_gcm, decrypt_aes_gcm, sha256, SALT_LEN, TAG_LEN
)

from stego.core import (
    MAGIC, bytes_to_bits, bits_to_bytes, safe_download_name_and_mime,
    build_secure_header, parse_secure_header
)

from stego.lsb import embed_bits_into_bytes, extract_bits_from_bytes

from stego.media import (
    load_image_from_file, save_image_to_bytes, image_capacity_bits,
    image_lsb_plane, image_diff, img_to_data_url,
    load_wav_from_file, save_wav_to_bytes, wav_capacity_bits,
    audio_to_data_url, mean_abs_byte_delta,
    image_bit_plane, image_lsb_change_mask
)

app = Flask(__name__)
app.secret_key = "acw1-secret"

# in-memory artifacts
_LAST_STEGO: bytes = b""
_LAST_STEGO_WAV: bytes = b""
_LAST_PAYLOAD: Tuple[bytes, str] = (b"", "recovered.bin")

@app.get("/")
def index():
    return render_template("index.html", result=None)

# -------- Image routes --------
@app.post("/embed")
def embed():
    global _LAST_STEGO
    try:
        cover = request.files.get("cover")
        payload = request.files.get("payload")
        n_lsb = int(request.form.get("lsb", "1"))
        key = int(request.form.get("key", "0"))

        if not cover or not payload:
            flash("Please provide both cover and payload files.")
            return redirect(url_for("index"))
        if n_lsb < 1 or n_lsb > 8:
            flash("LSBs must be between 1 and 8.")
            return redirect(url_for("index"))

        payload_bytes = payload.read()
        orig_name = os.path.basename(payload.filename or "payload.bin")

        # --- crypto ---
        salt = _os.urandom(SALT_LEN)
        key_bytes = derive_key(int(key), salt)
        nonce, ciphertext_with_tag = encrypt_aes_gcm(key_bytes, payload_bytes, aad=None)
        plain_sha = sha256(payload_bytes)

        # Secure header (store lsb_used; start_offset=0 for now)
        header = build_secure_header(
            plain_len=len(payload_bytes),
            filename=orig_name,
            lsb_used=n_lsb,
            start_offset=0,
            nonce=nonce,
            salt=salt,
            sha256_bytes=plain_sha,
        )

        # Embed header || ciphertext
        blob = header + ciphertext_with_tag
        data_bits = bytes_to_bits(blob)
        header_bytes_len = len(header)
        required_bytes = header_bytes_len + len(ciphertext_with_tag)

        cover.stream.seek(0)
        flat, shape = load_image_from_file(cover)
        cap_bits = image_capacity_bits(flat.size, n_lsb)
        cap_bytes = cap_bits // 8
        if data_bits.size > cap_bits:
            flash(f"Cover too small. Capacity={cap_bytes} bytes, Needed={required_bytes} bytes")
            return redirect(url_for("index"))

        stego_flat = embed_bits_into_bytes(flat, data_bits, n_lsb, key)
        stego_png = save_image_to_bytes(stego_flat, shape)
        _LAST_STEGO = stego_png

                # --- NEW: controls from form (with defaults) ---
        bit = int(request.form.get("bit", "0"))               # 0..7
        channel_map = {"all": None, "r": 0, "g": 1, "b": 2}
        channel = channel_map.get(request.form.get("channel", "all"), None)

        # --- existing previews you already build ---
        stego_img = Image.open(io.BytesIO(stego_png)).convert("RGB")
        stego_preview = img_to_data_url(stego_img)

        # --- NEW: single-bit plane and change mask previews ---
        bit_plane_img = image_bit_plane(stego_flat, shape, bit=bit, channel=channel)
        bit_plane_preview = img_to_data_url(bit_plane_img)

        change_mask_img = image_lsb_change_mask(flat, stego_flat, shape, bit=bit)
        change_mask_preview = img_to_data_url(change_mask_img)

        cover_img = Image.fromarray(flat.reshape(shape[0], shape[1], 3).astype("uint8"), mode="RGB")
        cover_preview = img_to_data_url(cover_img)
        cover_meta = f"{shape[1]}×{shape[0]} px, 3 channels (RGB)"

        stego_img = Image.open(io.BytesIO(stego_png)).convert("RGB")
        stego_preview = img_to_data_url(stego_img)
        lsb_img = image_lsb_plane(stego_flat, n_lsb, shape)
        lsb_preview = img_to_data_url(lsb_img)
        diff_img = image_diff(flat, stego_flat, shape)
        diff_preview = img_to_data_url(diff_img)

        utilization = round((required_bytes / cap_bytes) * 100, 2) if cap_bytes else 0.0
        capacity = {
            "cover_bytes": cap_bytes,
            "header_bytes": header_bytes_len,
            "payload_bytes": len(ciphertext_with_tag),
            "required_bytes": required_bytes,
            "utilization": utilization,
            "n_lsb": n_lsb,
            "hw": f"{shape[1]}×{shape[0]}",
            "wav_fmt": None,
        }

        result = {
            "cover_preview": cover_preview,
            "cover_meta": cover_meta,
            "stego_preview": stego_preview,
            "lsb_preview": lsb_preview,
            "bit_plane_preview": bit_plane_preview,      # NEW: precise single-bit plane
            "change_mask_preview": change_mask_preview,  # NEW: pixels changed at that bit
            "diff_preview": diff_preview,
            "capacity": capacity,
            "download_url": url_for("download_stego"),
        }
        return render_template("index.html", result=result)
    except Exception as e:
        flash(f"Embed error: {e}")
        return redirect(url_for("index"))

@app.get("/download/stego")
def download_stego():
    if not _LAST_STEGO:
        return redirect(url_for("index"))
    return send_file(io.BytesIO(_LAST_STEGO), as_attachment=True, download_name="stego.png", mimetype="image/png")

@app.post("/extract")
def extract():
    global _LAST_PAYLOAD
    try:
        stego = request.files.get("stego")
        n_lsb = int(request.form.get("lsb", "1"))
        key = int(request.form.get("key", "0"))
        if not stego:
            flash("Please provide a stego image.")
            return redirect(url_for("index"))

        stego.stream.seek(0)
        flat, shape = load_image_from_file(stego)

        # Reader that always starts from the top of the permutation
        def read_bits(nbits):
            return extract_bits_from_bytes(flat, n_lsb, key, nbits)

        # 1) Parse secure header to learn sizes/materials
        hdr, header_bits = parse_secure_header(read_bits)

        # 2) Ciphertext length = plaintext length + 16 (AES-GCM tag)
        cipher_len = hdr.plain_len + TAG_LEN
        total_bits = header_bits + cipher_len * 8

        # 3) Extract whole (header + ciphertext), then slice
        all_bits = extract_bits_from_bytes(flat, n_lsb, key, total_bits)
        all_bytes = bits_to_bytes(all_bits)
        header_bytes_len = header_bits // 8
        ciphertext_with_tag = all_bytes[header_bytes_len : header_bytes_len + cipher_len]

        # 4) Re-derive key & decrypt
        key_bytes = derive_key(int(key), hdr.salt)
        plaintext = decrypt_aes_gcm(key_bytes, hdr.nonce, ciphertext_with_tag, aad=None)

        # 5) Verify integrity (SHA-256 of plaintext)
        if sha256(plaintext) != hdr.sha256:
            raise ValueError("Integrity check failed (SHA-256 mismatch). Wrong key or corrupted data.")

        # 6) Save using original filename from header
        fname = secure_filename(hdr.name or "recovered.bin")
        if not os.path.splitext(fname)[1]:
            fname += ".bin"
        _LAST_PAYLOAD = (plaintext, fname)

        stego_img = Image.fromarray(flat.reshape(shape[0], shape[1], 3).astype("uint8"), mode="RGB")
        stego_preview = img_to_data_url(stego_img)
        result = {"payload_name": fname, "stego_preview": stego_preview}
        return render_template("index.html", result=result)

    except Exception as e:
        flash(f"Extract error: {e}")
        return redirect(url_for("index"))

# -------- Audio WAV routes --------
@app.post("/embed_audio")
def embed_audio():
    global _LAST_STEGO_WAV
    try:
        cover = request.files.get("cover_wav")
        payload = request.files.get("payload_wav")
        n_lsb = int(request.form.get("lsb", "1"))
        key = int(request.form.get("key", "0"))
        if not cover or not payload:
            flash("Please provide both cover WAV and payload files.")
            return redirect(url_for("index"))
        if n_lsb < 1 or n_lsb > 8:
            flash("LSBs must be between 1 and 8.")
            return redirect(url_for("index"))

        payload_bytes = payload.read()
        orig_name = os.path.basename(payload.filename or "payload.bin")

        salt = _os.urandom(SALT_LEN)
        key_bytes = derive_key(int(key), salt)
        nonce, ciphertext_with_tag = encrypt_aes_gcm(key_bytes, payload_bytes, aad=None)
        plain_sha = sha256(payload_bytes)

        header = build_secure_header(
            plain_len=len(payload_bytes),
            filename=orig_name,
            lsb_used=n_lsb,
            start_offset=0,
            nonce=nonce,
            salt=salt,
            sha256_bytes=plain_sha,
        )

        blob = header + ciphertext_with_tag
        data_bits = bytes_to_bits(blob)
        header_bytes_len = len(header)
        required_bytes = header_bytes_len + len(ciphertext_with_tag)

        cover.stream.seek(0)
        flat_bytes, wav_params, _ = load_wav_from_file(cover)
        cap_bits = wav_capacity_bits(flat_bytes.size, n_lsb)
        cap_bytes = cap_bits // 8
        if data_bits.size > cap_bits:
            flash(f"WAV too small. Capacity={cap_bytes} bytes, Needed={required_bytes} bytes")
            return redirect(url_for("index"))

        stego_flat = embed_bits_into_bytes(flat_bytes, data_bits, n_lsb, key)
        stego_wav = save_wav_to_bytes(stego_flat, wav_params)
        _LAST_STEGO_WAV = stego_wav

        cover_audio_preview = audio_to_data_url(save_wav_to_bytes(flat_bytes, wav_params))
        stego_audio_preview = audio_to_data_url(stego_wav)
        fmt = f"{wav_params['nchannels']} ch, {8*wav_params['sampwidth']}-bit, {wav_params['framerate']} Hz, {wav_params['nframes']} frames"
        audio_meta = fmt
        delta = mean_abs_byte_delta(flat_bytes, stego_flat)
        audio_diff_note = f"Mean absolute byte delta: {delta:.3f}"

        utilization = round((required_bytes / cap_bytes) * 100, 2) if cap_bytes else 0.0
        capacity = {
            "cover_bytes": cap_bytes,
            "header_bytes": header_bytes_len,
            "payload_bytes": len(ciphertext_with_tag),
            "required_bytes": required_bytes,
            "utilization": utilization,
            "n_lsb": n_lsb,
            "hw": None,
            "wav_fmt": fmt,
        }

        result = {
            "cover_audio_preview": cover_audio_preview,
            "stego_audio_preview": stego_audio_preview,
            "audio_meta": audio_meta,
            "audio_diff_note": audio_diff_note,
            "capacity": capacity,
            "download_wav_url": url_for("download_stego_wav"),
        }
        return render_template("index.html", result=result)
    except Exception as e:
        flash(f"Embed WAV error: {e}")
        return redirect(url_for("index"))

@app.get("/download/stego_wav")
def download_stego_wav():
    if not _LAST_STEGO_WAV:
        return redirect(url_for("index"))
    return send_file(io.BytesIO(_LAST_STEGO_WAV), as_attachment=True, download_name="stego.wav", mimetype="audio/wav")

@app.post("/extract_audio")
def extract_audio():
    global _LAST_PAYLOAD
    try:
        stego = request.files.get("stego_wav")
        n_lsb = int(request.form.get("lsb", "1"))
        key = int(request.form.get("key", "0"))
        if not stego:
            flash("Please provide a stego WAV.")
            return redirect(url_for("index"))

        stego.stream.seek(0)
        flat_bytes, wav_params, _ = load_wav_from_file(stego)

        # Reader function for header parse
        def read_bits(nbits):
            return extract_bits_from_bytes(flat_bytes, n_lsb, key, nbits)

        # 1) Parse secure header
        hdr, header_bits = parse_secure_header(read_bits)

        # 2) Ciphertext length = plaintext length + tag
        cipher_len = hdr.plain_len + TAG_LEN
        total_bits = header_bits + cipher_len * 8

        # 3) Extract full chunk, slice ciphertext
        all_bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, total_bits)
        all_bytes = bits_to_bytes(all_bits)
        header_bytes_len = header_bits // 8
        ciphertext_with_tag = all_bytes[header_bytes_len : header_bytes_len + cipher_len]

        # 4) Decrypt
        key_bytes = derive_key(int(key), hdr.salt)
        plaintext = decrypt_aes_gcm(key_bytes, hdr.nonce, ciphertext_with_tag, aad=None)

        # 5) Verify SHA-256
        if sha256(plaintext) != hdr.sha256:
            raise ValueError("Integrity check failed (SHA-256 mismatch). Wrong key or corrupted data.")

        # 6) Save as original name
        fname = secure_filename(hdr.name or "recovered.bin")
        if not os.path.splitext(fname)[1]:
            fname += ".bin"
        _LAST_PAYLOAD = (plaintext, fname)

        stego_audio_preview = audio_to_data_url(save_wav_to_bytes(flat_bytes, wav_params))
        result = {"payload_name": fname, "stego_audio_preview": stego_audio_preview}
        return render_template("index.html", result=result)

    except Exception as e:
        flash(f"Extract WAV error: {e}")
        return redirect(url_for("index"))

@app.get("/download/payload")
def download_payload():
    data, fname = _LAST_PAYLOAD
    if not data:
        return redirect(url_for("index"))
    ext = os.path.splitext(fname)[1]
    _name, mime = safe_download_name_and_mime(ext if ext else ".bin")
    return send_file(io.BytesIO(data), as_attachment=True, download_name=fname, mimetype=mime)

if __name__ == "__main__":
    app.run(debug=True)
