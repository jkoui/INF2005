import io
import os
from typing import Tuple
import traceback
import numpy as np
import os as _os
import subprocess
import tempfile
import binascii
import shutil


from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from PIL import Image
from werkzeug.utils import secure_filename


from stego.crypto import (
    derive_key, encrypt_aes_gcm, decrypt_aes_gcm, sha256, SALT_LEN, TAG_LEN
)

from stego.core import (
    MAGIC, bytes_to_bits, bits_to_bytes, safe_download_name_and_mime,
    build_secure_header, parse_secure_header, FIXED_LEN
)

from stego.lsb import embed_bits_into_bytes, extract_bits_from_bytes

from stego.media import (
    load_image_from_file, save_image_to_bytes, image_capacity_bits,
    image_lsb_plane, image_diff, img_to_data_url,
    load_wav_from_file, save_wav_to_bytes, wav_capacity_bits,
    audio_to_data_url, mean_abs_byte_delta,
    image_bit_plane, image_lsb_change_mask,
    load_video_from_file, save_video_to_bytes, video_capacity_bits,
    image_bit_plane, image_lsb_change_mask,
    make_change_overlay  
)

from stego.steganalysis import ( chi_square_heatmap )

app = Flask(__name__)
app.secret_key = "acw1-secret"

HEADER_KEY_MASK = 0xA5A5A5A5
def header_key(user_key: int) -> int:
    return int(user_key) ^ HEADER_KEY_MASK

# ---------- ROI helpers (shared) ----------
def _to_int_opt(s):
    """Return int(s) or None if empty/invalid."""
    try:
        return int(s)
    except Exception:
        return None

def parse_roi(form, img_w_actual: int, img_h_actual: int, is_audio: bool = False):
    """
    Read x1,y1,x2,y2 and nat_w,nat_h from the form.
    Returns: (region_dict_or_None, imgW, imgH, sel_w, sel_h)
    Raises ValueError if a zero-area ROI is provided.
    """

    # For audio
    if is_audio:
        start_time = _to_int_opt(form.get("start_time"))
        end_time = _to_int_opt(form.get("end_time"))

        # If any coordinate is missing, treat as "no ROI"
        if start_time is None or end_time is None:
            return None, img_w_actual, img_h_actual, None, None
        
        # Ensure valid time range
        if start_time >= end_time:
            raise ValueError("Start time must be less than end time.")

        return {"start_time": start_time, "end_time": end_time}, img_w_actual, img_h_actual, end_time - start_time, None

    # For image
    x1 = _to_int_opt(form.get("x1"))
    y1 = _to_int_opt(form.get("y1"))
    x2 = _to_int_opt(form.get("x2"))
    y2 = _to_int_opt(form.get("y2"))
    nat_w = _to_int_opt(form.get("nat_w"))
    nat_h = _to_int_opt(form.get("nat_h"))

    # Use the image's natural size reported by the client if present
    imgW = nat_w or img_w_actual
    imgH = nat_h or img_h_actual

    # If any coordinate is missing, treat as "no ROI"
    if not all(v is not None for v in (x1, y1, x2, y2)):
        return None, imgW, imgH, None, None

    # Clamp to bounds
    x1 = max(0, min(imgW, x1)); x2 = max(0, min(imgW, x2))
    y1 = max(0, min(imgH, y1)); y2 = max(0, min(imgH, y2))

    # Normalize order
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    sel_w = x2 - x1
    sel_h = y2 - y1
    if sel_w == 0 or sel_h == 0:
        # User typed/dragged a line/point; reject
        raise ValueError("Selected region has zero area.")

    region = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return region, imgW, imgH, sel_w, sel_h

# in-memory artifacts
_LAST_STEGO: bytes = b""
_LAST_STEGO_WAV: bytes = b""
_LAST_PAYLOAD: Tuple[bytes, str] = (b"", "recovered.bin")

@app.get("/")
def index():
    return render_template("index.html", result=None)

# ==========================================================
# MP4 Audio utilities
# ==========================================================
def extract_audio_from_mp4(mp4_path, wav_path):
    cmd = [
        "ffmpeg", "-y", "-i", mp4_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ac", "2", "-ar", "44100",
        wav_path
    ]
    subprocess.run(cmd, check=True)


def replace_audio_in_mp4(video_in, audio_wav, video_out):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_in,
        "-i", audio_wav,
        "-c:v", "copy",
        "-c:a", "pcm_s16le",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        video_out
    ]
    subprocess.run(cmd, check=True)
# ==========================================================

# -------- Image routes --------
@app.post("/embed")
def embed():
    global _LAST_STEGO
    try:
        cover = request.files.get("cover")
        payload = request.files.get("payload")
        additional_input = request.form.get("additional_input")
        print(f"Additional Input: {additional_input}")  # Debugging line
        n_lsb = int(request.form.get("lsb", "1"))
        key = int(request.form.get("key", "0"))

        # Ensure cover image is provided
        if not cover:
            flash("Please provide a cover image file.")
            return redirect(url_for("index"))

        # Validate payload or additional_input: only one should be filled
        if not payload and not additional_input:
            flash("Please provide either a payload file or input text.")
            return redirect(url_for("index"))

        if payload and additional_input:
            flash("Please provide only one payload: either a file or text input.")
            return redirect(url_for("index"))

        # Process the payload (either file or text input)
        if payload:
            # If a file is provided, process it as the payload
            payload_bytes = payload.read()
            orig_name = os.path.basename(payload.filename or "payload.bin")
        elif additional_input:
            # If text input is provided, treat it as the payload
            print(f"Processing input text as payload: {additional_input}")  # Debugging
            payload_bytes = additional_input.encode('utf-8')  # Convert text to bytes
            orig_name = "text_payload.txt"  # You can customize this name if needed

        else:
            flash("No valid payload provided.")
            return redirect(url_for("index"))
        


        # --- crypto ---
        salt = _os.urandom(SALT_LEN)
        key_bytes = derive_key(int(key), salt)
        nonce, ciphertext_with_tag = encrypt_aes_gcm(key_bytes, payload_bytes, aad=None)
        plain_sha = sha256(payload_bytes)

        cover.stream.seek(0)
        flat, shape = load_image_from_file(cover)
        H, W = shape
        C = 3  # RGB

        # ROI from form (optional, for EMBED only)
        try:
            region, imgW, imgH, sel_w, sel_h = parse_roi(request.form, W, H)
        except ValueError as ve:
            flash(f"{ve} Please draw a valid rectangle or leave it blank.")
            return redirect(url_for("index"))

        # Build secure header (stores ROI — defaults to full image if none)
        if region:
            x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
        else:
            x1, y1, x2, y2 = 0, 0, W, H

        # Secure header (store lsb_used; start_offset=0 for now)
        header = build_secure_header(
            plain_len=len(payload_bytes),
            filename=orig_name,
            lsb_used=n_lsb,
            start_offset=0,
            nonce=nonce,
            salt=salt,
            sha256_bytes=plain_sha,
            roi_x1=x1, roi_y1=y1, roi_x2=x2, roi_y2=y2
        )

        # Bits to embed
        header_bits = bytes_to_bits(header)
        cipher_bits = bytes_to_bits(ciphertext_with_tag)
        header_bytes_len = len(header)
        cipher_bytes_len  = len(ciphertext_with_tag)
        required_bytes = header_bytes_len + cipher_bytes_len

        # Reserve a top strip for the header so ROI can never overwrite it
        header_carrier_bytes = (header_bits.size + 7) // 8           # ceil
        header_carrier_pixels = (header_carrier_bytes + 2) // 3      # ceil
        header_rows = (header_carrier_pixels + W - 1) // W               # ceil
        reserved_y2 = min(H, header_rows)                                # strip is rows [0 .. reserved_y2)

        # Adjust/choose ROI so it does not include the reserved strip.
        if region:
            # Push ROI down if it overlaps the reserved strip.
            x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
            y1 = max(y1, reserved_y2)
            if y1 >= y2:
                flash(f"ROI is too small after reserving top {reserved_y2} row(s) for the header.")
                return redirect(url_for("index"))
        else:
            # No ROI provided → use the whole image *below* the header strip
            x1, y1, x2, y2 = 0, reserved_y2, W, H

        # Capacity of the (possibly adjusted) ROI
        sel_w = x2 - x1
        sel_h = y2 - y1
        roi_pixels = sel_w * sel_h
        cap_bits = roi_pixels * C * n_lsb
        cap_bytes = cap_bits // 8
        if cipher_bits.size > cap_bits:
            flash(f"Cover too small. ROI capacity={cap_bytes} bytes, Needed={cipher_bytes_len} bytes")
            return redirect(url_for("index"))
        
        # ---- Write header into the dedicated top strip only ----
        flat_mut = flat.copy()
        strip_len_bytes = reserved_y2 * W * C
        strip = flat_mut[:strip_len_bytes].copy()
        strip_after = embed_bits_into_bytes(strip, header_bits, n_lsb, header_key(key))
        flat_mut[:strip_len_bytes] = strip_after

        # ---- Write ciphertext into ROI only (never touches header strip) ----
        arr = flat_mut.reshape(H, W, C)
        roi = arr[y1:y2, x1:x2, :]
        roi_flat = roi.reshape(-1).copy()
        stego_roi_flat = embed_bits_into_bytes(roi_flat, cipher_bits, n_lsb, key)
        roi[:, :, :] = stego_roi_flat.reshape(roi.shape)
        stego_flat_all = arr.reshape(-1)

        stego_png = save_image_to_bytes(stego_flat_all, shape)
        _LAST_STEGO = stego_png

        overlay_img = make_change_overlay(
            cover_flat=flat,
            stego_flat=stego_flat_all,
            shape=(H, W),
            n_lsb=n_lsb,
            color=(255, 32, 32),   # red
            alpha=0.6,             # could be a UI slider in the form later
            dilate_px=1,           # could be a UI slider
            outline_only=False,    # could be a UI toggle
            roi=(x1, y1, x2, y2)   # draw cyan border around ROI
        )
        overlay_preview = img_to_data_url(overlay_img)

        # Chi-square heatmap (delta mode, over all channels)
        chi_img = chi_square_heatmap(
            cover_flat=flat,
            stego_flat=stego_flat_all,
            shape=(H, W),
            block=16,            # try 8, 16, or 32
            channel=None,        # None = R+G+B combined
            mode="delta",        # "delta" highlights suspected embedding
            alpha=0.60,          # overlay strength
            color=(255, 0, 0)    # red heat
        )
        chi_heatmap_preview = img_to_data_url(chi_img)

        # preview controls
        bit = int(request.form.get("bit", "0"))               # 0..7
        channel_map = {"all": None, "r": 0, "g": 1, "b": 2}
        channel = channel_map.get(request.form.get("channel", "all"), None)

        # previews
        cover_img = Image.fromarray(flat.reshape(H, W, C).astype("uint8"), mode="RGB")
        cover_preview = img_to_data_url(cover_img)
        cover_meta = f"{W}×{H} px, 3 channels (RGB)"

        stego_img = Image.open(io.BytesIO(stego_png)).convert("RGB")
        stego_preview = img_to_data_url(stego_img)

        lsb_img = image_lsb_plane(stego_flat_all, n_lsb, shape)
        lsb_preview = img_to_data_url(lsb_img)

        bit_plane_img = image_bit_plane(stego_flat_all, shape, bit=bit, channel=channel)
        bit_plane_preview = img_to_data_url(bit_plane_img)

        change_mask_img = image_lsb_change_mask(flat, stego_flat_all, shape, bit=bit)
        change_mask_preview = img_to_data_url(change_mask_img)

        diff_img = image_diff(flat, stego_flat_all, shape)
        diff_preview = img_to_data_url(diff_img)

        utilization = round((required_bytes / cap_bytes) * 100, 2) if cap_bytes else 0.0
        capacity = {
            "cover_bytes": cap_bytes,
            "header_bytes": header_bytes_len,
            "payload_bytes": len(ciphertext_with_tag),
            "required_bytes": required_bytes,
            "utilization": utilization,
            "n_lsb": n_lsb,
            "hw": f"{W}×{H}",
            "wav_fmt": None,
            "roi": f"x={x1}..{x2}, y={y1}..{y2} ({sel_w}×{sel_h}px) (top {reserved_y2} row(s) reserved)",
        }
        if region:
            capacity["roi"] = f"x={x1}..{x2}, y={y1}..{y2} ({sel_w}×{sel_h}px)"

        result = {
            "cover_preview": cover_preview,
            "cover_meta": cover_meta,
            "stego_preview": stego_preview, 
            "lsb_preview": lsb_preview,
            "bit_plane_preview": bit_plane_preview,      # NEW: precise single-bit plane
            "change_mask_preview": change_mask_preview,  # NEW: pixels changed at that bit
            "diff_preview": diff_preview,
            "overlay_preview": overlay_preview,
            "capacity": capacity,
            "download_url": url_for("download_stego"),
            "chi_heatmap_preview": chi_heatmap_preview,
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
        H, W = shape
        C = 3

        # --- Read header from the dedicated top strip ---
        # Use a strip big enough to hold the maximum possible header:
        # FIXED_LEN + up to 255 bytes of filename
        MAX_HDR_BYTES = FIXED_LEN + 255
        max_hdr_pixels = (MAX_HDR_BYTES + 2) // 3
        max_hdr_rows = (max_hdr_pixels + W - 1) // W
        reserved_y2 = min(H, max_hdr_rows)
        strip_len_bytes = reserved_y2 * W * C
        strip = flat[:strip_len_bytes]

        # Stateful reader over the strip only (matches how we embedded it)
        read_cursor = {"ofs": 0}
        def read_bits(nbits: int):
            # slice the unread portion of the strip and advance
            start = read_cursor["ofs"]
            # compute how many bytes we need to expose; we can just pass the
            # remaining strip because extract_bits_from_bytes reads from start
            remaining = strip[start:]
            out = extract_bits_from_bytes(remaining, n_lsb, header_key(key), nbits)
            # advance by however many *carrier* bytes were consumed
            # carrier bytes used = ceil(nbits / (8 * n_lsb)) per channel-byte
            used_bytes = (nbits + (n_lsb * 8) - 1) // (n_lsb * 8)
            read_cursor["ofs"] += used_bytes
            return out

        # 1) Parse secure header to learn sizes/materials
        hdr, header_bits = parse_secure_header(read_bits)

        # ROI from header (clamp just in case)
        x1 = max(0, min(W, hdr.roi_x1))
        y1 = max(0, min(H, hdr.roi_y1))
        x2 = max(0, min(W, hdr.roi_x2))
        y2 = max(0, min(H, hdr.roi_y2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid ROI stored in header.")

        # 2) Ciphertext length = plaintext length + 16 (AES-GCM tag)
        cipher_len = hdr.plain_len + TAG_LEN
        total_cipher_bits = cipher_len * 8

        arr = flat.reshape(H, W, C)
        roi_flat = arr[y1:y2, x1:x2, :].reshape(-1)

        cipher_bits = extract_bits_from_bytes(roi_flat, n_lsb, key, total_cipher_bits)
        ciphertext_with_tag = bits_to_bytes(cipher_bits)

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

        # Parse ROI for audio (start_time and end_time)
        region, _, _, sel_w, _ = parse_roi(request.form, 0, 0, is_audio=True)

        # Build secure header (audio case)
        header = build_secure_header(
            plain_len=len(payload_bytes),
            filename=orig_name,
            lsb_used=n_lsb,
            start_offset=0,
            nonce=nonce,
            salt=salt,
            sha256_bytes=plain_sha,
            roi_x1=0,  # Audio does not have x/y coordinates like image
            roi_y1=0,  # Audio uses time for ROI
            roi_x2=sel_w,  # Audio "width" is its time range
            roi_y2=sel_w,
        )

        blob = header + ciphertext_with_tag
        data_bits = bytes_to_bits(blob)
        header_bytes_len = len(header)
        required_bytes = header_bytes_len + len(ciphertext_with_tag)

        # WAV loading and capacity check
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

_LAST_STEGO_VIDEO: bytes = b""
_LAST_STEGO_METHOD: str = "iframe"

@app.post("/embed_video")
def embed_video():
    global _LAST_STEGO_VIDEO, _LAST_STEGO_METHOD
    try:
        cover = request.files.get("cover_video")
        payload = request.files.get("payload_video")
        method = request.form.get("method", "iframe")
        n_lsb = int(request.form.get("lsb", "1"))
        key = int(request.form.get("key", "0"))

        if not cover or not payload:
            flash("Please provide both cover and payload.")
            return redirect(url_for("index"))

        # --- crypto ---
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
            roi_x1=0, roi_y1=0, roi_x2=0, roi_y2=0
        )

        header_bits = bytes_to_bits(header)
        cipher_bits = bytes_to_bits(ciphertext_with_tag)
        all_bits = np.concatenate([header_bits, cipher_bits])

        with tempfile.TemporaryDirectory() as tmpdir:
            input_mp4 = os.path.join(tmpdir, "input.mp4")
            cover.save(input_mp4)

            if method == "iframe":
                # ---- Video I-frame stego ----
                flat, meta, frames = load_video_from_file(input_mp4)
                cap_bits = video_capacity_bits(flat.size, n_lsb)
                if all_bits.size > cap_bits:
                    flash("Payload too large for video (I-frame).")
                    return redirect(url_for("index"))

                flat_mut = embed_bits_into_bytes(flat.copy(), all_bits, n_lsb, key)
                stego_bytes = save_video_to_bytes(flat_mut, meta)

            else:
                # ---- Audio-track stego ----
                wav_in = os.path.join(tmpdir, "audio_in.wav")
                wav_out = os.path.join(tmpdir, "audio_out.wav")
                mov_out = os.path.join(tmpdir, "stego_audio.mov")  # stay consistent

                extract_audio_from_mp4(input_mp4, wav_in)

                flat_bytes, wav_params, _ = load_wav_from_file(wav_in)
                cap_bits = wav_capacity_bits(flat_bytes.size, n_lsb)
                if all_bits.size > cap_bits:
                    flash("Payload too large for audio track.")
                    return redirect(url_for("index"))

                flat_mut = embed_bits_into_bytes(flat_bytes.copy(), all_bits, n_lsb, key)
                stego_wav = save_wav_to_bytes(flat_mut, wav_params)
                with open(wav_out, "wb") as f:
                    f.write(stego_wav)

                # Mux back into MOV (PCM-safe)
                replace_audio_in_mp4(input_mp4, wav_out, mov_out)
                with open(mov_out, "rb") as f:
                    stego_bytes = f.read()

        _LAST_STEGO_VIDEO = stego_bytes
        _LAST_STEGO_METHOD = method
        result = {"download_video_url": url_for("download_stego_video")}
        return render_template("index.html", result=result)

    except Exception as e:
        flash(f"Embed Video error: {e}")
        return redirect(url_for("index"))

@app.get("/download/stego_video")
def download_stego_video():
    if not _LAST_STEGO_VIDEO:
        return redirect(url_for("index"))

    if _LAST_STEGO_METHOD == "iframe":
        filename = "stego_iframe.mp4"
        mimetype = "video/mp4"
    else:  # audio
        filename = "stego_audio.mov"
        mimetype = "video/quicktime"

    return send_file(
        io.BytesIO(_LAST_STEGO_VIDEO),
        as_attachment=True,
        download_name=filename,
        mimetype=mimetype
    )



@app.post("/extract_video")
def extract_video():
    global _LAST_PAYLOAD
    try:
        stego = request.files.get("stego_video")
        method = request.form.get("method", "iframe")
        n_lsb = int(request.form.get("lsb", "1"))
        key = int(request.form.get("key", "0"))

        if not stego:
            flash("Please provide a stego video.")
            return redirect(url_for("index"))

        with tempfile.TemporaryDirectory() as tmpdir:
            ext = os.path.splitext(stego.filename or "")[1].lower()
            if ext not in [".mp4", ".mov"]:
                ext = ".mp4"
            input_video = os.path.join(tmpdir, "stego" + ext)
            stego.save(input_video)

            if method == "iframe":
                # ---- Extract from I-frames ----
                flat, meta, frames = load_video_from_file(input_video)

                # 1) Estimate maximum header size
                MAX_HDR_BYTES = FIXED_LEN + 255
                max_hdr_bits = MAX_HDR_BYTES * 8

                # 2) Read a large chunk first (header + possible payload)
                all_bits = extract_bits_from_bytes(flat, n_lsb, key, max_hdr_bits)

                # 3) Parse header from that bitstream
                bit_cursor = {"ofs": 0}
                def read_bits(nbits: int):
                    start = bit_cursor["ofs"]
                    out = all_bits[start:start+nbits]
                    bit_cursor["ofs"] += nbits
                    return out

                hdr, header_bits = parse_secure_header(read_bits)

                # 4) Now we know ciphertext length
                cipher_len = hdr.plain_len + TAG_LEN
                total_bits = header_bits + cipher_len * 8

                # 5) Re-read header+ciphertext together (clean slice)
                all_bits = extract_bits_from_bytes(flat, n_lsb, key, total_bits)
                all_bytes = bits_to_bytes(all_bits)

                header_bytes_len = header_bits // 8
                ciphertext_with_tag = all_bytes[header_bytes_len: header_bytes_len + cipher_len]

            else:
                # ---- Extract from audio track ----
                wav_in = os.path.join(tmpdir, "audio_in.wav")
                extract_audio_from_mp4(input_video, wav_in)

                flat_bytes, wav_params, _ = load_wav_from_file(wav_in)

                MAX_HDR_BYTES = FIXED_LEN + 255
                max_hdr_bits = MAX_HDR_BYTES * 8
                all_bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, max_hdr_bits)

                bit_cursor = {"ofs": 0}
                def read_bits(nbits: int):
                    start = bit_cursor["ofs"]
                    out = all_bits[start:start+nbits]
                    bit_cursor["ofs"] += nbits
                    return out

                hdr, header_bits = parse_secure_header(read_bits)

                cipher_len = hdr.plain_len + TAG_LEN
                total_bits = header_bits + cipher_len * 8

                all_bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, total_bits)
                all_bytes = bits_to_bytes(all_bits)

                header_bytes_len = header_bits // 8
                ciphertext_with_tag = all_bytes[header_bytes_len: header_bytes_len + cipher_len]

        # ---- Common decrypt logic ----
        key_bytes = derive_key(int(key), hdr.salt)
        plaintext = decrypt_aes_gcm(key_bytes, hdr.nonce, ciphertext_with_tag, aad=None)

        if sha256(plaintext) != hdr.sha256:
            raise ValueError("Integrity check failed (SHA-256 mismatch). Wrong key or corrupted data.")

        fname = secure_filename(hdr.name or "recovered.bin")
        if not os.path.splitext(fname)[1]:
            fname += ".bin"
        _LAST_PAYLOAD = (plaintext, fname)

        result = {"payload_name": fname}
        return render_template("index.html", result=result)

    except Exception as e:
        import traceback
        print("EXTRACT_VIDEO ERROR:", traceback.format_exc())
        flash(f"Extract Video error: {str(e) or type(e).__name__}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

