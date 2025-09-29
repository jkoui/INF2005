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
import math


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

from stego.lsb import (
    embed_bits_into_bytes,
    extract_bits_from_bytes,
    LOC_LEN, build_locator, parse_locator,
    write_bits_sequential, read_bits_sequential, embed_bits_into_bytes_seq, extract_bits_from_bytes_seq
)

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

from stego.steganalysis import ( chi_square_heatmap, decode_mono, waveform_compare_stacked  )

app = Flask(__name__)
app.secret_key = "acw1-secret"

HEADER_KEY_MASK = 0xA5A5A5A5
def header_key(user_key: int) -> int:
    return int(user_key) ^ HEADER_KEY_MASK

def guess_extension_from_bytes(data: bytes) -> str:
    """Return a likely file extension starting with dot, or '' if unknown."""
    # Common magic numbers
    sigs = [
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"%PDF-", ".pdf"),
        (b"\xFF\xD8\xFF", ".jpg"),
        (b"GIF87a", ".gif"),
        (b"GIF89a", ".gif"),
        (b"PK\x03\x04", ".zip"),         # also docx/xlsx/pptx
        (b"OggS", ".ogg"),
        (b"\x1A\x45\xDF\xA3", ".mkv"),   # Matroska
        (b"ID3", ".mp3"),
        (b"RIFF", ".wav"),               # check 'WAVE' later if you want
        (b"\x00\x00\x00\x18ftyp", ".mp4"),
        (b"\x00\x00\x00\x20ftyp", ".mp4"),
    ]
    for magic, ext in sigs:
        if data.startswith(magic):
            return ext

    # Text heuristic: decodes as UTF-8 and no NUL bytes
    try:
        _ = data.decode("utf-8")
        if b"\x00" not in data:
            return ".txt"
    except Exception:
        pass

    return ""  # unknown → caller can fall back to .bin


# ---------- ROI helpers (shared) ----------
def to_float_opt(s):
    """Return float or None if empty/invalid."""
    try:
        return float(s)
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
        start_time = to_float_opt(form.get("start_time"))
        end_time = to_float_opt(form.get("end_time"))

        # If any coordinate is missing, treat as "no ROI"
        if start_time is None or end_time is None:
            return None, img_w_actual, img_h_actual, None, None
        
        # Ensure valid time range
        if start_time >= end_time:
            raise ValueError("Start time must be less than end time.")

        return {"start_time": start_time, "end_time": end_time}, img_w_actual, img_h_actual, end_time - start_time, None

    # For image
    x1f = to_float_opt(form.get("x1"))
    y1f = to_float_opt(form.get("y1"))
    x2f = to_float_opt(form.get("x2"))
    y2f = to_float_opt(form.get("y2"))
    nat_w = to_float_opt(form.get("nat_w"))
    nat_h = to_float_opt(form.get("nat_h"))

    # Use natural size if provided
    imgW = int(nat_w) if nat_w is not None else int(img_w_actual)
    imgH = int(nat_h) if nat_h is not None else int(img_h_actual)

    # Missing coords -> no ROI
    if not all(v is not None for v in (x1f, y1f, x2f, y2f)):
        return None, imgW, imgH, None, None

    # Clamp to bounds
    x1f = max(0.0, min(float(imgW), x1f)); x2f = max(0.0, min(float(imgW), x2f))
    y1f = max(0.0, min(float(imgH), y1f)); y2f = max(0.0, min(float(imgH), y2f))

    # Normalize order
    if x2f < x1f: x1f, x2f = x2f, x1f
    if y2f < y1f: y1f, y2f = y2f, y1f
    
    # Convert to integer pixel rectangle (inclusive intent)
    x1 = int(math.floor(x1f))
    y1 = int(math.floor(y1f))
    x2 = int(math.ceil(x2f))
    y2 = int(math.ceil(y2f))

    # Ensure within bounds after rounding
    x1 = max(0, min(imgW, x1)); x2 = max(0, min(imgW, x2))
    y1 = max(0, min(imgH, y1)); y2 = max(0, min(imgH, y2))

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

def _hex(b: bytes, n: int = None) -> str:
    return (b if n is None else b[:n]).hex()


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
        needed_carrier_bytes = (header_bits.size + n_lsb - 1) // n_lsb  # ceil(bits / n_lsb)
        header_carrier_pixels = (needed_carrier_bytes + 2) // 3         # 3 bytes per RGB pixel
        header_rows = (header_carrier_pixels + W - 1) // W
        reserved_y2 = min(H, header_rows)                             # strip is rows [0 .. reserved_y2)

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
        
        # ---- Write header into the dedicated top strip only (ALWAYS 1 LSB) ----
        # ---- Write header into the dedicated top strip only (ALWAYS 1 LSB) ----
        n_lsb_header = 1
        flat_mut = flat.copy()

        # Compute final reserved_y2 from the real header size at 1-LSB
        needed_carrier_bytes = (header_bits.size + n_lsb_header - 1) // n_lsb_header
        header_carrier_pixels = (needed_carrier_bytes + 2) // 3
        header_rows = (header_carrier_pixels + W - 1) // W
        reserved_y2 = min(H, header_rows)

        # Re-apply ROI so it starts below the final header strip
        if region:
            x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
            y1 = max(y1, reserved_y2)
            if y1 >= y2:
                flash(f"ROI is too small after reserving top {reserved_y2} row(s) for the header.")
                return redirect(url_for("index"))
        else:
            x1, y1, x2, y2 = 0, reserved_y2, W, H

        # Recompute ROI-dependent capacity *after* the final clamp
        sel_w = x2 - x1
        sel_h = y2 - y1
        roi_pixels = sel_w * sel_h
        cap_bits = roi_pixels * C * n_lsb
        cap_bytes = cap_bits // 8
        if cipher_bits.size > cap_bits:
            flash(f"Cover too small. ROI capacity={cap_bytes} bytes, Needed={cipher_bytes_len} bytes")
            return redirect(url_for("index"))

        # Now write the header strip
        strip_len_bytes = reserved_y2 * W * C
        strip = flat_mut[:strip_len_bytes].copy()
        strip_after = embed_bits_into_bytes(strip, header_bits, n_lsb_header, header_key(key))
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

def try_parse_header_from_strip(flat, W, H, key: int):
    MAX_HDR_BYTES = FIXED_LEN + 255
    max_hdr_bits  = MAX_HDR_BYTES * 8

    n_lsb_header = 1
    # rows sized for worst-case header at 1 LSB
    needed_carrier_bytes = (max_hdr_bits + n_lsb_header - 1) // n_lsb_header
    max_hdr_pixels = (needed_carrier_bytes + 2) // 3
    max_hdr_rows   = (max_hdr_pixels + W - 1) // W
    reserved_y2    = min(H, max_hdr_rows)

    strip_len_bytes = reserved_y2 * W * 3
    if strip_len_bytes <= 0:
        raise ValueError("Header strip is empty.")

    strip = flat[:strip_len_bytes]
    cap_bits_in_strip = strip_len_bytes * n_lsb_header
    req_bits = min(max_hdr_bits, cap_bits_in_strip)

    all_hdr_bits = extract_bits_from_bytes(strip, n_lsb_header, header_key(key), req_bits)

    bit_cursor = 0
    def read_bits(nbits: int):
        nonlocal bit_cursor
        if bit_cursor + nbits > all_hdr_bits.size:
            raise ValueError("Header truncated in carrier.")
        out = all_hdr_bits[bit_cursor: bit_cursor + nbits]
        bit_cursor += nbits
        return out

    hdr, header_bits = parse_secure_header(read_bits)
    return hdr, header_bits, n_lsb_header, reserved_y2

@app.post("/extract")
def extract():
    global _LAST_PAYLOAD
    try:
        stego = request.files.get("stego")
        user_n_lsb = int(request.form.get("lsb", "1"))  # may be wrong
        key = int(request.form.get("key", "0"))

        if not stego:
            flash("Please provide a stego image.")
            return redirect(url_for("index"))

        stego.stream.seek(0)
        flat, shape = load_image_from_file(stego)
        H, W = shape

       # --- Robust header recovery (always 1 LSB) ---
        try:
            hdr, header_bits, n_lsb_header, reserved_y2 = try_parse_header_from_strip(flat, W, H, key)
        except Exception as e:
            flash(f"Extract error: {type(e).__name__}: {e or 'failed to parse header (check key)'}")
            return redirect(url_for("index"))

        # ROI clamped; ensure we start below header strip
        x1 = max(0, min(W, hdr.roi_x1))
        y1 = max(reserved_y2, max(0, min(H, hdr.roi_y1)))
        x2 = max(0, min(W, hdr.roi_x2))
        y2 = max(0, min(H, hdr.roi_y2))
        if x2 <= x1 or y2 <= y1:
            flash("Extract error: Invalid/empty ROI after accounting for header strip.")
            return redirect(url_for("index"))

        # Use the payload LSBs stored in header for ciphertext extraction
        n_lsb_payload = int(getattr(hdr, "lsb_used", 1)) or 1
        if user_n_lsb != n_lsb_payload:
            flash(f"Extract error: you selected {user_n_lsb} LSBs, but the payload was embedded with {n_lsb_payload} LSBs.")
            return redirect(url_for("index"))

        cipher_len = hdr.plain_len + TAG_LEN
        total_cipher_bits = cipher_len * 8
        arr = flat.reshape(H, W, 3)
        roi_flat = arr[y1:y2, x1:x2, :].reshape(-1)
        cipher_bits = extract_bits_from_bytes(roi_flat, n_lsb_payload, key, total_cipher_bits)
        ciphertext_with_tag = bits_to_bytes(cipher_bits)   
        # Decrypt & verify
        key_bytes = derive_key(int(key), hdr.salt)
        try:
            plaintext = decrypt_aes_gcm(key_bytes, hdr.nonce, ciphertext_with_tag, aad=None)
        except Exception as e:
            # This is typically empty-message exception on MAC failure
            flash("Extract error: Decryption failed (wrong key or corrupted data).")
            return redirect(url_for("index"))

        if sha256(plaintext) != hdr.sha256:
            flash("Extract error: Integrity check failed (SHA-256 mismatch).")
            return redirect(url_for("index"))

        # Save output
        fname = secure_filename(hdr.name or "")  # allow empty here
        ext = os.path.splitext(fname)[1]
        if not ext:
            # Guess from content; if still unknown, fall back to .bin
            ext = guess_extension_from_bytes(plaintext) or ".bin"
            base = fname if fname else "recovered"
            fname = f"{base}{ext}"

        _LAST_PAYLOAD = (plaintext, fname)

        stego_img = Image.fromarray(flat.reshape(H, W, 3).astype("uint8"))
        stego_preview = img_to_data_url(stego_img)
        result = {
            "payload_name": fname,
            "stego_preview": stego_preview,
            "note": f"Header LSB: 1; Payload LSB (from header): {n_lsb_payload} (you selected {user_n_lsb})"
        }
        return render_template("index.html", result=result)

    except Exception as e:
        import traceback
        print("EXTRACT ERROR TRACEBACK:\n", traceback.format_exc())
        flash(f"Extract error: {type(e).__name__}: {e or 'unknown'}")
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

        # --- crypto
        salt = _os.urandom(SALT_LEN)
        key_bytes = derive_key(int(key), salt)
        nonce, ciphertext_with_tag = encrypt_aes_gcm(key_bytes, payload_bytes, aad=None)
        plain_sha = sha256(payload_bytes)

        # --- load WAV as raw bytes (carrier space)
        cover.stream.seek(0)
        flat_bytes, wav_params, _ = load_wav_from_file(cover)
        fr = wav_params["framerate"]
        sw = wav_params["sampwidth"]         # bytes per sample
        nc = wav_params["nchannels"]
        frame_size = sw * nc                  # bytes per frame
        byte_rate = fr * frame_size
        total_bytes = flat_bytes.size

        # --- Parse ROI (seconds) -> bytes (clamped)
        region, _, _, sel_w, _ = parse_roi(request.form, 0, 0, is_audio=True)

        start_byte_raw, end_byte_raw = 0, total_bytes
        if region:
            st = float(region["start_time"])
            et = float(region["end_time"])
            start_byte_raw = int(max(0, min(total_bytes, st * byte_rate)))
            end_byte_raw   = int(max(0, min(total_bytes, et * byte_rate)))
            if end_byte_raw <= start_byte_raw:
                flash("Invalid audio ROI after clamping. Please choose a larger time range.")
                return redirect(url_for("index"))

            # align to frame boundaries
            start_byte_raw = (start_byte_raw // frame_size) * frame_size
            end_byte_raw   = (end_byte_raw   // frame_size) * frame_size
            end_byte_raw   = max(end_byte_raw, start_byte_raw + frame_size)

        # How many carrier BYTES are needed to store the LOC_LEN-bytes locator at n_lsb bits/byte?
        LOC_GUARD_BYTES = (LOC_LEN * 8 + n_lsb - 1) // n_lsb  # ceil(LOC_LEN*8 / n_lsb)

        # --- Decide actual embed span (avoid clobbering locator at file start)
        # Locator lives at the very start of the file. If no ROI, we shift embed start past locator.
        # --- Decide actual embed span and avoid clobbering the locator prefix ---
        if region:
            raw_start = start_byte_raw
            raw_end   = end_byte_raw
        else:
            raw_start = 0
            raw_end   = total_bytes

        # Always reserve the locator prefix at the beginning of the stream
        reserve = LOC_GUARD_BYTES  # depends on n_lsb
        embed_start = max(raw_start, reserve)
        embed_end   = raw_end

        if embed_end <= embed_start:
            flash("Selected region is too early/small and collides with the locator prefix. Please choose a later/larger ROI.")
            return redirect(url_for("index"))

        if not (0 <= embed_start < embed_end <= total_bytes):
            flash("Invalid embedding span.")
            return redirect(url_for("index"))

        embed_len_bytes = embed_end - embed_start
        flash(f"[DBG-EMBED] embed_start={embed_start}, embed_end={embed_end}, len={embed_len_bytes}, n_lsb={n_lsb}")


        # --- Build secure header (records the true embedding offsets)
        header = build_secure_header(
            plain_len=len(payload_bytes),
            filename=orig_name,
            lsb_used=n_lsb,
            start_offset=embed_start,
            nonce=nonce,
            salt=salt,
            sha256_bytes=plain_sha,
            roi_x1=embed_start, roi_y1=0,
            roi_x2=embed_end,   roi_y2=embed_len_bytes,
        )

        flash(f"[DBG-EMBED] header_magic={_hex(MAGIC)}, header_len={len(header)}")

        blob = header + ciphertext_with_tag
        data_bits = bytes_to_bits(blob)
        header_bytes_len = len(header)
        required_bytes = header_bytes_len + len(ciphertext_with_tag)

        # --- Capacity check over the actual slice
        cap_bits = wav_capacity_bits(embed_len_bytes, n_lsb)
        cap_bytes = cap_bits // 8
        if data_bits.size > cap_bits:
            where = "selected region" if region else "WAV"
            flash(f"{where} too small. Capacity={cap_bytes} bytes, Needed={required_bytes} bytes")
            return redirect(url_for("index"))

        # --- Embed inside [embed_start:embed_end] with PRNG over that slice
        stego_flat = flat_bytes.copy()
        slice_cover = flat_bytes[embed_start:embed_end]
        # slice_stego = embed_bits_into_bytes_seq(slice_cover, data_bits, n_lsb)
        slice_stego = embed_bits_into_bytes(slice_cover, data_bits, n_lsb, key)
        stego_flat[embed_start:embed_end] = slice_stego

        # --- Immediately read back from the same slice using PRNG (must start with '41435731' = 'ACW1')
        test_slice = stego_flat[embed_start:embed_end]
        perm_len   = test_slice.size * n_lsb
        from stego.core import prng_perm  # ensure this is your new deterministic version
        perm_head  = prng_perm(perm_len, key)[:16]  # 16 *bit slots* (not bytes)

        peek_bits_embed = extract_bits_from_bytes(test_slice, n_lsb, key, 16*8)
        peek_bytes_embed = bits_to_bytes(peek_bits_embed)
        flash(f"[DBG-EMBED] prng_first_slots={perm_head[:8]} ...")
        flash(f"[DBG-EMBED] prng_peek_first16={peek_bytes_embed[:16].hex()}")


        # --- Write tiny locator prefix sequentially at file start
        loc_bytes = build_locator(embed_start, embed_end)
        loc_bits  = bytes_to_bits(loc_bytes)
        write_bits_sequential(stego_flat, n_lsb, loc_bits, start_byte=0)
        flash(f"[DBG-EMBED] locator bytes={LOC_LEN}, guard_bytes={ (LOC_LEN*8 + n_lsb - 1)//n_lsb }")

        # --- Save WAV
        stego_wav = save_wav_to_bytes(stego_flat, wav_params)
        _LAST_STEGO_WAV = stego_wav

        # ---- Previews & stats (highlight the real embed span)
        mono_cover = decode_mono(flat_bytes, wav_params)
        mono_stego = decode_mono(stego_flat,  wav_params)

        roi_start_ms = 1000.0 * embed_start / byte_rate
        roi_end_ms   = 1000.0 * embed_end   / byte_rate

        full_img = waveform_compare_stacked(
            mono_cover, mono_stego,
            framerate=fr,
            start_sample=0, num_samples=None,
            width=1100, lane_h=180, lane_gap=28,
            show_diff=True, diff_gain=2.0,
            highlight_ms=(roi_start_ms, roi_end_ms)
        )
        waveform_full_preview = img_to_data_url(full_img)

        # zoom region: actual embed span (or 150ms mid if somehow tiny)
        start_sample = embed_start // (sw * nc)
        num_samples  = max(1, (embed_end - embed_start) // (sw * nc))
        if num_samples <= 1:
            center = mono_cover.size // 2
            num_samples = max(1, int(fr * 0.150))
            start_sample = max(0, center - num_samples // 2)

        zoom_img = waveform_compare_stacked(
            mono_cover, mono_stego,
            framerate=fr,
            start_sample=start_sample, num_samples=num_samples,
            width=1100, lane_h=180, lane_gap=28,
            show_diff=True, diff_gain="auto",
            highlight_ms=None
        )
        waveform_zoom_preview = img_to_data_url(zoom_img)

        cover_audio_preview = audio_to_data_url(save_wav_to_bytes(flat_bytes, wav_params))
        stego_audio_preview = audio_to_data_url(stego_wav)
        fmt = f"{wav_params['nchannels']} ch, {8*wav_params['sampwidth']}-bit, {wav_params['framerate']} Hz, {wav_params['nframes']} frames"
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
            "roi_time": f"{roi_start_ms/1000:.3f}s – {roi_end_ms/1000:.3f}s",
            "roi_bytes": embed_len_bytes,
        }

        result = {
            "cover_audio_preview": cover_audio_preview,
            "stego_audio_preview": stego_audio_preview,
            "audio_meta": fmt,
            "audio_diff_note": audio_diff_note,
            "capacity": capacity,
            "download_wav_url": url_for("download_stego_wav"),
            "waveform_full_preview": waveform_full_preview,
            "waveform_zoom_preview": waveform_zoom_preview,
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
        key   = int(request.form.get("key", "0"))
        if not stego:
            flash("Please provide a stego WAV.")
            return redirect(url_for("index"))

        stego.stream.seek(0)
        flat_bytes, wav_params, _ = load_wav_from_file(stego)
        total_bytes = flat_bytes.size

        # --- 0) Read locator prefix (sequential; no PRNG)
        loc_bits_needed = LOC_LEN * 8
        loc_bits = read_bits_sequential(flat_bytes, n_lsb, loc_bits_needed, start_byte=0)
        loc_bytes = bits_to_bytes(loc_bits)
        embed_start, embed_end = parse_locator(loc_bytes)
        flash(f"[DBG-EXTRACT] locator embed_start={embed_start}, embed_end={embed_end}, total_bytes={total_bytes}, n_lsb={n_lsb}")


        if not (0 <= embed_start < embed_end <= total_bytes):
            raise ValueError("Invalid locator offsets.")

        slice_bytes = flat_bytes[embed_start:embed_end]

        perm_len   = slice_bytes.size * n_lsb
        from stego.core import prng_perm
        perm_head  = prng_perm(perm_len, key)[:16]
        flash(f"[DBG-EXTRACT] prng_first_slots={perm_head[:8]} ...")


        # Reader that operates over the SAME slice/permutation as embed
        def read_bits(nbits):
            return extract_bits_from_bytes(slice_bytes, n_lsb, key, nbits)
        
        # Peek first 16 bytes (128 bits) via the PRNG over the SAME SLICE
        peek_bits = extract_bits_from_bytes(slice_bytes, n_lsb, key, 16*8)
        peek_bytes = bits_to_bytes(peek_bits)
        flash(f"[DBG-EXTRACT] first16={_hex(peek_bytes, 16)}")

        # 1) Parse secure header
        hdr, header_bits = parse_secure_header(read_bits)

        flash(f"[DBG-EXTRACT] parsed header_bits={header_bits}, hdr.start_offset={hdr.start_offset}, hdr.lsb_used={hdr.lsb_used}")
        flash(f"[DBG-EXTRACT] hdr.name_len={len(hdr.name or b'') if hasattr(hdr,'name') else 'n/a'}")


        # 2) Ciphertext length (plaintext + tag)
        cipher_len = hdr.plain_len + TAG_LEN
        total_bits = header_bits + cipher_len * 8

        # 3) Extract header+ciphertext from the slice, then split
        all_bits = extract_bits_from_bytes(slice_bytes, n_lsb, key, total_bits)
        all_bytes = bits_to_bytes(all_bits)
        header_bytes_len = header_bits // 8
        ciphertext_with_tag = all_bytes[header_bytes_len : header_bytes_len + cipher_len]

        # 4) Decrypt and verify
        key_bytes = derive_key(int(key), hdr.salt)
        plaintext = decrypt_aes_gcm(key_bytes, hdr.nonce, ciphertext_with_tag, aad=None)
        if sha256(plaintext) != hdr.sha256:
            raise ValueError("Integrity check failed (SHA-256 mismatch). Wrong key or corrupted data.")

        # 5) Save using original name
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

def transcode_to_playable_mp4(in_path: str, out_path: str):
    """
    Make a browser-friendly MP4 preview (H.264 + yuv420p).
    This WILL NOT preserve LSBs; preview-only.
    """
    cmd = [
        "ffmpeg", "-y", "-i", in_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",  # good for web playback
        "-an",                      # keep video-only for preview; remove if you want audio
        out_path
    ]
    subprocess.run(cmd, check=True)

_LAST_ORIG_VIDEO: bytes = b""
_LAST_STEGO_VIDEO_PREVIEW_MP4: bytes = b""

def write_bytes(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


@app.post("/embed_video")
def embed_video():
    global _LAST_STEGO_VIDEO, _LAST_STEGO_METHOD
    # ADD THESE:
    global _LAST_ORIG_VIDEO, _LAST_STEGO_VIDEO_PREVIEW_MP4

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

                # Save lossless stego with whatever container/codec the helper returns
                stego_lossless_path = os.path.join(tmpdir, "stego_lossless.mkv")
                stego_bytes_lossless = save_video_to_bytes(flat_mut, meta)  # <-- no kwargs
                write_bytes(stego_lossless_path, stego_bytes_lossless)

                # Make a browser-playable MP4 preview
                preview_mp4_path = os.path.join(tmpdir, "stego_preview.mp4")
                transcode_to_playable_mp4(stego_lossless_path, preview_mp4_path)

                with open(preview_mp4_path, "rb") as f:
                    _LAST_STEGO_VIDEO_PREVIEW_MP4 = f.read()

                # Canonical (lossless) for download/extraction
                stego_bytes = stego_bytes_lossless

                # Keep original for side-by-side preview
                with open(input_mp4, "rb") as f:
                    _LAST_ORIG_VIDEO = f.read()

            else:
                # ---- Audio-track stego (unchanged) ----
                wav_in = os.path.join(tmpdir, "audio_in.wav")
                wav_out = os.path.join(tmpdir, "audio_out.wav")
                mov_out = os.path.join(tmpdir, "stego_audio.mov")

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

                replace_audio_in_mp4(input_mp4, wav_out, mov_out)
                with open(mov_out, "rb") as f:
                    stego_bytes = f.read()

        _LAST_STEGO_VIDEO = stego_bytes
        _LAST_STEGO_METHOD = method

        result = {
            "download_video_url": url_for("download_stego_video"),
            "orig_stream_url": url_for("stream_original_video"),
            "stego_stream_url": url_for("stream_stego_preview")
        }
        return render_template("index.html", result=result)

    except Exception as e:
        flash(f"Embed Video error: {e}")
        return redirect(url_for("index"))

@app.get("/stream/original_video")
def stream_original_video():
    if not _LAST_ORIG_VIDEO:
        return redirect(url_for("index"))
    # assume it's mp4
    return send_file(io.BytesIO(_LAST_ORIG_VIDEO), mimetype="video/mp4")

@app.get("/stream/stego_preview")
def stream_stego_preview():
    if not _LAST_STEGO_VIDEO_PREVIEW_MP4:
        return redirect(url_for("index"))
    return send_file(io.BytesIO(_LAST_STEGO_VIDEO_PREVIEW_MP4), mimetype="video/mp4")


@app.get("/download/stego_video")
def download_stego_video():
    if not _LAST_STEGO_VIDEO:
        return redirect(url_for("index"))

    if _LAST_STEGO_METHOD == "iframe":
        # true, lossless stego for extraction
        filename = "stego_iframe_lossless.mkv"
        mimetype = "video/x-matroska"
    else:
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
            # Ensure that both .mp4 and .mkv are supported for extraction
            ext = os.path.splitext(stego.filename or "")[1].lower()
            if ext not in [".mp4", ".mov", ".mkv"]:
                ext = ".mp4"  # Default to MP4 if extension is unsupported
            
            input_video = os.path.join(tmpdir, f"stego{ext}")
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

