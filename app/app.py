import io
import os
from typing import Tuple

from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from PIL import Image
from werkzeug.utils import secure_filename

from stego.core import (
    MAGIC, bytes_to_bits, bits_to_bytes, build_header, parse_header, safe_download_name_and_mime
)
from stego.lsb import embed_bits_into_bytes, extract_bits_from_bytes
from stego.media import (
    load_image_from_file, save_image_to_bytes, image_capacity_bits,
    image_lsb_plane, image_diff, img_to_data_url,
    load_wav_from_file, save_wav_to_bytes, wav_capacity_bits,
    audio_to_data_url, mean_abs_byte_delta
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
        header = build_header(payload_bytes, orig_name)
        data_bits = bytes_to_bits(header + payload_bytes)
        header_bytes_len = len(header)
        required_bytes = header_bytes_len + len(payload_bytes)

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
            "payload_bytes": len(payload_bytes),
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

        fixed_bits = extract_bits_from_bytes(flat, n_lsb, key, 72)
        fixed = bits_to_bytes(fixed_bits)
        if fixed[:4] != MAGIC:
            raise ValueError("Invalid magic / wrong key.")
        payload_len = int.from_bytes(fixed[4:8], "big")
        name_len = fixed[8]

        header_bits_len = 72 + name_len * 8
        header_bits = extract_bits_from_bytes(flat, n_lsb, key, header_bits_len)
        header_blob = bits_to_bytes(header_bits)
        filename = header_blob[9:9 + name_len].decode("utf-8", errors="ignore") if name_len else "recovered.bin"

        total_bits = header_bits_len + payload_len * 8
        bits = extract_bits_from_bytes(flat, n_lsb, key, total_bits)
        blob = bits_to_bytes(bits)
        data = blob[9 + name_len : 9 + name_len + payload_len]

        fname = secure_filename(filename)
        if not os.path.splitext(fname)[1]:
            fname += ".bin"
        _LAST_PAYLOAD = (data, fname)

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
        header = build_header(payload_bytes, orig_name)
        data_bits = bytes_to_bits(header + payload_bytes)
        header_bytes_len = len(header)
        required_bytes = header_bytes_len + len(payload_bytes)

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
            "payload_bytes": len(payload_bytes),
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

        fixed_bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, 72)
        fixed = bits_to_bytes(fixed_bits)
        if fixed[:4] != MAGIC:
            raise ValueError("Invalid magic / wrong key.")
        payload_len = int.from_bytes(fixed[4:8], "big")
        name_len = fixed[8]

        header_bits_len = 72 + name_len * 8
        header_bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, header_bits_len)
        header_blob = bits_to_bytes(header_bits)
        filename = header_blob[9:9 + name_len].decode("utf-8", errors="ignore") if name_len else "recovered.bin"

        total_bits = header_bits_len + payload_len * 8
        bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, total_bits)
        blob = bits_to_bytes(bits)
        data = blob[9 + name_len : 9 + name_len + payload_len]

        fname = secure_filename(filename)
        if not os.path.splitext(fname)[1]:
            fname += ".bin"
        _LAST_PAYLOAD = (data, fname)

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
