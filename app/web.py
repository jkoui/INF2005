# web_app.py
import base64
import io
import os
import struct
from typing import Tuple

import numpy as np
from flask import Flask, render_template_string, request, send_file, redirect, url_for, flash
from PIL import Image

app = Flask(__name__)
app.secret_key = "acw1-secret"  # for flash messages

# ---------------- Core stego helpers (image PNG/BMP) ----------------
MAGIC = b"ACW1"

def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        bits = np.pad(bits, (0, 8 - (bits.size % 8)))
    return np.packbits(bits.astype(np.uint8)).tobytes()

def build_header(payload: bytes, ext: str) -> bytes:
    ext_bytes = ext.encode("utf-8")
    if len(ext_bytes) > 255:
        raise ValueError("File extension too long.")
    return MAGIC + struct.pack(">I", len(payload)) + struct.pack("B", len(ext_bytes)) + ext_bytes

def parse_header(bit_reader) -> Tuple[int, str]:
    fixed_bits = bit_reader(72)  # 4 + 4 + 1 bytes
    fixed = bits_to_bytes(fixed_bits)
    if fixed[:4] != MAGIC:
        raise ValueError("Invalid magic / wrong key.")
    payload_len = struct.unpack(">I", fixed[4:8])[0]
    ext_len = fixed[8]
    ext = ""
    if ext_len:
        ext_bits = bit_reader(ext_len * 8)
        ext = bits_to_bytes(ext_bits).decode("utf-8", errors="ignore")
    return payload_len, ext

def prng_perm(total_slots: int, key: int) -> np.ndarray:
    rng = np.random.default_rng(seed=int(key) & 0xFFFFFFFF)
    return rng.permutation(total_slots)

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

def embed_bits_into_image(flat: np.ndarray, data_bits: np.ndarray, n_lsb: int, key: int) -> np.ndarray:
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

def extract_bits_from_image(flat: np.ndarray, n_lsb: int, key: int, num_bits: int) -> np.ndarray:
    total_slots = flat.size * n_lsb
    if num_bits > total_slots:
        raise ValueError("Requested bits exceed capacity.")
    perm = prng_perm(total_slots, key)
    tgt = perm[:num_bits]
    byte_idx = tgt // n_lsb
    bit_pos  = (tgt % n_lsb).astype(np.uint8)
    vals = flat[byte_idx]
    return ((vals >> bit_pos) & 1).astype(np.uint8)

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

# ---------------- Views ----------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ACW1 Stego (Web)</title>
  <style>
    body { font-family: system-ui, Arial, sans-serif; margin: 24px; }
    fieldset { margin-bottom: 24px; }
    label { display:block; margin-top: 8px; }
    .row { display:flex; gap:24px; }
    .card { border:1px solid #ddd; border-radius:12px; padding:16px; }
    .preview img { max-width: 320px; border:1px solid #ccc; border-radius:8px; }
    .btn { padding:8px 14px; border-radius:8px; border:1px solid #333; background:#111; color:#fff; cursor:pointer;}
    .btn.secondary { background:#fff; color:#111; }
    .note { color:#666; font-size: 0.9em; }
    .flash { background:#ffe9e9; color:#900; padding:10px 12px; border-radius:8px; margin-bottom:16px; border:1px solid #f5b5b5;}
  </style>
</head>
<body>
  <h1>ACW1 Stego (Web)</h1>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      {% for m in messages %}<div class="flash">{{ m }}</div>{% endfor %}
    {% endif %}
  {% endwith %}

  <div class="row">
    <div class="card" style="flex:1">
      <h2>Embed (hide payload in image)</h2>
      <form action="{{ url_for('embed') }}" method="post" enctype="multipart/form-data">
        <label>Cover image (PNG/BMP): <input required type="file" name="cover"></label>
        <label>Payload (any file): <input required type="file" name="payload"></label>
        <label>LSBs (1–8): <input type="number" name="lsb" min="1" max="8" value="1" required></label>
        <label>Key (integer): <input type="number" name="key" value="12345" required></label>
        <button class="btn" type="submit">Embed →</button>
      </form>
      <p class="note">This version supports images; WAV can be added later similarly.</p>
    </div>

    <div class="card" style="flex:1">
      <h2>Extract (recover payload)</h2>
      <form action="{{ url_for('extract') }}" method="post" enctype="multipart/form-data">
        <label>Stego image (PNG/BMP): <input required type="file" name="stego"></label>
        <label>LSBs (1–8): <input type="number" name="lsb" min="1" max="8" value="1" required></label>
        <label>Key (integer): <input type="number" name="key" value="12345" required></label>
        <button class="btn secondary" type="submit">Extract ←</button>
      </form>
    </div>
  </div>

  {% if result %}
    <hr>
    <h2>Result</h2>
    <div class="row">
      {% if result.stego_preview %}
        <div class="card preview">
          <h3>Stego image preview</h3>
          <img src="{{ result.stego_preview }}">
        </div>
      {% endif %}
      {% if result.lsb_preview %}
        <div class="card preview">
          <h3>LSB-plane preview</h3>
          <img src="{{ result.lsb_preview }}">
        </div>
      {% endif %}
      {% if result.diff_preview %}
        <div class="card preview">
          <h3>Difference preview</h3>
          <img src="{{ result.diff_preview }}">
        </div>
      {% endif %}
    </div>
    {% if result.download_url %}
      <p><a class="btn" href="{{ result.download_url }}">Download stego image</a></p>
    {% endif %}
    {% if result.payload_name %}
      <p><a class="btn" href="{{ url_for('download_payload') }}">Download extracted payload ({{ result.payload_name }})</a></p>
    {% endif %}
  {% endif %}
</body>
</html>
"""

# memory store for last files to download (simple demo only)
_LAST_STEGO: bytes = b""
_LAST_PAYLOAD: Tuple[bytes, str] = (b"", "recovered.bin")

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, result=None)

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

        # Read files
        payload_bytes = payload.read()
        ext = os.path.splitext(payload.filename or "")[1]
        header = build_header(payload_bytes, ext)
        data_bits = bytes_to_bits(header + payload_bytes)

        # Cover image
        cover.stream.seek(0)
        flat, shape = load_image_from_file(cover)
        cap = image_capacity_bits(flat.size, n_lsb)
        if data_bits.size > cap:
            flash(f"Cover too small. Capacity={cap//8} bytes, Needed={data_bits.size//8} bytes")
            return redirect(url_for("index"))

        stego_flat = embed_bits_into_image(flat, data_bits, n_lsb, key)
        # Save stego PNG to bytes
        stego_png = save_image_to_bytes(stego_flat, shape)
        _LAST_STEGO = stego_png

        # Previews
        stego_img = Image.open(io.BytesIO(stego_png)).convert("RGB")
        stego_preview = img_to_data_url(stego_img)
        lsb_img = image_lsb_plane(stego_flat, n_lsb, shape)
        lsb_preview = img_to_data_url(lsb_img)
        diff_img = image_diff(flat, stego_flat, shape)
        diff_preview = img_to_data_url(diff_img)

        result = {
            "stego_preview": stego_preview,
            "lsb_preview": lsb_preview,
            "diff_preview": diff_preview,
            "download_url": url_for("download_stego")
        }
        return render_template_string(INDEX_HTML, result=result)
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

        # Header peek helpers
        def reader(num_bits):
            return extract_bits_from_image(flat, n_lsb, key, num_bits)
        payload_len, ext = parse_header(reader)
        header_bits_len = 72 + len(ext.encode("utf-8")) * 8
        total_bits = header_bits_len + payload_len * 8
        bits = extract_bits_from_image(flat, n_lsb, key, total_bits)
        blob = bits_to_bytes(bits)
        ext_len = blob[8]
        header_total = 9 + ext_len
        data = blob[header_total:header_total + payload_len]
        fname = f"recovered{ext or '.bin'}"
        _LAST_PAYLOAD = (data, fname)

        result = {"payload_name": fname}
        return render_template_string(INDEX_HTML, result=result)
    except Exception as e:
        flash(f"Extract error: {e}")
        return redirect(url_for("index"))

@app.get("/download/payload")
def download_payload():
    data, fname = _LAST_PAYLOAD
    if not data:
        return redirect(url_for("index"))
    return send_file(io.BytesIO(data), as_attachment=True, download_name=fname)

if __name__ == "__main__":
    # For local dev only
    app.run(debug=True)
