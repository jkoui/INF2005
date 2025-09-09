# web_app.py
import base64
import io
import os
import struct
import wave
from typing import Tuple
from werkzeug.utils import secure_filename

import numpy as np
from flask import Flask, render_template_string, request, send_file, redirect, url_for, flash
from PIL import Image

app = Flask(__name__)
app.secret_key = "acw1-secret"  # for flash messages

# ---------------- Core stego helpers ----------------
MAGIC = b"ACW1"

def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr).astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        bits = np.pad(bits, (0, 8 - (bits.size % 8)))
    return np.packbits(bits.astype(np.uint8)).tobytes()

# ---------------- Core stego helpers ----------------
MAGIC = b"ACW1"

def build_header(payload: bytes, filename: str) -> bytes:
    """
    Header layout:
      MAGIC(4) + PAYLOAD_LEN(4, big-endian) + NAME_LEN(1) + NAME(NAME_LEN bytes, utf-8)
    Where NAME is the full original filename, e.g. 'report_v3.pdf'
    """
    name = os.path.basename(filename or "recovered.bin")
    name_bytes = name.encode("utf-8")
    if len(name_bytes) > 255:
        raise ValueError("Filename too long (max 255 bytes in utf-8).")
    return MAGIC + struct.pack(">I", len(payload)) + struct.pack("B", len(name_bytes)) + name_bytes

def parse_header(bit_reader) -> Tuple[int, str]:
    """
    Parse the same header and return (payload_len, filename).
    NOTE: bit_reader must return EXACTLY the requested number of bits starting from the start
          of the embedding permutation; see extract routes below for safe usage.
    """
    fixed_bits = bit_reader(72)  # 4 + 4 + 1 bytes
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


def prng_perm(total_slots: int, key: int) -> np.ndarray:
    rng = np.random.default_rng(seed=int(key) & 0xFFFFFFFF)
    return rng.permutation(total_slots)

# ---------------- Image helpers ----------------
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

def embed_bits_into_bytes(flat: np.ndarray, data_bits: np.ndarray, n_lsb: int, key: int) -> np.ndarray:
    """Generic byte-wise LSB embedder (used by both image/audio)."""
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

# ---------------- WAV helpers ----------------
def load_wav_from_file(file_storage):
    """
    Returns:
      flat_bytes (np.uint8 1-D),
      params (dict with nchannels, sampwidth, framerate, nframes, comptype, compname),
      raw_frames (bytes)
    """
    with wave.open(file_storage.stream, "rb") as wf:
        params = dict(
            nchannels=wf.getnchannels(),
            sampwidth=wf.getsampwidth(),
            framerate=wf.getframerate(),
            nframes=wf.getnframes(),
            comptype=wf.getcomptype(),
            compname=wf.getcompname(),
        )
        if params["comptype"] != "NONE":
            raise ValueError("Only uncompressed PCM WAV is supported.")
        if params["sampwidth"] not in (1, 2):
            raise ValueError("Only 8-bit or 16-bit PCM WAV is supported.")
        frames = wf.readframes(params["nframes"])
    flat = np.frombuffer(frames, dtype=np.uint8)  # byte-level view
    return flat.copy(), params, frames  # copy to detach from read-only buffer

def save_wav_to_bytes(flat: np.ndarray, params: dict) -> bytes:
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(params["nchannels"])
        wf.setsampwidth(params["sampwidth"])
        wf.setframerate(params["framerate"])
        nframes = params["nframes"]
        # frames = bytes length / (channels * sampwidth)
        # We trust original nframes; wave will compute from length anyway.
        wf.writeframes(flat.tobytes())
    bio.seek(0)
    return bio.read()

def _safe_download_name_from_ext(ext: str) -> Tuple[str, str]:
    """
    Build a safe download filename & a reasonable MIME type from an extension.
    ext is expected to include the leading dot (e.g. '.pdf').
    """
    # default
    mime = "application/octet-stream"
    name = f"recovered{ext or '.bin'}"

    # normalize/sanitize filename
    name = secure_filename(name)

    # simple MIME mapping (extend as you wish)
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

def wav_capacity_bits(flat_len: int, n_lsb: int) -> int:
    return flat_len * n_lsb  # byte-level capacity

def audio_to_data_url(wav_bytes: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")

def mean_abs_byte_delta(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(a[:n].astype(np.int16) - b[:n].astype(np.int16))))

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
    .row { display:flex; gap:24px; flex-wrap: wrap; }
    .card { border:1px solid #ddd; border-radius:12px; padding:16px; }
    .preview img, .preview audio { max-width: 320px; border:1px solid #ccc; border-radius:8px; }
    .btn { padding:8px 14px; border-radius:8px; border:1px solid #333; background:#111; color:#fff; cursor:pointer;}
    .btn.secondary { background:#fff; color:#111; }
    .note { color:#666; font-size: 0.9em; }
    .flash { background:#ffe9e9; color:#900; padding:10px 12px; border-radius:8px; margin-bottom:16px; border:1px solid #f5b5b5;}
    table.kv { border-collapse: collapse; }
    table.kv td { padding:6px 10px; border-bottom:1px dashed #ddd; }
    table.kv td.key { color:#555; }
    .muted { color:#666; }
    h2 { margin-top: 0; }
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
    <div class="card" style="flex:1; min-width: 340px;">
      <h2>Embed (Image)</h2>
      <form action="{{ url_for('embed') }}" method="post" enctype="multipart/form-data">
        <label>Cover image (PNG/BMP): <input required type="file" name="cover" accept=".png,.bmp,image/png,image/bmp"></label>
        <label>Payload (any file): <input required type="file" name="payload"></label>
        <label>LSBs (1–8): <input type="number" name="lsb" min="1" max="8" value="1" required></label>
        <label>Key (integer): <input type="number" name="key" value="12345" required></label>
        <button class="btn" type="submit">Embed →</button>
      </form>
      <p class="note">Supports images (PNG/BMP).</p>
    </div>

    <div class="card" style="flex:1; min-width: 340px;">
      <h2>Extract (Image)</h2>
      <form action="{{ url_for('extract') }}" method="post" enctype="multipart/form-data">
        <label>Stego image (PNG/BMP): <input required type="file" name="stego" accept=".png,.bmp,image/png,image/bmp"></label>
        <label>LSBs (1–8): <input type="number" name="lsb" min="1" max="8" value="1" required></label>
        <label>Key (integer): <input type="number" name="key" value="12345" required></label>
        <button class="btn secondary" type="submit">Extract ←</button>
      </form>
    </div>
  </div>

  <div class="row">
    <div class="card" style="flex:1; min-width: 340px;">
      <h2>Embed (Audio WAV)</h2>
      <form action="{{ url_for('embed_audio') }}" method="post" enctype="multipart/form-data">
        <label>Cover audio (WAV/PCM 8/16-bit): <input required type="file" name="cover_wav" accept=".wav,audio/wav"></label>
        <label>Payload (any file): <input required type="file" name="payload_wav"></label>
        <label>LSBs (1–8): <input type="number" name="lsb" min="1" max="8" value="1" required></label>
        <label>Key (integer): <input type="number" name="key" value="12345" required></label>
        <button class="btn" type="submit">Embed (WAV) →</button>
      </form>
      <p class="note">Only uncompressed PCM WAV is supported.</p>
    </div>

    <div class="card" style="flex:1; min-width: 340px;">
      <h2>Extract (Audio WAV)</h2>
      <form action="{{ url_for('extract_audio') }}" method="post" enctype="multipart/form-data">
        <label>Stego audio (WAV/PCM): <input required type="file" name="stego_wav" accept=".wav,audio/wav"></label>
        <label>LSBs (1–8): <input type="number" name="lsb" min="1" max="8" value="1" required></label>
        <label>Key (integer): <input type="number" name="key" value="12345" required></label>
        <button class="btn secondary" type="submit">Extract (WAV) ←</button>
      </form>
    </div>
  </div>

  {% if result %}
    <hr>
    <h2>Result</h2>
    <div class="row">
      {% if result.cover_preview %}
        <div class="card preview">
          <h3>Cover image preview</h3>
          <img src="{{ result.cover_preview }}">
          {% if result.cover_meta %}
            <p class="muted">{{ result.cover_meta }}</p>
          {% endif %}
        </div>
      {% endif %}

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

      {% if result.cover_audio_preview %}
        <div class="card preview">
          <h3>Cover audio preview</h3>
          <audio controls src="{{ result.cover_audio_preview }}"></audio>
          {% if result.audio_meta %}
            <p class="muted">{{ result.audio_meta }}</p>
          {% endif %}
        </div>
      {% endif %}

      {% if result.stego_audio_preview %}
        <div class="card preview">
          <h3>Stego audio preview</h3>
          <audio controls src="{{ result.stego_audio_preview }}"></audio>
          {% if result.audio_diff_note %}
            <p class="muted">{{ result.audio_diff_note }}</p>
          {% endif %}
        </div>
      {% endif %}

      {% if result.capacity %}
        <div class="card" style="min-width:320px;">
          <h3>Capacity Info</h3>
          <table class="kv">
            <tr><td class="key">Cover capacity</td><td>{{ result.capacity.cover_bytes }} bytes</td></tr>
            <tr><td class="key">Header size</td><td>{{ result.capacity.header_bytes }} bytes</td></tr>
            <tr><td class="key">Payload size</td><td>{{ result.capacity.payload_bytes }} bytes</td></tr>
            <tr><td class="key">Required total</td><td>{{ result.capacity.required_bytes }} bytes</td></tr>
            <tr><td class="key">Utilization</td><td>{{ result.capacity.utilization }}%</td></tr>
            <tr><td class="key">LSBs used</td><td>{{ result.capacity.n_lsb }}</td></tr>
            {% if result.capacity.hw %}<tr><td class="key">Image size</td><td>{{ result.capacity.hw }}</td></tr>{% endif %}
            {% if result.capacity.wav_fmt %}<tr><td class="key">Audio format</td><td>{{ result.capacity.wav_fmt }}</td></tr>{% endif %}
          </table>
        </div>
      {% endif %}
    </div>

    {% if result.download_url %}
      <p><a class="btn" href="{{ result.download_url }}">Download stego image</a></p>
    {% endif %}
    {% if result.payload_name %}
      <p><a class="btn" href="{{ url_for('download_payload') }}">Download extracted payload ({{ result.payload_name }})</a></p>
    {% endif %}
    {% if result.download_wav_url %}
      <p><a class="btn" href="{{ result.download_wav_url }}">Download stego audio</a></p>
    {% endif %}
  {% endif %}
</body>
</html>
"""

# memory store for last downloads (simple demo only)
_LAST_STEGO: bytes = b""
_LAST_STEGO_WAV: bytes = b""
_LAST_PAYLOAD: Tuple[bytes, str] = (b"", "recovered.bin")

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, result=None)

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

        # Read payload & header (store full original filename)
        payload_bytes = payload.read()
        orig_name = os.path.basename(payload.filename or "payload.bin")
        header = build_header(payload_bytes, orig_name)
        data_bits = bytes_to_bits(header + payload_bytes)
        header_bytes_len = len(header)
        required_bytes = header_bytes_len + len(payload_bytes)


        # Cover image
        cover.stream.seek(0)
        flat, shape = load_image_from_file(cover)
        cap_bits = image_capacity_bits(flat.size, n_lsb)
        cap_bytes = cap_bits // 8
        if data_bits.size > cap_bits:
            flash(f"Cover too small. Capacity={cap_bytes} bytes, Needed={required_bytes} bytes")
            return redirect(url_for("index"))

        # Embed
        stego_flat = embed_bits_into_bytes(flat, data_bits, n_lsb, key)
        stego_png = save_image_to_bytes(stego_flat, shape)
        _LAST_STEGO = stego_png

        # Previews for comparison
        cover_img = Image.fromarray(flat.reshape(shape[0], shape[1], 3).astype(np.uint8), mode="RGB")
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

        # Step 1: read fixed 72-bit header (MAGIC + payload_len + name_len)
        fixed_bits = extract_bits_from_bytes(flat, n_lsb, key, 72)
        fixed = bits_to_bytes(fixed_bits)
        if fixed[:4] != MAGIC:
            raise ValueError("Invalid magic / wrong key.")
        payload_len = struct.unpack(">I", fixed[4:8])[0]
        name_len = fixed[8]

        # Step 2: read full header to get filename
        header_bits_len = 72 + name_len * 8
        header_bits = extract_bits_from_bytes(flat, n_lsb, key, header_bits_len)
        header_blob = bits_to_bytes(header_bits)
        filename = header_blob[9:9 + name_len].decode("utf-8", errors="ignore") if name_len else "recovered.bin"

        # Step 3: read header + payload and slice payload
        total_bits = header_bits_len + payload_len * 8
        bits = extract_bits_from_bytes(flat, n_lsb, key, total_bits)
        blob = bits_to_bytes(bits)
        data = blob[9 + name_len : 9 + name_len + payload_len]

        # Save with original filename (sanitized)
        fname = secure_filename(filename)
        if not os.path.splitext(fname)[1]:
            fname += ".bin"
        _LAST_PAYLOAD = (data, fname)

        # Optional: preview
        stego_img = Image.fromarray(flat.reshape(shape[0], shape[1], 3).astype(np.uint8), mode="RGB")
        stego_preview = img_to_data_url(stego_img)

        result = {"payload_name": fname, "stego_preview": stego_preview}
        return render_template_string(INDEX_HTML, result=result)
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

        # Read payload & header (store full original filename)
        payload_bytes = payload.read()
        orig_name = os.path.basename(payload.filename or "payload.bin")
        header = build_header(payload_bytes, orig_name)
        data_bits = bytes_to_bits(header + payload_bytes)
        header_bytes_len = len(header)
        required_bytes = header_bytes_len + len(payload_bytes)

        # Cover WAV
        cover.stream.seek(0)
        flat_bytes, wav_params, raw_frames = load_wav_from_file(cover)
        cap_bits = wav_capacity_bits(flat_bytes.size, n_lsb)
        cap_bytes = cap_bits // 8
        if data_bits.size > cap_bits:
            flash(f"WAV too small. Capacity={cap_bytes} bytes, Needed={required_bytes} bytes")
            return redirect(url_for("index"))

        # Embed
        stego_flat = embed_bits_into_bytes(flat_bytes, data_bits, n_lsb, key)
        stego_wav = save_wav_to_bytes(stego_flat, wav_params)
        _LAST_STEGO_WAV = stego_wav

        # Previews & meta
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
        return render_template_string(INDEX_HTML, result=result)
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

        # Step 1: fixed header
        fixed_bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, 72)
        fixed = bits_to_bytes(fixed_bits)
        if fixed[:4] != MAGIC:
            raise ValueError("Invalid magic / wrong key.")
        payload_len = struct.unpack(">I", fixed[4:8])[0]
        name_len = fixed[8]

        # Step 2: full header to get filename
        header_bits_len = 72 + name_len * 8
        header_bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, header_bits_len)
        header_blob = bits_to_bytes(header_bits)
        filename = header_blob[9:9 + name_len].decode("utf-8", errors="ignore") if name_len else "recovered.bin"

        # Step 3: header + payload
        total_bits = header_bits_len + payload_len * 8
        bits = extract_bits_from_bytes(flat_bytes, n_lsb, key, total_bits)
        blob = bits_to_bytes(bits)
        data = blob[9 + name_len : 9 + name_len + payload_len]

        # Save with original filename (sanitized)
        fname = secure_filename(filename)
        if not os.path.splitext(fname)[1]:
            fname += ".bin"
        _LAST_PAYLOAD = (data, fname)

        stego_audio_preview = audio_to_data_url(save_wav_to_bytes(flat_bytes, wav_params))
        result = {"payload_name": fname, "stego_audio_preview": stego_audio_preview}
        return render_template_string(INDEX_HTML, result=result)
    except Exception as e:
        flash(f"Extract WAV error: {e}")
        return redirect(url_for("index"))


@app.get("/download/payload")
def download_payload():
    data, fname = _LAST_PAYLOAD
    if not data:
        return redirect(url_for("index"))

    #pick MIME based on extension
    ext = os.path.splitext(fname)[1]
    _name, mime = _safe_download_name_from_ext(ext if ext else ".bin")

    return send_file(
        io.BytesIO(data),
        as_attachment=True,
        download_name=fname,     # includes .pdf/.png/…
        mimetype=mime            # hint for browsers
    )


if __name__ == "__main__":
    app.run(debug=True)
