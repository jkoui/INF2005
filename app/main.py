import os
import struct
import wave
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import io
import sys
from typing import Tuple, Optional

# ============================================================
# ACW1 header: MAGIC(4) + PAYLOAD_LEN(4) + EXT_LEN(1) + EXT(EXT_LEN)
MAGIC = b"ACW1"

def bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.uint8)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        bits = np.pad(bits, (0, 8 - (bits.size % 8)))
    b = np.packbits(bits.astype(np.uint8)).tobytes()
    return b

def build_header(payload: bytes, ext: str) -> bytes:
    ext_bytes = ext.encode("utf-8")
    if len(ext_bytes) > 255:
        raise ValueError("File extension too long.")
    return MAGIC + struct.pack(">I", len(payload)) + struct.pack("B", len(ext_bytes)) + ext_bytes

def parse_header(bit_reader) -> Tuple[int, str]:
    # Read fixed 9 bytes = 72 bits
    fixed_bits = bit_reader(72)
    fixed = bits_to_bytes(fixed_bits)
    if fixed[:4] != MAGIC:
        raise ValueError("Invalid magic / wrong key.")
    payload_len = struct.unpack(">I", fixed[4:8])[0]
    ext_len = fixed[8]
    if ext_len:
        ext_bits = bit_reader(ext_len * 8)
        ext = bits_to_bytes(ext_bits).decode("utf-8", errors="ignore")
    else:
        ext = ""
    return payload_len, ext

def prng_perm(total_slots: int, key: int) -> np.ndarray:
    # Deterministic permutation, independent of platform
    rng = np.random.default_rng(seed=int(key) & 0xFFFFFFFF)
    return rng.permutation(total_slots)

# ============================================================
# IMAGE STEGO (PNG/BMP, RGB)
def load_image_bytes(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # H,W,3
    h, w, c = arr.shape
    flat = arr.reshape(-1)  # bytes view
    return flat, (h, w)

def save_image_bytes(flat: np.ndarray, shape: Tuple[int,int], path: str):
    h, w = shape
    arr = flat.reshape(h, w, 3).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)

def image_capacity_bits(flat_bytes_len: int, n_lsb: int) -> int:
    return flat_bytes_len * n_lsb

def embed_bits_into_image(flat: np.ndarray, data_bits: np.ndarray, n_lsb: int, key: int) -> np.ndarray:
    flat = flat.copy()
    total_slots = flat.size * n_lsb  # slot = (byte_index, lsb_pos)
    if data_bits.size > total_slots:
        raise ValueError("Payload too large for selected LSBs / cover.")

    # We map sequential data bits to slots determined by a permutation
    # We'll fill slots in order; within each byte, slots correspond to bit positions [0..n_lsb-1]
    perm = prng_perm(total_slots, key)
    # For efficient addressing, compute target byte indexes and bit positions
    tgt_slot_idx = perm[:data_bits.size]
    byte_idx = tgt_slot_idx // n_lsb
    bit_pos = (tgt_slot_idx % n_lsb).astype(np.uint8)

    # Clear & set
    mask_clear = ~(1 << bit_pos)
    # Because bit_pos varies per element, we need element-wise operations
    # Convert to uint16 to avoid overflow warnings
    vals = flat[byte_idx].astype(np.uint16)
    vals = vals & mask_clear
    vals = vals | ((data_bits.astype(np.uint16) & 1) << bit_pos)
    flat[byte_idx] = vals.astype(np.uint8)
    return flat

def extract_bits_from_image(flat: np.ndarray, n_lsb: int, key: int, num_bits: int) -> np.ndarray:
    total_slots = flat.size * n_lsb
    if num_bits > total_slots:
        raise ValueError("Requested bits exceed capacity.")
    perm = prng_perm(total_slots, key)
    tgt_slot_idx = perm[:num_bits]
    byte_idx = tgt_slot_idx // n_lsb
    bit_pos = (tgt_slot_idx % n_lsb).astype(np.uint8)
    vals = flat[byte_idx]
    bits = (vals >> bit_pos) & 1
    return bits.astype(np.uint8)

def image_lsb_plane(flat: np.ndarray, n_lsb: int, shape: Tuple[int,int]) -> Image.Image:
    # Visualize combined LSBs as grayscale (scale up to 0..255)
    mask = (1 << n_lsb) - 1
    vals = (flat & mask).astype(np.uint16)
    # Normalize to 0..255
    if mask > 0:
        scaled = (vals * (255 // mask)).astype(np.uint8)
    else:
        scaled = vals.astype(np.uint8)
    h, w = shape
    img = scaled.reshape(h, w, 3)
    gray = np.mean(img, axis=2).astype(np.uint8)
    return Image.fromarray(gray, mode="L")

def image_diff(original_flat: np.ndarray, stego_flat: np.ndarray, shape: Tuple[int,int]) -> Image.Image:
    diff = np.abs(stego_flat.astype(np.int16) - original_flat.astype(np.int16)).astype(np.uint8)
    h, w = shape
    img = diff.reshape(h, w, 3)
    # Emphasize differences
    img = np.clip(img * 64, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")

# ============================================================
# WAV (16-bit PCM) STEGO (basic)
def load_wav_samples(path: str) -> Tuple[np.ndarray, dict]:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM WAV supported.")
        raw = wf.readframes(n_frames)
    # int16 little-endian
    samples = np.frombuffer(raw, dtype=np.int16)
    meta = {"n_channels": n_channels, "framerate": framerate, "sampwidth": sampwidth}
    return samples.copy(), meta

def save_wav_samples(samples: np.ndarray, meta: dict, path: str):
    samples = samples.astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(meta["n_channels"])
        wf.setsampwidth(meta["sampwidth"])
        wf.setframerate(meta["framerate"])
        wf.writeframes(samples.tobytes())

def wav_capacity_bits(samples_len: int, n_lsb: int) -> int:
    return samples_len * n_lsb  # per sample

def embed_bits_into_wav(samples: np.ndarray, data_bits: np.ndarray, n_lsb: int, key: int) -> np.ndarray:
    samples = samples.copy().astype(np.int32)  # widen for bit ops
    total_slots = samples.size * n_lsb
    if data_bits.size > total_slots:
        raise ValueError("Payload too large for selected LSBs / cover.")
    perm = prng_perm(total_slots, key)
    tgt_slot_idx = perm[:data_bits.size]
    sample_idx = tgt_slot_idx // n_lsb
    bit_pos = (tgt_slot_idx % n_lsb).astype(np.uint8)

    vals = samples[sample_idx]
    mask_clear = ~(1 << bit_pos)
    vals = vals & mask_clear
    vals = vals | ((data_bits.astype(np.int32) & 1) << bit_pos)
    samples[sample_idx] = vals
    return samples.astype(np.int16)

def extract_bits_from_wav(samples: np.ndarray, n_lsb: int, key: int, num_bits: int) -> np.ndarray:
    total_slots = samples.size * n_lsb
    if num_bits > total_slots:
        raise ValueError("Requested bits exceed capacity.")
    perm = prng_perm(total_slots, key)
    tgt_slot_idx = perm[:num_bits]
    sample_idx = tgt_slot_idx // n_lsb
    bit_pos = (tgt_slot_idx % n_lsb).astype(np.uint8)
    vals = samples[sample_idx].astype(np.int32)
    bits = (vals >> bit_pos) & 1
    return bits.astype(np.uint8)

# ============================================================
# GUI
class App:
    def __init__(self, root):
        self.root = root
        root.title("ACW1 Stego (LSB, Keyed)")

        self.cover_path = None
        self.payload_path = None
        self.stego_bytes = None
        self.cover_kind = None  # "image" or "wav"
        self.image_cache = {}   # for display

        # Controls
        frm = tk.Frame(root)
        frm.pack(padx=10, pady=10)

        self.btn_cover = tk.Button(frm, text="Choose Cover (PNG/BMP/WAV)", command=self.choose_cover)
        self.btn_cover.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        self.lbl_cover = tk.Label(frm, text="No cover selected")
        self.lbl_cover.grid(row=0, column=1, sticky="w")

        self.btn_payload = tk.Button(frm, text="Choose Payload (any file)", command=self.choose_payload)
        self.btn_payload.grid(row=1, column=0, sticky="ew", padx=4, pady=4)

        self.lbl_payload = tk.Label(frm, text="No payload selected")
        self.lbl_payload.grid(row=1, column=1, sticky="w")

        tk.Label(frm, text="LSBs (1-8):").grid(row=2, column=0, sticky="e")
        self.lsb_var = tk.IntVar(value=1)
        self.scl_lsb = tk.Scale(frm, from_=1, to=8, orient="horizontal", variable=self.lsb_var)
        self.scl_lsb.grid(row=2, column=1, sticky="w")

        tk.Label(frm, text="Key (integer):").grid(row=3, column=0, sticky="e")
        self.ent_key = tk.Entry(frm)
        self.ent_key.insert(0, "12345")
        self.ent_key.grid(row=3, column=1, sticky="w")

        self.btn_embed = tk.Button(frm, text="Embed →", command=self.do_embed)
        self.btn_embed.grid(row=4, column=0, sticky="ew", padx=4, pady=4)

        self.btn_save_stego = tk.Button(frm, text="Save Stego", command=self.save_stego, state="disabled")
        self.btn_save_stego.grid(row=4, column=1, sticky="w", padx=4, pady=4)

        self.btn_extract = tk.Button(frm, text="Extract ←", command=self.do_extract)
        self.btn_extract.grid(row=5, column=0, sticky="ew", padx=4, pady=4)

        # Image panels
        self.canvas_cover = tk.Label(root)
        self.canvas_cover.pack(side="left", padx=10, pady=10)

        self.canvas_stego = tk.Label(root)
        self.canvas_stego.pack(side="left", padx=10, pady=10)

        self.canvas_viz = tk.Label(root)
        self.canvas_viz.pack(side="left", padx=10, pady=10)

        self.status = tk.Label(root, text="Ready", anchor="w")
        self.status.pack(fill="x")

    # ------------- UI helpers -------------
    def choose_cover(self):
        path = filedialog.askopenfilename(title="Choose cover", filetypes=[
            ("Image/Audio", "*.png *.bmp *.wav"),
            ("All files", "*.*"),
        ])
        if not path:
            return
        self.cover_path = path
        self.cover_kind = "wav" if path.lower().endswith(".wav") else "image"
        self.lbl_cover.config(text=os.path.basename(path))
        self.update_cover_preview()

    def choose_payload(self):
        path = filedialog.askopenfilename(title="Choose payload", filetypes=[("All files", "*.*")])
        if not path:
            return
        self.payload_path = path
        self.lbl_payload.config(text=os.path.basename(path))

    def get_key(self) -> int:
        try:
            return int(self.ent_key.get())
        except:
            raise ValueError("Key must be an integer.")

    def update_cover_preview(self):
        if self.cover_kind == "image":
            try:
                img = Image.open(self.cover_path).convert("RGB")
                self.show_image(self.canvas_cover, img, tag="cover")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot preview image: {e}")
        else:
            # For WAV show a simple banner
            banner = Image.new("RGB", (320, 240), (30, 30, 60))
            self.draw_text(banner, "WAV cover selected")
            self.show_image(self.canvas_cover, banner, tag="cover")

    def show_image(self, widget, img: Image.Image, tag: str):
        max_w, max_h = 320, 240
        img2 = img.copy()
        img2.thumbnail((max_w, max_h))
        tkimg = ImageTk.PhotoImage(img2)
        widget.config(image=tkimg)
        widget.image = tkimg
        self.image_cache[tag] = img2

    def draw_text(self, img: Image.Image, text: str):
        # simple centered text using Pillow basic drawing
        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.load_default()
        except:
            f = None
        w, h = d.textsize(text, font=f)
        W, H = img.size
        d.text(((W-w)//2, (H-h)//2), text, fill=(220,220,220), font=f)

    # ------------- Actions -------------
    def do_embed(self):
        if not self.cover_path:
            messagebox.showwarning("Missing", "Choose a cover file first.")
            return
        if not self.payload_path:
            messagebox.showwarning("Missing", "Choose a payload file first.")
            return
        try:
            n_lsb = self.lsb_var.get()
            key = self.get_key()
            with open(self.payload_path, "rb") as f:
                payload = f.read()
            ext = os.path.splitext(self.payload_path)[1]  # e.g., ".pdf"
            header = build_header(payload, ext)
            all_bytes = header + payload
            data_bits = bytes_to_bits(all_bytes)

            if self.cover_kind == "image":
                flat, shape = load_image_bytes(self.cover_path)
                cap = image_capacity_bits(flat.size, n_lsb)
                if data_bits.size > cap:
                    messagebox.showerror("Capacity", f"Cover too small.\nCapacity: {cap//8} bytes\nNeeded: {data_bits.size//8} bytes")
                    return
                stego_flat = embed_bits_into_image(flat, data_bits, n_lsb, key)
                # For preview only (not yet saved)
                self.stego_bytes = ("image", stego_flat, shape)
                # Previews
                self.show_image(self.canvas_stego, Image.open(self.cover_path).convert("RGB"), tag="stego_orig_like")
                # Viz
                lsb_img = image_lsb_plane(stego_flat, n_lsb, shape)
                self.show_image(self.canvas_viz, lsb_img.convert("RGB"), tag="viz")
                # Also show diff (optional): diff between cover and stego
                try:
                    diff_img = image_diff(flat, stego_flat, shape)
                    self.show_image(self.canvas_stego, diff_img, tag="stego_diff")
                except Exception:
                    pass

            else:  # WAV
                samples, meta = load_wav_samples(self.cover_path)
                cap = wav_capacity_bits(samples.size, n_lsb)
                if data_bits.size > cap:
                    messagebox.showerror("Capacity", f"WAV cover too small.\nCapacity: {cap//8} bytes\nNeeded: {data_bits.size//8} bytes")
                    return
                stego_samples = embed_bits_into_wav(samples, data_bits, n_lsb, key)
                self.stego_bytes = ("wav", stego_samples, meta)
                banner = Image.new("RGB", (320, 240), (60, 30, 30))
                self.draw_text(banner, "WAV stego ready")
                self.show_image(self.canvas_stego, banner, tag="stego")
                self.show_image(self.canvas_viz, banner, tag="viz")

            self.status.config(text="Embed OK. Click 'Save Stego' to write to file.")
            self.btn_save_stego.config(state="normal")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Embed failed.")

    def save_stego(self):
        if not self.stego_bytes:
            return
        kind = self.stego_bytes[0]
        if kind == "image":
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("BMP", "*.bmp")])
            if not path:
                return
            _, stego_flat, shape = self.stego_bytes
            try:
                save_image_bytes(stego_flat, shape, path)
                messagebox.showinfo("Saved", f"Stego image saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
            if not path:
                return
            _, stego_samples, meta = self.stego_bytes
            try:
                save_wav_samples(stego_samples, meta, path)
                messagebox.showinfo("Saved", f"Stego WAV saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def do_extract(self):
        stego_path = filedialog.askopenfilename(title="Choose stego file", filetypes=[
            ("Image/Audio", "*.png *.bmp *.wav"), ("All files", "*.*")
        ])
        if not stego_path:
            return
        try:
            n_lsb = self.lsb_var.get()
            key = self.get_key()

            if stego_path.lower().endswith(".wav"):
                samples, meta = load_wav_samples(stego_path)
                # First read header fixed part to know sizes:
                def reader(num_bits):
                    return extract_bits_from_wav(samples, n_lsb, key, num_bits)
                payload_len, ext = parse_header(reader)
                header_bits_len = 72 + len(ext.encode("utf-8")) * 8
                total_bits = header_bits_len + payload_len * 8
                bits = extract_bits_from_wav(samples, n_lsb, key, total_bits)
            else:
                flat, shape = load_image_bytes(stego_path)
                def reader(num_bits):
                    return extract_bits_from_image(flat, n_lsb, key, num_bits)
                payload_len, ext = parse_header(reader)
                header_bits_len = 72 + len(ext.encode("utf-8")) * 8
                total_bits = header_bits_len + payload_len * 8
                bits = extract_bits_from_image(flat, n_lsb, key, total_bits)

            blob = bits_to_bytes(bits)
            # Strip header
            fixed = 9  # 4 + 4 + 1
            ext_len = blob[8]
            header_total = 9 + ext_len
            data = blob[header_total:]
            data = data[:payload_len]

            # Save dialog
            default_name = "recovered" + (ext if ext else ".bin")
            save_path = filedialog.asksaveasfilename(initialfile=default_name)
            if not save_path:
                return
            with open(save_path, "wb") as f:
                f.write(data)
            messagebox.showinfo("Extracted", f"Payload saved to:\n{save_path}")
            self.status.config(text="Extract OK.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status.config(text="Extract failed.")

def main():
    root = tk.Tk()
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
