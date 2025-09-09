## Features
- Embed and extract payloads from **images (PNG/BMP)** and **WAV audio (16-bit PCM)**.
- User-defined **key (integer)** to influence LSB embedding/extraction.
- Selectable **1–8 LSBs** for embedding.
- Capacity check with error messages if payload is too large.
- Visualization for images:
  - **LSB-plane preview**.
  - **Difference preview** between cover and stego.

## Installation
1. Clone the repo or download the source.
2. Navigate into the project folder. (cd app)
3. Install dependencies with: pip install -r requirements.txt
4. If want to run basic GUI pop up, do python main.py
5. If want to run a basic webpage version, do python web.py

## A watered down explanation of what needs to be done:
1. You start with a cover (a normal picture or a sound file).
This is like a blank notebook where you’ll write your secret.

2. You also have a payload (your secret: could be text, PDF, image, or even a small program 💾).
This is what you want to hide.

3. You press Embed →
Your program sneaks the secret bits into the tiny corners (LSBs) of the cover.
To everyone else, the picture/sound still looks the same.
But inside, it now carries your secret. This is called the stego object.

4. When you press Extract ←
You give the program the stego object + the key (password number) you used.
The program goes back into the tiny corners and pulls the secret out.
If you type the wrong key, you get garbage (not the real secret).

So the steps are:
1. Pick cover (picture or sound).
2. Pick secret (the thing you want to hide).
3. Choose LSBs (how many tiny corners to use) + type a key (your secret number).
4. Embed → → save the new stego file.
5. Extract ← → recover the hidden secret later.