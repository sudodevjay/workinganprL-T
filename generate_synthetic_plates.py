"""Generates synthetic Indian-plate-style crops to close gaps found in the
collected dataset:

1. Zero real examples at length 11 (2-letter state + 2-digit RTO + 3-letter
   series + 4-digit number) - the most character-crowded standard format,
   and therefore the one most exposed to CTC character-dropping.
2. Letters that appear in real plates but are rare in our collected sample
   (Q, S, Z, G, Y, X, V - all legitimate per RTO rules, just uncommon) get
   deliberately over-represented in the random series/letters drawn here,
   so the model sees more contrastive examples of exactly the characters
   users report getting confused for one another.

Real plates never contain I or O in the series letters (RTO doesn't issue
them - confirmed against Wikipedia's Vehicle registration plates of India
article), so those two are excluded from the letter pool entirely, matching
what real data would show if it existed.

Rendered onto a plate-style background and then run through the same kind
of degradation real photographed crops show (blur, jpeg noise, brightness
jitter, slight rotation) so these don't look "too clean" relative to the
photographed training data - a synthetic example that's crisper than
   anything the model will see at inference just teaches it to expect an
   unrealistic input distribution.
4. Bike/truck rear plates are often stacked across two lines, with the
   state/RTO code on the top row and the series/number on the bottom row.
   Real stacked crops are sparse in the photographed data, so this generator
   deliberately adds two-line crops with white, yellow/commercial, and green
   plate backgrounds.
"""
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "ocr_training", "synthetic")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
]
FONTS = [ImageFont.truetype(f, 62) for f in FONT_CANDIDATES if os.path.exists(f)]
SMALL_FONTS = [ImageFont.truetype(f, 54) for f in FONT_CANDIDATES if os.path.exists(f)]
if not FONTS:
    FONTS = [ImageFont.load_default()]
if not SMALL_FONTS:
    SMALL_FONTS = FONTS

STATE_CODES = [
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PN", "PY", "RJ", "SK", "TN", "TG", "TR",
    "TS", "UA", "UK", "UP", "WB",
]
# I and O never appear in series letters (RTO rule) - excluded from the pool.
# Q,S,Z,G,Y,X,V weighted up since they're the real-but-rare/confusable ones
# users reported trouble with; common letters still included so the pool
# isn't artificially skewed away from realistic frequency entirely.
ALLOWED_SERIES_LETTERS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
LETTER_POOL = (
    ALLOWED_SERIES_LETTERS * 4
    + list("EFUW") * 2
    + list("BDGQSXYZ") * 1
)
DIGIT_POOL = list("0123456789")

PLATE_STYLES = [
    # (background, text_color, weight)
    ((255, 255, 255), (10, 10, 10), 6),   # white/private
    ((255, 200, 30), (10, 10, 10), 2),    # yellow/commercial
    ((10, 110, 60), (255, 255, 255), 1),  # green/EV
]


def clear_generated_synthetic_dir():
    removed = 0
    for name in os.listdir(OUT_DIR):
        if name.startswith("synth_") and name.lower().endswith(".jpg"):
            os.remove(os.path.join(OUT_DIR, name))
            removed += 1
    if removed:
        print(f"Removed {removed} old generated synthetic crops")


def random_plate_text(length_mode):
    state = random.choice(STATE_CODES)
    if length_mode == 11:
        rto = f"{random.randint(1, 99):02d}"
        series = "".join(random.choices(LETTER_POOL, k=3))
        number = f"{random.randint(1, 9999):04d}"
    elif length_mode == 10:
        rto = f"{random.randint(1, 99):02d}"
        series = "".join(random.choices(LETTER_POOL, k=2))
        number = f"{random.randint(1, 9999):04d}"
    elif length_mode == 9:
        rto = f"{random.randint(1, 9):01d}"
        series = "".join(random.choices(LETTER_POOL, k=2))
        number = f"{random.randint(1, 9999):04d}"
    else:  # 8
        rto = f"{random.randint(1, 9):01d}"
        series = "".join(random.choices(LETTER_POOL, k=1))
        number = f"{random.randint(1, 9999):04d}"
    return state + rto + series + number


def render_plate(text):
    bg, fg, _w = random.choices(PLATE_STYLES, weights=[p[2] for p in PLATE_STYLES])[0]
    font = random.choice(FONTS)

    pad_x, pad_y = 22, 14
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    img_w, img_h = text_w + pad_x * 2, text_h + pad_y * 2
    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, font=font, fill=fg)

    border = random.randint(3, 6)
    draw.rectangle([0, 0, img_w - 1, img_h - 1], outline=(30, 30, 30), width=border)
    return np.array(img)[:, :, ::-1].copy()  # RGB -> BGR


def split_two_line_text(text):
    rto_digits = 1 if text[2].isdigit() and not text[3].isdigit() else 2
    top_len = 2 + rto_digits
    return text[:top_len], text[top_len:]


def render_two_line_plate(text):
    bg, fg, _w = random.choices(PLATE_STYLES, weights=[p[2] for p in PLATE_STYLES])[0]
    top_text, bottom_text = split_two_line_text(text)
    top_font = random.choice(SMALL_FONTS)
    bottom_font = random.choice(FONTS)

    dummy = Image.new("RGB", (10, 10))
    measure = ImageDraw.Draw(dummy)
    top_bbox = measure.textbbox((0, 0), top_text, font=top_font)
    bottom_bbox = measure.textbbox((0, 0), bottom_text, font=bottom_font)
    top_w, top_h = top_bbox[2] - top_bbox[0], top_bbox[3] - top_bbox[1]
    bottom_w, bottom_h = bottom_bbox[2] - bottom_bbox[0], bottom_bbox[3] - bottom_bbox[1]

    pad_x = random.randint(22, 34)
    gap = random.randint(8, 18)
    img_w = max(top_w, bottom_w) + pad_x * 2
    base_h = top_h + bottom_h + gap + random.randint(34, 52)
    target_aspect = random.uniform(1.15, 1.75)
    img_h = max(base_h, int(img_w / target_aspect))

    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)
    border = random.randint(4, 7)
    draw.rectangle([0, 0, img_w - 1, img_h - 1], outline=(30, 30, 30), width=border)

    text_block_h = top_h + gap + bottom_h
    y = (img_h - text_block_h) // 2
    top_x = (img_w - top_w) // 2 + random.randint(-4, 4)
    bottom_x = (img_w - bottom_w) // 2 + random.randint(-4, 4)
    draw.text((top_x - top_bbox[0], y - top_bbox[1]), top_text, font=top_font, fill=fg)
    draw.text(
        (bottom_x - bottom_bbox[0], y + top_h + gap - bottom_bbox[1]),
        bottom_text,
        font=bottom_font,
        fill=fg,
    )

    if random.random() < 0.35:
        screw_color = tuple(max(0, min(255, c + random.randint(-35, 35))) for c in bg)
        radius = random.randint(3, 5)
        for x in (pad_x // 2, img_w - pad_x // 2):
            draw.ellipse(
                [x - radius, img_h // 2 - radius, x + radius, img_h // 2 + radius],
                fill=screw_color,
                outline=(80, 80, 80),
            )

    return np.array(img)[:, :, ::-1].copy()  # RGB -> BGR


def degrade(img):
    h, w = img.shape[:2]

    angle = random.uniform(-6, 6)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if random.random() < 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    alpha = random.uniform(0.75, 1.25)
    beta = random.uniform(-25, 25)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.6:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    scale = random.uniform(0.5, 0.85)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 70)])[1]
    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return img


def run():
    random.seed(SEED)
    np.random.seed(SEED)
    clear_generated_synthetic_dir()

    plan = [
        ("single", 11, 400),
        ("single", 8, 100),
        ("two_line", 10, 350),
        ("two_line", 9, 250),
        ("two_line", 11, 150),
        ("two_line", 8, 100),
    ]
    manifest = []
    idx = 0
    for layout, length_mode, count in plan:
        for _ in range(count):
            text = random_plate_text(length_mode)
            img = render_two_line_plate(text) if layout == "two_line" else render_plate(text)
            img = degrade(img)
            fname = f"synth_{layout}_{length_mode}_{idx:04d}_{text}.jpg"
            cv2.imwrite(os.path.join(OUT_DIR, fname), img)
            manifest.append((fname, text))
            idx += 1

    print(f"Generated {len(manifest)} synthetic plates -> {OUT_DIR}")
    lengths = {}
    for _f, t in manifest:
        lengths[len(t)] = lengths.get(len(t), 0) + 1
    print("Length distribution:", lengths)
    layouts = {}
    for f, _t in manifest:
        layout = "two_line" if f.startswith("synth_two_line_") else "single"
        layouts[layout] = layouts.get(layout, 0) + 1
    print("Layout distribution:", layouts)
    return manifest


if __name__ == "__main__":
    run()
