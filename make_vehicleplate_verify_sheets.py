"""Contact sheets for manually verifying main.py's OCR output against the
actual plate text, on the clean/unseen vehicleplate/ test set. Each cell
shows a zoomed-in crop of the detected plate region (not the whole vehicle
scene, which is too small to read at thumbnail size) plus the OCR's
predicted text as a caption, so a human/AI reviewer can compare directly."""
import json
import os

import cv2
from PIL import Image, ImageDraw, ImageFont

import main

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "vehicleplate")
RESULTS_PATH = os.path.join(BASE_DIR, "test_outputs", "vehicleplate_clean_test_results.json")
SHEETS_DIR = os.path.join(BASE_DIR, "test_outputs", "vehicleplate_verify_sheets")
MANIFEST_PATH = os.path.join(BASE_DIR, "test_outputs", "vehicleplate_verify_sheets_manifest.json")
os.makedirs(SHEETS_DIR, exist_ok=True)

with open(RESULTS_PATH, encoding="utf-8") as f:
    results = json.load(f)

COLS, ROWS = 4, 4
PER_SHEET = COLS * ROWS
CELL_W, CELL_H = 460, 220
IMG_H = 160
PADDING = 6
try:
    font = ImageFont.truetype("consola.ttf", 16)
except OSError:
    font = ImageFont.load_default()


def get_plate_crop(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    candidates = main.plate_detector.detect_many(image) if main.plate_detector.available else []
    if not candidates:
        return None
    crop = candidates[0]["crop"]
    h, w = crop.shape[:2]
    scale = max(1, 400 // max(h, 1))
    if scale > 1:
        crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


manifest = {}
for sheet_idx in range(0, len(results), PER_SHEET):
    batch = results[sheet_idx:sheet_idx + PER_SHEET]
    sheet_num = sheet_idx // PER_SHEET

    sheet = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), "white")
    draw = ImageDraw.Draw(sheet)

    for i, item in enumerate(batch):
        col, row = i % COLS, i // COLS
        x0, y0 = col * CELL_W, row * CELL_H

        plate_arr = get_plate_crop(os.path.join(SOURCE_DIR, item["source"]))
        if plate_arr is not None:
            crop = Image.fromarray(plate_arr)
            crop.thumbnail((CELL_W - 2 * PADDING, IMG_H))
            paste_x = x0 + (CELL_W - crop.width) // 2
            paste_y = y0 + PADDING
            sheet.paste(crop, (paste_x, paste_y))
        else:
            draw.text((x0 + PADDING, y0 + PADDING + 60), "NO PLATE DETECTED", fill="red", font=font)

        draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], outline="black", width=1)
        label = f"[{i}] OCR: {item['plate_text'] or '(empty)'}"
        draw.text((x0 + PADDING, y0 + IMG_H + PADDING), label, fill="black", font=font)
        draw.text((x0 + PADDING, y0 + IMG_H + PADDING + 20), item["source"], fill="gray", font=font)

    out_path = os.path.join(SHEETS_DIR, f"sheet_{sheet_num:03d}.png")
    sheet.save(out_path)
    manifest[sheet_num] = [item["source"] for item in batch]

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)

print(f"Done. {len(manifest)} sheets written to {SHEETS_DIR}")
