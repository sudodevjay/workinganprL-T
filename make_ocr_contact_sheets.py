"""Builds labeled contact-sheet grids (like archive/make_contact_sheets.py)
but scoped to only the crops in ocr_training/labels_draft.tsv that still have
an empty label -- so the sheets cover exactly what still needs a human/AI
transcription pass, not the whole 803-image set again."""
import csv
import json
import os

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROPS_DIR = os.path.join(BASE_DIR, "ocr_training", "crops")
LABELS_PATH = os.path.join(BASE_DIR, "ocr_training", "labels_draft.tsv")
SHEETS_DIR = os.path.join(BASE_DIR, "ocr_training", "contact_sheets")
MANIFEST_PATH = os.path.join(BASE_DIR, "ocr_training", "contact_sheets_manifest.json")

os.makedirs(SHEETS_DIR, exist_ok=True)

COLS, ROWS = 4, 4
PER_SHEET = COLS * ROWS
CELL_W, CELL_H = 460, 220
IMG_H = 170
PADDING = 6

try:
    font = ImageFont.truetype("consola.ttf", 16)
except OSError:
    font = ImageFont.load_default()

with open(LABELS_PATH, encoding="utf-8") as f:
    rows = list(csv.reader(f, delimiter="\t"))

empty_filenames = [row[0] for row in rows if len(row) < 2 or not row[1].strip()]
print(f"{len(empty_filenames)} crops still need a label (of {len(rows)} total)")

manifest = {}
for sheet_idx in range(0, len(empty_filenames), PER_SHEET):
    batch = empty_filenames[sheet_idx:sheet_idx + PER_SHEET]
    sheet_num = sheet_idx // PER_SHEET

    sheet = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), "white")
    draw = ImageDraw.Draw(sheet)

    for i, fname in enumerate(batch):
        col, row = i % COLS, i // COLS
        x0, y0 = col * CELL_W, row * CELL_H

        crop = Image.open(os.path.join(CROPS_DIR, fname)).convert("RGB")
        crop.thumbnail((CELL_W - 2 * PADDING, IMG_H))
        paste_x = x0 + (CELL_W - crop.width) // 2
        paste_y = y0 + PADDING
        sheet.paste(crop, (paste_x, paste_y))

        draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], outline="black", width=1)
        label = os.path.splitext(fname)[0]
        draw.text((x0 + PADDING, y0 + IMG_H + PADDING + 4), f"[{i}] {label}", fill="black", font=font)

    out_path = os.path.join(SHEETS_DIR, f"sheet_{sheet_num:03d}.png")
    sheet.save(out_path)
    manifest[sheet_num] = batch

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)

print(f"Done. {len(manifest)} sheets written to {SHEETS_DIR}")
