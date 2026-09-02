"""Contact sheets for re-verifying archive/ labels that contain a rare
confusable letter (Q,S,Z,G,Y,X,V) - the archive dataset was hand-labeled in
an earlier session and never independently visually re-checked. Scoped to
just the confusable-letter subset (not all 1798) since that's where a
mislabel actually teaches the model the wrong thing about exactly the
characters users report getting confused."""
import csv
import json
import os
import re

from PIL import Image, ImageDraw, ImageFont

ARCHIVE_DIR = r"D:\bhikaji\bhikajianpr\archive"
CROPS_DIR = os.path.join(ARCHIVE_DIR, "crops")
LABELS_PATH = os.path.join(ARCHIVE_DIR, "labels_draft.tsv")
SHEETS_DIR = os.path.join(ARCHIVE_DIR, "verify_sheets")
MANIFEST_PATH = os.path.join(ARCHIVE_DIR, "verify_sheets_manifest.json")
os.makedirs(SHEETS_DIR, exist_ok=True)

RARE = set("QSZGYXV")


def is_usable(label):
    if not label or label in ("?", "NOT_PLATE"):
        return False
    if "FOREIGN" in label:
        return False
    if label.endswith("?"):
        return False
    return True


entries = []
with open(LABELS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        fname, label = line.split("\t")
        if not is_usable(label):
            continue
        if any(ch in RARE for ch in label):
            entries.append((fname, label))

seen = set()
deduped = []
for fname, label in entries:
    stem = re.sub(r"_\d+\.jpg$", "", fname)
    key = (stem, label)
    if key in seen:
        continue
    seen.add(key)
    deduped.append((fname, label))

print(f"Reviewing {len(deduped)} representative crops")

COLS, ROWS = 4, 4
PER_SHEET = COLS * ROWS
CELL_W, CELL_H = 460, 220
IMG_H = 170
PADDING = 6
try:
    font = ImageFont.truetype("consola.ttf", 15)
except OSError:
    font = ImageFont.load_default()

manifest = {}
for sheet_idx in range(0, len(deduped), PER_SHEET):
    batch = deduped[sheet_idx:sheet_idx + PER_SHEET]
    sheet_num = sheet_idx // PER_SHEET

    sheet = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (fname, label) in enumerate(batch):
        col, row = i % COLS, i // COLS
        x0, y0 = col * CELL_W, row * CELL_H

        crop = Image.open(os.path.join(CROPS_DIR, fname)).convert("RGB")
        crop.thumbnail((CELL_W - 2 * PADDING, IMG_H))
        paste_x = x0 + (CELL_W - crop.width) // 2
        paste_y = y0 + PADDING
        sheet.paste(crop, (paste_x, paste_y))

        draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1], outline="black", width=1)
        draw.text((x0 + PADDING, y0 + IMG_H + PADDING + 4), f"[{i}] {fname} -> {label}", fill="black", font=font)

    out_path = os.path.join(SHEETS_DIR, f"sheet_{sheet_num:03d}.png")
    sheet.save(out_path)
    manifest[sheet_num] = batch

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)

print(f"Done. {len(manifest)} sheets written to {SHEETS_DIR}")
