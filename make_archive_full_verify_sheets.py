"""Full remaining-archive verification sheets - covers every archive/
label NOT already individually visually confirmed in the earlier
rare-confusable-letter pass (see verify_sheets_manifest.json), so that
every single archive entry gets checked against its actual crop image,
not just the confusable-letter subset."""
import json
import os

from PIL import Image, ImageDraw, ImageFont

ARCHIVE_DIR = r"D:\bhikaji\bhikajianpr\archive"
CROPS_DIR = os.path.join(ARCHIVE_DIR, "crops")
LABELS_PATH = os.path.join(ARCHIVE_DIR, "labels_draft.tsv")
PREV_MANIFEST_PATH = os.path.join(ARCHIVE_DIR, "verify_sheets_manifest.json")
SHEETS_DIR = os.path.join(ARCHIVE_DIR, "verify_sheets_full")
MANIFEST_PATH = os.path.join(ARCHIVE_DIR, "verify_sheets_full_manifest.json")
os.makedirs(SHEETS_DIR, exist_ok=True)


def is_usable(label):
    if not label or label in ("?", "NOT_PLATE"):
        return False
    if "FOREIGN" in label:
        return False
    if label.endswith("?"):
        return False
    return True


with open(PREV_MANIFEST_PATH, encoding="utf-8") as f:
    prev_manifest = json.load(f)
already_checked = set()
for _sheet, items in prev_manifest.items():
    for fname, _label in items:
        already_checked.add(fname)

entries = []
with open(LABELS_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        fname, label = line.split("\t")
        if not is_usable(label):
            continue
        if fname in already_checked:
            continue
        entries.append((fname, label))

print(f"Already individually checked: {len(already_checked)}")
print(f"Remaining to check: {len(entries)}")

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
for sheet_idx in range(0, len(entries), PER_SHEET):
    batch = entries[sheet_idx:sheet_idx + PER_SHEET]
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
