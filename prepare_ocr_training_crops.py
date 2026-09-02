"""Step 1 of the new PaddleOCR training round: build a single crops/ folder
from ocr_training/raw_images/, which is a mix of two kinds of source images:

  - full camera-frame images (whole vehicle/scene visible) -> run the YOLO
    plate detector (main.plate_detector, weights/best.pt) and crop the best
    detected plate box out of it.
  - already plate-only crops (saved earlier by the live pipeline) -> use the
    image as-is, no detection needed (and often none possible, since there's
    no vehicle context left for the detector to key off of).

Every source filename that already encodes a plate-text guess (the
`cam01_HH-MM-SS-ffffff_PLATETEXT.jpg` convention used by the live snapshot
saver) carries that guess forward into labels_draft.tsv as a *draft* label to
be verified/corrected by hand later - it is the pipeline's own OCR reading,
not verified ground truth, so it must not be trusted as-is."""
import csv
import os
import re

import cv2

import main

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "ocr_training", "raw_images")
CROPS_DIR = os.path.join(BASE_DIR, "ocr_training", "crops")
DRAFT_LABELS_PATH = os.path.join(BASE_DIR, "ocr_training", "labels_draft.tsv")

# cam01_11-16-59-133910_TL12L387.jpg -> plate guess "TL12L387"
GUESS_SUFFIX_RE = re.compile(r"_([A-Z0-9]{4,12})\.jpg$", re.IGNORECASE)

os.makedirs(CROPS_DIR, exist_ok=True)


def guess_from_filename(name: str) -> str:
    match = GUESS_SUFFIX_RE.search(name)
    if not match:
        return ""
    candidate = match.group(1).upper()
    # "vehicle_20260819-021545.jpg" style names have no plate suffix - the
    # matched group there is a bare timestamp, not a plausible plate guess.
    if candidate.isdigit():
        return ""
    return candidate


def iter_source_images():
    for root, _dirs, files in os.walk(RAW_DIR):
        for name in sorted(files):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(root, name), name


def run():
    detected = as_is = skipped = unreadable = 0
    draft_rows = []

    sources = list(iter_source_images())
    print(f"Found {len(sources)} source images in {RAW_DIR}")

    for i, (path, name) in enumerate(sources, start=1):
        image = cv2.imread(path)
        if image is None:
            unreadable += 1
            print(f"[{i}/{len(sources)}] SKIP unreadable file: {name}")
            continue

        stem = os.path.splitext(name)[0]
        guess = guess_from_filename(name)

        candidates = main.plate_detector.detect_many(image) if main.plate_detector.available else []
        if candidates:
            crop = candidates[0]["crop"]
            out_name = f"{stem}__det.jpg"
            cv2.imwrite(os.path.join(CROPS_DIR, out_name), crop)
            draft_rows.append((out_name, guess))
            detected += 1
            print(f"[{i}/{len(sources)}] {name} -> detected crop, guess='{guess or '?'}'")
            continue

        metrics = main.plate_crop_metrics(image)
        if metrics["usable"]:
            out_name = f"{stem}__asis.jpg"
            cv2.imwrite(os.path.join(CROPS_DIR, out_name), image)
            draft_rows.append((out_name, guess))
            as_is += 1
            print(f"[{i}/{len(sources)}] {name} -> already a usable crop, guess='{guess or '?'}'")
            continue

        skipped += 1
        print(f"[{i}/{len(sources)}] {name} -> SKIPPED (no plate detected, and not a usable crop itself)")

    with open(DRAFT_LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for out_name, guess in draft_rows:
            writer.writerow([out_name, guess])

    print("\n=== SUMMARY ===")
    print(f"total source images:     {len(sources)}")
    print(f"detected via YOLO:       {detected}")
    print(f"used as-is (pre-crop):   {as_is}")
    print(f"skipped (no plate):      {skipped}")
    print(f"unreadable files:        {unreadable}")
    print(f"\nCrops saved to:  {CROPS_DIR}")
    print(f"Draft labels tsv: {DRAFT_LABELS_PATH} ({len(draft_rows)} rows, guesses need verification)")


if __name__ == "__main__":
    run()
