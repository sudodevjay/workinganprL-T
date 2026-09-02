"""Merges the old archive/ dataset (1798 manually-labeled plate crops), the
new ocr_training/ dataset (739 crops, filename-guesses verified/corrected by
hand + AI transcription + dual-camera cross-checking), and synthetic
long/short/two-line plate crops (ocr_training/synthetic/, see
generate_synthetic_plates.py) into a single release/ folder ready to zip and
upload to Colab for PaddleOCR rec fine-tuning.

Uncertain entries (trailing '?', 'NOT_PLATE', 'FOREIGN') are excluded from
training by default, same convention archive/prepare_paddleocr_data.py used -
training on a label we weren't sure about risks teaching the model the wrong
thing. Re-run after correcting labels to pick up fixes.

Known-bad or ambiguous real labels can be listed in
ocr_training/exclude_training.tsv and are skipped from the release. Real crops
that look like stacked bike/truck plates get a few extra train-only copies
because those are scarce in the real data. Synthetic crops are train-only for
the same reason - val should only ever score against real photographs.
"""
import csv
import os
import random
import shutil

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "archive"))
NEW_CROPS_DIR = os.path.join(BASE_DIR, "ocr_training", "crops")
NEW_LABELS_PATH = os.path.join(BASE_DIR, "ocr_training", "labels_draft.tsv")
EXCLUSIONS_PATH = os.path.join(BASE_DIR, "ocr_training", "exclude_training.tsv")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "ocr_training", "synthetic")

RELEASE_DIR = os.path.join(BASE_DIR, "ocr_training", "release")
RELEASE_CROPS_DIR = os.path.join(RELEASE_DIR, "crops")
VAL_FRACTION = 0.1
SEED = 42
INCLUDE_UNCERTAIN = False
RARE_LETTERS = set("QSZGYXV")
RARE_LETTER_EXTRA_COPIES = 2
DOUBLE_LINE_MIN_ASPECT_RATIO = 0.6
# Kept in sync with main.py's is_double_line threshold -- widening this to
# 2.2 was tried and reverted there (measured net worse on the held-out eval).
DOUBLE_LINE_MAX_ASPECT_RATIO = 1.8
# Bumped from 3 -- ocr_training/eval_ocr_accuracy.py's held-out run still
# shows two-line (bike/truck) plates misread more often than single-line
# ones, including a boundary-duplicate failure mode (main.py's
# recognize_double_line join, see its overlap comment) that only enough real
# two-line examples can teach the model to resist.
DOUBLE_LINE_EXTRA_COPIES = 5
# ocr_training/eval_ocr_accuracy.py's held-out run found dropped-character
# misreads (a genuinely doubled digit/letter collapsing into one, e.g.
# "3722" -> "372") in plates whose label has an adjacent repeated character --
# this is CTC's known repeated-symbol collapse failure mode (see PlateOCR's
# _decode_ctc in main.py). Duplicating these in training gives the model more
# examples of exactly the sequences it tends to collapse.
REPEATED_CHAR_EXTRA_COPIES = 2

os.makedirs(RELEASE_CROPS_DIR, exist_ok=True)


def is_usable(label: str) -> bool:
    if not label or label == "?" or label == "NOT_PLATE":
        return False
    if "FOREIGN" in label:
        return False
    if label.endswith("?") and not INCLUDE_UNCERTAIN:
        return False
    return True


def load_exclusions():
    exclusions = set()
    if not os.path.exists(EXCLUSIONS_PATH):
        return exclusions
    with open(EXCLUSIONS_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            source = (row.get("source") or "").strip()
            filename = (row.get("filename") or "").strip()
            if source and filename:
                exclusions.add((source, filename))
    return exclusions


def load_archive_entries(exclusions):
    entries = []
    labels_path = os.path.join(ARCHIVE_DIR, "labels_draft.tsv")
    with open(labels_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fname, raw_label = line.split("\t")
            if ("archive", fname) in exclusions:
                continue
            if not is_usable(raw_label):
                continue
            label = raw_label.rstrip("?") if INCLUDE_UNCERTAIN else raw_label
            src = os.path.join(ARCHIVE_DIR, "crops", fname)
            if not os.path.exists(src):
                continue
            out_name = f"old_{fname}"
            # Archive filenames are "<source_photo_id>_<box_index>.jpg" -- several
            # crops can come from the same source photo (multiple plates in one
            # shot). Group by the photo id so a train/val split never puts two
            # crops of the same photo (same lighting/background/camera) on both
            # sides, which would leak that photo's visual conditions into val.
            group_key = ("archive", fname.rsplit("_", 1)[0])
            entries.append((src, out_name, label, group_key))
    return entries


def load_new_entries(exclusions):
    entries = []
    with open(NEW_LABELS_PATH, encoding="utf-8") as f:
        for fname, raw_label in csv.reader(f, delimiter="\t"):
            if ("new", fname) in exclusions:
                continue
            if not is_usable(raw_label):
                continue
            label = raw_label.rstrip("?") if INCLUDE_UNCERTAIN else raw_label
            src = os.path.join(NEW_CROPS_DIR, fname)
            if not os.path.exists(src):
                continue
            out_name = f"new_{fname}"
            # Each row here is already one crop per detection event (no shared
            # source photo across rows), so the row itself is its own group.
            group_key = ("new", fname)
            entries.append((src, out_name, label, group_key))
    return entries


def load_synthetic_entries():
    entries = []
    if not os.path.isdir(SYNTHETIC_DIR):
        return entries
    for fname in os.listdir(SYNTHETIC_DIR):
        if not fname.lower().endswith(".jpg"):
            continue
        label = fname.rsplit("_", 1)[-1].rsplit(".", 1)[0]
        src = os.path.join(SYNTHETIC_DIR, fname)
        out_name = fname
        entries.append((src, out_name, label))
    return entries


def clear_release_crops():
    for name in os.listdir(RELEASE_CROPS_DIR):
        path = os.path.join(RELEASE_CROPS_DIR, name)
        if os.path.isfile(path):
            os.remove(path)


def has_adjacent_repeat(label: str) -> bool:
    return any(a == b for a, b in zip(label, label[1:]))


def is_double_line_crop(src: str) -> bool:
    image = cv2.imread(src)
    if image is None or image.size == 0:
        return False
    height, width = image.shape[:2]
    aspect_ratio = width / float(max(height, 1))
    return DOUBLE_LINE_MIN_ASPECT_RATIO <= aspect_ratio <= DOUBLE_LINE_MAX_ASPECT_RATIO


def run():
    clear_release_crops()

    exclusions = load_exclusions()
    archive_entries = load_archive_entries(exclusions)
    new_entries = load_new_entries(exclusions)
    synthetic_entries = load_synthetic_entries()
    print(f"manual exclusions:        {len(exclusions)}")
    print(f"archive usable entries:   {len(archive_entries)}")
    print(f"new usable entries:       {len(new_entries)}")
    print(f"synthetic entries:        {len(synthetic_entries)}")
    print(f"synthetic two-line crops: {sum(1 for _src, out_name, _label in synthetic_entries if out_name.startswith('synth_two_line_'))}")

    real_entries = archive_entries + new_entries
    real_double_line_total = sum(1 for src, _out_name, _label, _group in real_entries if is_double_line_crop(src))
    for src, out_name, _label, _group in real_entries:
        shutil.copyfile(src, os.path.join(RELEASE_CROPS_DIR, out_name))
    for src, out_name, _label in synthetic_entries:
        shutil.copyfile(src, os.path.join(RELEASE_CROPS_DIR, out_name))

    # Group-aware split: shuffle whole source-photo groups (not individual
    # crops) between train/val. Splitting per-crop let two crops from the same
    # archive photo (same lighting/background/camera) land on opposite sides,
    # which leaked that photo's visual conditions into "held-out" val instead
    # of testing on genuinely unseen conditions -- see load_archive_entries.
    groups = {}
    for entry in real_entries:
        groups.setdefault(entry[3], []).append(entry)
    group_keys = list(groups.keys())
    random.seed(SEED)
    random.shuffle(group_keys)

    val_target = int(len(real_entries) * VAL_FRACTION)
    val_entries, train_entries = [], []
    val_count = 0
    for key in group_keys:
        group_entries = groups[key]
        if val_count < val_target:
            val_entries.extend(group_entries)
            val_count += len(group_entries)
        else:
            train_entries.extend(group_entries)

    train_double_line_base = [
        entry for entry in train_entries
        if is_double_line_crop(entry[0])
    ]
    rare_extra = [
        entry for entry in train_entries
        if any(ch in RARE_LETTERS for ch in entry[2])
    ] * RARE_LETTER_EXTRA_COPIES
    double_line_extra = train_double_line_base * DOUBLE_LINE_EXTRA_COPIES
    repeated_char_extra = [
        entry for entry in train_entries
        if has_adjacent_repeat(entry[2])
    ] * REPEATED_CHAR_EXTRA_COPIES

    train_entries = [
        (src, out_name, label)
        for src, out_name, label, _group in train_entries + rare_extra + double_line_extra + repeated_char_extra
    ] + synthetic_entries
    random.shuffle(train_entries)

    train_path = os.path.join(RELEASE_DIR, "train_list.txt")
    val_path = os.path.join(RELEASE_DIR, "val_list.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        for _src, out_name, label in train_entries:
            f.write(f"crops/{out_name}\t{label}\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for _src, out_name, label, _group in val_entries:
            f.write(f"crops/{out_name}\t{label}\n")

    shutil.copyfile(os.path.join(ARCHIVE_DIR, "en_dict.txt"), os.path.join(RELEASE_DIR, "en_dict.txt"))

    print(f"\nReal usable total:  {len(real_entries)}")
    print(f"Real double-line-like total: {real_double_line_total}")
    print(f"Rare-letter duplicate copies added: {len(rare_extra)}")
    print(f"Double-line duplicate copies added: {len(double_line_extra)}")
    print(f"Repeated-char duplicate copies added: {len(repeated_char_extra)}")
    print(f"Train (real+dupes+synthetic): {len(train_entries)} -> {train_path}")
    print(f"Val (real only):              {len(val_entries)} -> {val_path}")
    print(f"Crops copied to: {RELEASE_CROPS_DIR}")


if __name__ == "__main__":
    run()
