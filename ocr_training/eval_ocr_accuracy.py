"""Honest held-out accuracy check for the deployed plate-recognition model.

Unlike the old ocr_eval_new_* report (which scored labels_draft.tsv -- 664 of
its 739 images turned out to already be inside train_list.txt, i.e. mostly
scoring the model against images it was fine-tuned on), this script scores
ONLY ocr_training/release/val_list.txt, the group-aware held-out split
build_ocr_release.py produces (see its group_key logic -- no two crops from
the same source photo can land on both sides of the split, so val is
genuinely unseen).

Runs images through main.plate_ocr.recognize_best(), the exact function
production calls on a plate crop, so this measures the whole real pipeline
(multi-crop-variant OCR + the INDIAN_SLOT_PATTERNS/state-code grammar
correction), not just the raw model.

Reports overall + a breakdown by source: "archive" (the diverse
multi-state/multi-source Kaggle-style dataset) vs "new" (this deployment's own
cam01/cam02/vehicle captures) -- the archive number is the closest proxy this
repo has for "how well does this generalize to a plate/camera it has never
specifically seen", i.e. the "another site" question.
"""
import csv
import json
import os
import sys

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))
import main

VAL_LIST_PATH = os.path.join(BASE_DIR, "release", "val_list.txt")
CROPS_DIR = os.path.join(BASE_DIR, "release", "crops")
REPORT_TSV = os.path.join(BASE_DIR, "ocr_eval_heldout_report.tsv")
REPORT_JSON = os.path.join(BASE_DIR, "ocr_eval_heldout_summary.json")


def classify_error(pred: str, gt: str) -> str:
    if pred == gt:
        return "correct"
    if len(pred) < len(gt):
        return "dropped_char"
    if len(pred) > len(gt):
        return "extra_char"
    return "same_length_confusion"


def source_of(out_name: str) -> str:
    if out_name.startswith("old_"):
        return "archive"
    if out_name.startswith("new_"):
        return "new"
    return "other"


def run():
    entries = []
    with open(VAL_LIST_PATH, encoding="utf-8") as f:
        for line in f:
            rel_path, label = line.rstrip("\n").split("\t")
            entries.append((rel_path, label))

    print(f"Evaluating {len(entries)} held-out images against the deployed model "
          f"({main.settings.PLATE_OCR_MODEL_DIR})")

    rows = []
    stats = {}

    for i, (rel_path, label) in enumerate(entries, start=1):
        out_name = os.path.basename(rel_path)
        source = source_of(out_name)
        image_path = os.path.join(CROPS_DIR, out_name)
        image = cv2.imread(image_path)

        gt = main.normalize_plate_text(label)
        if image is None:
            pred, confidence, variant = "", 0.0, "unreadable_file"
        else:
            pred, confidence, variant = main.plate_ocr.recognize_best(image)
            pred = main.normalize_plate_text(pred)

        category = classify_error(pred, gt)
        cer = main.levenshtein_distance(pred, gt) / max(len(gt), 1)

        bucket = stats.setdefault(source, {"total": 0, "correct": 0, "cer_sum": 0.0})
        bucket["total"] += 1
        bucket["cer_sum"] += cer
        if category == "correct":
            bucket["correct"] += 1

        rows.append({
            "filename": out_name,
            "source": source,
            "ground_truth": gt,
            "ocr": pred,
            "correct": category == "correct",
            "category": category,
            "cer": round(cer, 4),
            "confidence": round(confidence, 4),
            "variant": variant,
        })
        if i % 50 == 0 or i == len(entries):
            print(f"  [{i}/{len(entries)}]")

    with open(REPORT_TSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {"model_dir": main.settings.PLATE_OCR_MODEL_DIR, "by_source": {}}
    overall_total = overall_correct = 0
    overall_cer_sum = 0.0
    for source, bucket in stats.items():
        acc = 100.0 * bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
        mean_cer = bucket["cer_sum"] / bucket["total"] if bucket["total"] else 0.0
        summary["by_source"][source] = {
            "total": bucket["total"],
            "correct": bucket["correct"],
            "exact_match_accuracy_percent": round(acc, 2),
            "mean_character_error_rate": round(mean_cer, 4),
        }
        overall_total += bucket["total"]
        overall_correct += bucket["correct"]
        overall_cer_sum += bucket["cer_sum"]

    summary["overall"] = {
        "total": overall_total,
        "correct": overall_correct,
        "exact_match_accuracy_percent": round(100.0 * overall_correct / overall_total, 2) if overall_total else 0.0,
        "mean_character_error_rate": round(overall_cer_sum / overall_total, 4) if overall_total else 0.0,
    }
    summary["wrong_examples_first_50"] = [r for r in rows if not r["correct"]][:50]

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== HELD-OUT ACCURACY (genuinely unseen images) ===")
    for source, s in summary["by_source"].items():
        print(f"{source:10s}: {s['correct']}/{s['total']} exact match = "
              f"{s['exact_match_accuracy_percent']}%  (mean CER {s['mean_character_error_rate']})")
    o = summary["overall"]
    print(f"{'overall':10s}: {o['correct']}/{o['total']} exact match = "
          f"{o['exact_match_accuracy_percent']}%  (mean CER {o['mean_character_error_rate']})")
    print(f"\nReport: {REPORT_TSV}\nSummary: {REPORT_JSON}")


if __name__ == "__main__":
    run()
