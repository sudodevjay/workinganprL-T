"""Clean, uncontaminated accuracy test for the newly trained OCR model.

Uses anpr-ai-service/vehicleplate/ (116 images) - verified to have ZERO
filename overlap with anything in ocr_training/ (the training data), unlike
indiannumberplate/New folder/New folder/ which turned out to have 177/182
of its remaining images already baked into the training crops. This is a
genuinely held-out test set, not seen during training.

Read-only - does not write snapshots anywhere, just reports accuracy."""
import csv
import json
import os

import cv2

import main

SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicleplate")


def run():
    image_names = sorted(
        name for name in os.listdir(SOURCE_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Found {len(image_names)} images in {SOURCE_DIR}")

    read_ok = read_invalid = unreadable_files = 0
    results = []

    for i, name in enumerate(image_names, start=1):
        path = os.path.join(SOURCE_DIR, name)
        image = cv2.imread(path)
        if image is None:
            unreadable_files += 1
            print(f"[{i}/{len(image_names)}] SKIP unreadable image file: {name}")
            continue

        result = main.process_frame(overview_image=image, include_plate_crop=False)
        plate_text = result.get("plate_text", "")
        valid = bool(result.get("plate_text_valid"))
        confidence = float(result.get("plate_confidence") or 0.0)

        if valid:
            read_ok += 1
        else:
            read_invalid += 1

        results.append({
            "source": name,
            "plate_text": plate_text,
            "plate_text_valid": valid,
            "plate_confidence": round(confidence, 4),
            "vehicle_type": result.get("vehicle_type"),
            "vehicle_color": result.get("vehicle_color"),
            "plate_color": result.get("plate_color"),
        })
        print(f"[{i}/{len(image_names)}] {name} -> plate='{plate_text}' valid={valid} conf={confidence:.3f}")

    print("\n=== SUMMARY (clean, unseen test set) ===")
    print(f"total images:            {len(image_names)}")
    print(f"valid plate read:        {read_ok}")
    print(f"unreadable/invalid text: {read_invalid}")
    print(f"unreadable image files:  {unreadable_files}")

    return results


if __name__ == "__main__":
    all_results = run()
    out_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vehicleplate_clean_test_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "vehicleplate_clean_test_results.csv"), "w", newline="", encoding="utf-8") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
