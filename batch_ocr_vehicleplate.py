"""Batch-run the production ANPR pipeline (plate detect + OCR + vehicle type +
vehicle color + plate color) over every image in vehicleplate/, using
main.process_frame() exactly as anpr-ai-service already does (no changes to
main.py). For each image, the detected/cropped number-plate image itself
(process_frame's plate_crop_image) is saved into anpr-multicam's snapshot
folder, using the same day/plates naming convention the live service uses
(anpr-multicam/routes/anpr_routes.py:_save_snapshot). All extracted details
(plate text, confidence, vehicle type/color, plate color) are written to a
CSV + JSON report so they're easy to inspect."""
import csv
import json
import os
import re
from datetime import datetime

import cv2

import main
from config import settings

SOURCE_DIR = os.path.abspath(os.path.join(settings.BASE_DIR, "..", "vehicleplate"))
SNAPSHOTS_DIR = os.path.abspath(os.path.join(settings.BASE_DIR, "..", "anpr-multicam", "logs", "snapshots"))
CAMERA_NAME = "vehicleplate_batch"
_SAFE_NAME_PART = re.compile(r"[^A-Za-z0-9_-]+")


def save_plate_crop(plate_crop_image, plate_text: str) -> str:
    encoded = main.encode_jpeg(plate_crop_image)
    if encoded is None:
        return ""
    now = datetime.now()
    day_plates_dir = os.path.join(SNAPSHOTS_DIR, now.strftime("%Y-%m-%d"), "plates")
    os.makedirs(day_plates_dir, exist_ok=True)
    safe_plate = _SAFE_NAME_PART.sub("_", plate_text or "UNREADABLE")
    fname = f"{CAMERA_NAME}_{now.strftime('%H-%M-%S-%f')}_{safe_plate}.jpg"
    path = os.path.join(day_plates_dir, fname)
    with open(path, "wb") as f:
        f.write(encoded)
    return path


def run():
    image_names = sorted(
        name for name in os.listdir(SOURCE_DIR)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Found {len(image_names)} images in {SOURCE_DIR}")

    read_ok = read_invalid = no_plate_crop = unreadable_files = 0
    results = []

    for i, name in enumerate(image_names, start=1):
        path = os.path.join(SOURCE_DIR, name)
        image = cv2.imread(path)
        if image is None:
            unreadable_files += 1
            print(f"[{i}/{len(image_names)}] SKIP unreadable image file: {name}")
            continue

        result = main.process_frame(overview_image=image, include_plate_crop=True)
        plate_text = result.get("plate_text", "")
        valid = bool(result.get("plate_text_valid"))
        confidence = float(result.get("plate_confidence") or 0.0)
        plate_crop_image = result.get("plate_crop_image")

        if valid:
            read_ok += 1
        else:
            read_invalid += 1

        saved_path = ""
        if plate_crop_image is not None:
            saved_path = save_plate_crop(plate_crop_image, plate_text if valid else "")
        else:
            no_plate_crop += 1

        results.append({
            "source": name,
            "plate_text": plate_text,
            "plate_text_valid": valid,
            "plate_confidence": round(confidence, 4),
            "vehicle_type": result.get("vehicle_type"),
            "vehicle_color": result.get("vehicle_color"),
            "plate_color": result.get("plate_color"),
            "plate_state_code": result.get("plate_state_code"),
            "plate_state_name": result.get("plate_state_name"),
            "plate_crop_saved_to": saved_path,
        })
        print(f"[{i}/{len(image_names)}] {name} -> plate='{plate_text}' valid={valid} conf={confidence:.3f} "
              f"type={result.get('vehicle_type')} color={result.get('vehicle_color')} "
              f"plate_color={result.get('plate_color')}")

    print("\n=== SUMMARY ===")
    print(f"total images:            {len(image_names)}")
    print(f"valid plate read:        {read_ok}")
    print(f"unreadable/invalid text: {read_invalid}")
    print(f"no plate crop detected:  {no_plate_crop}")
    print(f"unreadable image files:  {unreadable_files}")

    return results


if __name__ == "__main__":
    all_results = run()
    out_dir = os.path.join(os.path.dirname(__file__), "test_outputs")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "vehicleplate_batch_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(out_dir, "vehicleplate_batch_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\nDetails saved to {json_path}\nand {csv_path}")
