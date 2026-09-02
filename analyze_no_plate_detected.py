"""Diagnose why the plate detector found no usable plate crop for a given
list of source images (from the vehicleplate/ batch run). For each image,
inspects the raw YOLO plate-detector output (bypassing the confidence/size
filters in PlateDetector.detect_many) and the vehicle detector's box, to
separate 'model never proposed a plate box' from 'proposed but too
low-confidence/too small/wrong aspect' from 'no vehicle detected at all'.
Does not modify main.py - only reads from its already-loaded models."""
import json
import os

import cv2

import main
from config import settings

SOURCE_DIR = os.path.abspath(os.path.join(settings.BASE_DIR, "..", "vehicleplate"))


def raw_plate_boxes(image):
    if main.plate_detector.model is None:
        return []
    results = main.plate_detector.model.predict(image, device=main.plate_detector.device, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    h, w = image.shape[:2]
    out = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        out.append({
            "confidence": float(box.conf),
            "width": x2 - x1,
            "height": y2 - y1,
            "area_ratio": ((x2 - x1) * (y2 - y1)) / max(w * h, 1),
        })
    return out


def analyze(names):
    rows = []
    for i, name in enumerate(names, start=1):
        path = os.path.join(SOURCE_DIR, name)
        image = cv2.imread(path)
        if image is None:
            continue
        h, w = image.shape[:2]

        raw_boxes = raw_plate_boxes(image)
        best_box = max(raw_boxes, key=lambda b: b["confidence"], default=None)

        vehicle_candidates = main.vehicle_color_detector.vehicle_candidates(image)
        best_vehicle = max(vehicle_candidates, key=lambda v: v["confidence"], default=None)
        vehicle_area_ratio = (best_vehicle["area"] / (w * h)) if best_vehicle else 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        if best_box is None:
            reason = "no_box_proposed"
        elif best_box["confidence"] < settings.PLATE_DETECTOR_CONFIDENCE:
            reason = "low_confidence_box"
        else:
            reason = "box_ok_but_crop_filtered"

        rows.append({
            "source": name,
            "image_w": w,
            "image_h": h,
            "sharpness": round(sharpness, 1),
            "brightness": round(brightness, 1),
            "vehicle_label": best_vehicle["label"] if best_vehicle else "none",
            "vehicle_area_ratio": round(vehicle_area_ratio, 4),
            "best_plate_box_confidence": round(best_box["confidence"], 3) if best_box else 0.0,
            "best_plate_box_area_ratio": round(best_box["area_ratio"], 5) if best_box else 0.0,
            "num_raw_boxes": len(raw_boxes),
            "reason": reason,
        })
        if i % 50 == 0:
            print(f"...{i}/{len(names)}")
    return rows


if __name__ == "__main__":
    results_json = os.path.join(os.path.dirname(__file__), "test_outputs", "vehicleplate_batch_results.json")
    data = json.load(open(results_json, encoding="utf-8"))
    failed_names = [d["source"] for d in data if not d["plate_crop_saved_to"] and not d["plate_text_valid"]]
    print(f"Analyzing {len(failed_names)} no-plate-detected images...")

    rows = analyze(failed_names)

    out_path = os.path.join(os.path.dirname(__file__), "test_outputs", "no_plate_detected_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

    from collections import Counter
    print("\n=== reason breakdown ===")
    for reason, count in Counter(r["reason"] for r in rows).most_common():
        print(f"{reason}: {count}")

    print("\n=== vehicle label breakdown ===")
    for label, count in Counter(r["vehicle_label"] for r in rows).most_common():
        print(f"{label}: {count}")

    small_vehicle = [r for r in rows if r["vehicle_area_ratio"] and r["vehicle_area_ratio"] < 0.05]
    no_vehicle = [r for r in rows if r["vehicle_label"] == "none"]
    dark = [r for r in rows if r["brightness"] < 60]
    blurry = [r for r in rows if r["sharpness"] < 50]
    print(f"\nvehicle area_ratio < 0.05 (small/far in frame): {len(small_vehicle)}")
    print(f"no vehicle detected at all in frame: {len(no_vehicle)}")
    print(f"dark images (brightness < 60): {len(dark)}")
    print(f"blurry images (sharpness < 50): {len(blurry)}")
