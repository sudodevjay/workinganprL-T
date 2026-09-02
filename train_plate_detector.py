import argparse
import os
import shutil
from pathlib import Path

import yaml

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "runs" / "ultralytics"))

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DEFAULT_DATA = REPO_DIR / "vehicle-plate-color.v1i.yolo26" / "data.yaml"
DEFAULT_BASE_MODEL = BASE_DIR / "weights" / "yolo11n.pt"
DEFAULT_OUTPUT = BASE_DIR / "weights" / "plate_detector.pt"


def _resolve_split_path(dataset_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)

    candidates = [dataset_root / path]
    parts = path.parts
    while parts and parts[0] == "..":
        parts = parts[1:]
        candidates.append(dataset_root / Path(*parts))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0].resolve())


def _prepare_data_yaml(data_path: Path) -> Path:
    data_path = data_path.resolve()
    dataset_root = data_path.parent
    with open(data_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for split in ("train", "val", "test"):
        if data.get(split):
            data[split] = _resolve_split_path(dataset_root, data[split])

    output = BASE_DIR / "training_runs" / "plate_detector_data.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return output


def main():
    parser = argparse.ArgumentParser(description="Train YOLO plate detector and install the best weight.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="YOLO data.yaml path")
    parser.add_argument("--model", default=str(DEFAULT_BASE_MODEL), help="Base YOLO model path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to copy best.pt")
    args = parser.parse_args()

    data_yaml = _prepare_data_yaml(Path(args.data))
    model = YOLO(args.model)
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(BASE_DIR / "training_runs"),
        name="plate_detector",
        exist_ok=True,
    )

    best_weight = Path(result.save_dir) / "weights" / "best.pt"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, output)
    print(f"Plate detector saved to: {output}")


if __name__ == "__main__":
    main()
