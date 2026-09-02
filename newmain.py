"""Standalone vehicle-arrival image collector.

main.py runs the full ANPR pipeline (plate detection + OCR + backend
posting) and only keeps the number-plate crop as its saved snapshot. This
script is separate from that pipeline: it just watches each enabled camera
from config/cameras.yaml and, as soon as a vehicle shows up, saves the full
camera frame (the whole vehicle, not a plate crop) to disk under
captured_vehicles/<camera_name>/.

Run:
    python newmain.py

Stop with Ctrl+C.
"""

import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from ultralytics import YOLO

from config import settings

try:
    cv2.setNumThreads(settings.RUNTIME_OPENCV_THREADS)
except Exception:
    pass


CAPTURE_OUTPUT_DIR = os.path.join(settings.BASE_DIR, "captured_vehicles")
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


class CameraUnavailable(Exception):
    pass


def decode_image(image_bytes):
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def enabled_cameras():
    for camera in settings.CAMERAS:
        if camera.get("enabled", False) and camera.get("process_enabled", True):
            yield camera


def camera_source_type(camera: dict) -> str:
    if camera.get("source_type"):
        return str(camera["source_type"]).strip().lower()
    if camera.get("video_path"):
        return "video"
    return "ip"


def is_video_source(camera: dict) -> bool:
    return camera_source_type(camera) in {"video", "file", "local_video"}


def resolve_local_path(path: str) -> str:
    raw_path = str(path or "").strip()
    if raw_path.startswith("file://"):
        raw_path = raw_path[7:]
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(settings.BASE_DIR, raw_path))


def video_path(camera: dict) -> str:
    return resolve_local_path(camera.get("video_path") or camera.get("ip", ""))


def digest_auth():
    if not settings.CAMERA_USERNAME and not settings.CAMERA_PASSWORD:
        return None
    return HTTPDigestAuth(settings.CAMERA_USERNAME, settings.CAMERA_PASSWORD)


def camera_url(camera: dict, path: str) -> str:
    base = camera["ip"].strip()
    if base.startswith("http://") or base.startswith("https://"):
        return f"{base.rstrip('/')}/{path.lstrip('/')}"
    return f"http://{base}/{path.lstrip('/')}"


def snapshot_params(camera: dict) -> dict:
    params = {}
    resolution = camera.get("snapshot_resolution", settings.CAMERA_SNAPSHOT_RESOLUTION)
    if resolution:
        params["resolution"] = resolution
    compression = camera.get("snapshot_compression", settings.CAMERA_SNAPSHOT_COMPRESSION)
    if compression is not None:
        params["compression"] = compression
    return params


def open_video_capture(camera: dict):
    path = video_path(camera)
    if not os.path.exists(path):
        raise CameraUnavailable(f"Video file not found: {path}")

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise CameraUnavailable(f"Could not open video file: {path}")
    return capture


def read_video_frame(capture, loop: bool):
    ok, frame = capture.read()
    if ok:
        return frame
    if not loop:
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = capture.read()
    return frame if ok else None


def get_ip_snapshot(camera: dict) -> bytes | None:
    try:
        response = requests.get(
            camera_url(camera, settings.CAMERA_SNAPSHOT_PATH),
            params=snapshot_params(camera),
            auth=digest_auth(),
            timeout=settings.CAMERA_SNAPSHOT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CameraUnavailable(str(exc)) from exc
    return response.content


def motion_prefilter_signature(frame):
    """Small grayscale copy of a frame, cheap to diff against the next one."""
    width = settings.VEHICLE_TRIGGER_MOTION_PREFILTER_WIDTH
    height = max(1, int(frame.shape[0] * (width / frame.shape[1])))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def motion_prefilter_changed(prev_signature, curr_signature) -> bool:
    if prev_signature is None or prev_signature.shape != curr_signature.shape:
        return True
    diff = cv2.absdiff(prev_signature, curr_signature)
    changed_pixels = np.count_nonzero(diff > settings.VEHICLE_TRIGGER_MOTION_PREFILTER_PIXEL_THRESHOLD)
    changed_ratio = changed_pixels / diff.size
    return changed_ratio >= settings.VEHICLE_TRIGGER_MOTION_PREFILTER_MIN_CHANGE_RATIO


class VehicleDetector:
    """Thin wrapper around just the vehicle-presence YOLO model -- no color
    or type classifiers loaded, since this script only needs to know
    "is a vehicle in frame", not what it looks like."""

    def __init__(self, model_path, device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame, confidence: float, min_area_ratio: float) -> list[dict]:
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(frame, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        frame_area = max(frame.shape[0] * frame.shape[1], 1)
        candidates = []
        for box in boxes:
            label = self.model.names[int(box.cls)]
            box_confidence = float(box.conf)
            if label not in VEHICLE_CLASSES or box_confidence < confidence:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = max(x2 - x1, 1) * max(y2 - y1, 1)
            if area / frame_area < min_area_ratio:
                continue
            candidates.append({"bbox": [x1, y1, x2, y2], "label": label, "confidence": box_confidence})
        return candidates


vehicle_detector = VehicleDetector(
    model_path=settings.VEHICLE_DETECTOR_MODEL_PATH,
    device=settings.VEHICLE_COLOR_DEVICE,
)


class VehicleArrivalDetector:
    """Per-camera state machine that fires once per vehicle visit (on the
    no-vehicle -> vehicle transition), not once per frame -- so a vehicle
    sitting in frame for a few seconds produces one saved image, not
    dozens."""

    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.vehicle_present = False
        self.last_vehicle_seen = 0.0
        self.last_motion_signature = None

    def check(self, frame):
        """Returns the winning vehicle candidate dict on a new arrival,
        otherwise None."""
        now = time.time()

        if settings.VEHICLE_TRIGGER_MOTION_PREFILTER_ENABLED:
            signature = motion_prefilter_signature(frame)
            moved = motion_prefilter_changed(self.last_motion_signature, signature)
            self.last_motion_signature = signature
            if not moved and not self.vehicle_present:
                return None

        candidates = vehicle_detector.detect(
            frame,
            confidence=settings.VEHICLE_TRIGGER_CONFIDENCE,
            min_area_ratio=settings.VEHICLE_TRIGGER_MIN_AREA_RATIO,
        )

        if candidates:
            self.last_vehicle_seen = now
            is_new_arrival = not self.vehicle_present
            self.vehicle_present = True
            if is_new_arrival:
                return max(candidates, key=lambda item: item["confidence"])
            return None

        if self.vehicle_present and now - self.last_vehicle_seen > settings.VEHICLE_TRIGGER_HOLD_SECONDS:
            self.vehicle_present = False
        return None


def save_full_frame(camera_name: str, frame) -> str | None:
    now = datetime.now()
    date_dir = os.path.join(CAPTURE_OUTPUT_DIR, camera_name, now.strftime("%Y-%m-%d"))
    os.makedirs(date_dir, exist_ok=True)

    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    file_path = os.path.join(date_dir, f"{camera_name}_{timestamp}.jpg")
    if not cv2.imwrite(file_path, frame):
        return None
    return file_path


def run_ip_camera(camera: dict, arrival_detector: VehicleArrivalDetector, stop_event: threading.Event):
    camera_name = camera["name"]
    interval = settings.CAMERA_SNAPSHOT_INTERVAL_SECONDS
    print(f"[{camera_name}] started (ip) ip={camera['ip']}")

    while not stop_event.is_set():
        try:
            snapshot = get_ip_snapshot(camera)
            frame = decode_image(snapshot) if snapshot else None
            if frame is not None and frame.size > 0:
                vehicle = arrival_detector.check(frame)
                if vehicle is not None:
                    saved_path = save_full_frame(camera_name, frame)
                    print(
                        f"[{camera_name}] vehicle arrived ({vehicle['label']} "
                        f"confidence={vehicle['confidence']:.3f}) -> saved {saved_path}"
                    )
        except CameraUnavailable as exc:
            print(f"[{camera_name}] camera unavailable: {exc}")
        except requests.RequestException as exc:
            print(f"[{camera_name}] HTTP error: {exc}")
        except Exception as exc:
            print(f"[{camera_name}] worker error: {exc}")
        finally:
            stop_event.wait(interval)


def run_video_camera(camera: dict, arrival_detector: VehicleArrivalDetector, stop_event: threading.Event):
    camera_name = camera["name"]
    path = video_path(camera)

    try:
        capture = open_video_capture(camera)
    except CameraUnavailable as exc:
        print(f"[{camera_name}] video unavailable: {exc}")
        return

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_delay = 1.0 / max(source_fps, 1.0)
    realtime = bool(camera.get("video_realtime", False))
    loop = bool(camera.get("video_loop", True))
    process_every = int(
        camera.get("process_every_n_frames")
        or max(1, round(source_fps * settings.CAMERA_SNAPSHOT_INTERVAL_SECONDS))
    )
    frame_index = 0

    print(f"[{camera_name}] started (video) video={path}")
    try:
        while not stop_event.is_set():
            iteration_started = time.time()
            frame = read_video_frame(capture, loop=loop)
            if frame is None:
                print(f"[{camera_name}] video playback completed")
                break

            if frame_index % process_every == 0:
                try:
                    vehicle = arrival_detector.check(frame)
                    if vehicle is not None:
                        saved_path = save_full_frame(camera_name, frame)
                        print(
                            f"[{camera_name}] vehicle arrived ({vehicle['label']} "
                            f"confidence={vehicle['confidence']:.3f}) -> saved {saved_path}"
                        )
                except Exception as exc:
                    print(f"[{camera_name}] worker error: {exc}")

            frame_index += 1
            if realtime:
                elapsed = time.time() - iteration_started
                remaining = frame_delay - elapsed
                if remaining > 0:
                    stop_event.wait(remaining)
    finally:
        capture.release()


def camera_worker(camera: dict, stop_event: threading.Event):
    arrival_detector = VehicleArrivalDetector(camera["name"])
    if is_video_source(camera):
        run_video_camera(camera, arrival_detector, stop_event)
    else:
        run_ip_camera(camera, arrival_detector, stop_event)


def main():
    cameras = list(enabled_cameras())
    if not cameras:
        print("No enabled cameras found in config/cameras.yaml.")
        return

    os.makedirs(CAPTURE_OUTPUT_DIR, exist_ok=True)
    print(f"Saving vehicle images to: {CAPTURE_OUTPUT_DIR}")

    stop_event = threading.Event()
    threads = []
    for camera in cameras:
        thread = threading.Thread(
            target=camera_worker,
            args=(camera, stop_event),
            daemon=True,
            name=f"vehicle-capture-{camera['name']}",
        )
        thread.start()
        threads.append(thread)

    print(f"Running {len(threads)} camera worker(s). Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
        for thread in threads:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
