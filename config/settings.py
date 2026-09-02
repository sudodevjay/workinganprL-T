import os

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_yaml(filename):
    with open(os.path.join(BASE_DIR, "config", filename)) as f:
        return yaml.safe_load(f)


def _abs_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


_settings = _load_yaml("settings.yaml")

SERVICE_HOST = _settings["service"]["host"]
SERVICE_PORT = _settings["service"]["port"]

_runtime = _settings.get("runtime", {})
RUNTIME_OPENCV_THREADS = int(_runtime.get("opencv_threads", 1))
RUNTIME_PADDLE_CPU_THREADS = int(_runtime.get("paddle_cpu_threads", 2))
RUNTIME_TORCH_THREADS = int(_runtime.get("torch_threads", 2))
RUNTIME_TORCH_INTEROP_THREADS = int(_runtime.get("torch_interop_threads", 1))
RUNTIME_MAX_CONCURRENT_INFERENCE = int(_runtime.get("max_concurrent_inference", 3))

VEHICLE_DETECTOR_MODEL_PATH = _abs_path(_settings["vehicle_color"]["detector_model_path"])
VEHICLE_SEGMENTATION_MODEL_PATH = _abs_path(_settings["vehicle_color"]["segmentation_model_path"])
TYPE_CLASSIFIER_MODEL_PATH = _abs_path(_settings["vehicle_color"]["type_classifier_model_path"])
VEHICLE_COLOR_CONFIDENCE = _settings["vehicle_color"]["confidence"]
VEHICLE_COLOR_DEVICE = _settings["vehicle_color"]["device"]

_vehicle_trigger = _settings.get("vehicle_trigger", {})
VEHICLE_TRIGGER_ENABLED = bool(_vehicle_trigger.get("enabled", True))
VEHICLE_TRIGGER_CONFIDENCE = float(_vehicle_trigger.get("confidence", 0.25))
VEHICLE_TRIGGER_MIN_AREA_RATIO = float(_vehicle_trigger.get("min_area_ratio", 0.01))
VEHICLE_TRIGGER_HOLD_SECONDS = float(_vehicle_trigger.get("hold_seconds", 1.5))
VEHICLE_TRIGGER_NO_VEHICLE_LOG_INTERVAL_SECONDS = float(
    _vehicle_trigger.get("no_vehicle_log_interval_seconds", 10.0)
)

_vehicle_trigger_motion = _vehicle_trigger.get("motion_prefilter", {})
VEHICLE_TRIGGER_MOTION_PREFILTER_ENABLED = bool(_vehicle_trigger_motion.get("enabled", True))
VEHICLE_TRIGGER_MOTION_PREFILTER_WIDTH = int(_vehicle_trigger_motion.get("downscale_width", 320))
VEHICLE_TRIGGER_MOTION_PREFILTER_MIN_CHANGE_RATIO = float(
    _vehicle_trigger_motion.get("min_change_ratio", 0.015)
)
VEHICLE_TRIGGER_MOTION_PREFILTER_PIXEL_THRESHOLD = int(
    _vehicle_trigger_motion.get("pixel_diff_threshold", 25)
)

PLATE_DETECTOR_ENABLED = _settings["plate_detector"]["enabled"]
PLATE_DETECTOR_MODEL_PATH = _abs_path(_settings["plate_detector"]["model_path"])
PLATE_DETECTOR_CONFIDENCE = _settings["plate_detector"]["confidence"]
PLATE_DETECTOR_PADDING_RATIO = _settings["plate_detector"]["padding_ratio"]
PLATE_DETECTOR_CLASSES = _settings["plate_detector"].get("classes", [])

PLATE_COLOR_ENABLED = _settings["plate_color"]["enabled"]

PLATE_OCR_ENABLED = _settings["plate_ocr"]["enabled"]
PLATE_OCR_MODEL_DIR = _abs_path(_settings["plate_ocr"]["model_dir"])
PLATE_OCR_CHAR_DICT_PATH = _abs_path(_settings["plate_ocr"]["char_dict_path"])
PLATE_OCR_MIN_CONFIDENCE = _settings["plate_ocr"]["min_confidence"]
PLATE_OCR_POST_UNCERTAIN = bool(_settings["plate_ocr"].get("post_uncertain", False))
PLATE_OCR_POST_UNREADABLE = bool(_settings["plate_ocr"].get("post_unreadable", False))
PLATE_OCR_FALLBACK_MIN_CONFIDENCE = float(_settings["plate_ocr"].get("fallback_min_confidence", 0.0))
_plate_ocr_sar_refine = _settings["plate_ocr"].get("sar_refine", {})
PLATE_OCR_SAR_REFINE_ENABLED = bool(_plate_ocr_sar_refine.get("enabled", False))
PLATE_OCR_SAR_REFINE_CHECKPOINT_PATH = _abs_path(
    _plate_ocr_sar_refine.get("checkpoint_path", "weights/indian_plate_rec_v2_sar_best_accuracy.pdparams")
)
_plate_ocr_aggregation = _settings["plate_ocr"].get("aggregation", {})
PLATE_OCR_AGGREGATION_ENABLED = _plate_ocr_aggregation.get("enabled", True)
PLATE_OCR_AGGREGATION_WINDOW_SIZE = int(_plate_ocr_aggregation.get("window_size", 10))
PLATE_OCR_AGGREGATION_MIN_CANDIDATES = int(_plate_ocr_aggregation.get("min_candidates", 7))
PLATE_OCR_AGGREGATION_FRAME_DELAY_SECONDS = float(_plate_ocr_aggregation.get("frame_delay_seconds", 0.08))
PLATE_TRACK_MAX_FRAMES = int(_plate_ocr_aggregation.get("track_max_frames", 40))

_storage = _settings.get("storage", {})
STORAGE_ENABLED = bool(_storage.get("enabled", True))
PLATE_IMAGE_DIR = _abs_path(_storage.get("plate_image_dir", "captured_plates"))
VEHICLE_IMAGE_DIR = _abs_path(_storage.get("vehicle_image_dir", "captured_vehicles"))

BACKEND_EVENT_URL = os.getenv("BACKEND_EVENT_URL", _settings["backend"]["event_url"])
BACKEND_EVENT_TIMEOUT = _settings["backend"]["event_timeout"]
BACKEND_EVENT_TOKEN = os.getenv("AI_EVENT_TOKEN", _settings["backend"].get("event_token", ""))

CAMERA_AUTO_START_WORKERS = _settings["camera"].get("auto_start_workers", True)
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME", _settings["camera"].get("username", ""))
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", _settings["camera"].get("password", ""))
CAMERA_STREAM_PATH = _settings["camera"]["stream_path"]
CAMERA_SNAPSHOT_PATH = _settings["camera"]["snapshot_path"]
CAMERA_SNAPSHOT_RESOLUTION = _settings["camera"].get("snapshot_resolution", "")
CAMERA_SNAPSHOT_COMPRESSION = _settings["camera"].get("snapshot_compression")
CAMERA_SNAPSHOT_INTERVAL_SECONDS = _settings["camera"]["snapshot_interval_seconds"]
CAMERA_SNAPSHOT_TIMEOUT_SECONDS = _settings["camera"]["snapshot_timeout_seconds"]
CAMERA_DUPLICATE_SECONDS = _settings["camera"]["duplicate_seconds"]

try:
    CAMERAS = _load_yaml("cameras.yaml").get("cameras", [])
except FileNotFoundError:
    CAMERAS = []
