import difflib
import math
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(__file__), "runs", "ultralytics"))

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from paddle import inference
from requests.auth import HTTPDigestAuth

from ultralytics import YOLO

from config import settings

try:
    cv2.setNumThreads(settings.RUNTIME_OPENCV_THREADS)
except Exception:
    pass

try:
    import torch

    torch.set_num_threads(settings.RUNTIME_TORCH_THREADS)
    torch.set_num_interop_threads(settings.RUNTIME_TORCH_INTEROP_THREADS)
except Exception:
    pass


MJPEG_BOUNDARY = "frame"
MIN_PLATE_CROP_WIDTH = 48
MIN_PLATE_CROP_HEIGHT = 16
MIN_PLATE_CROP_AREA = 900
MIN_PLATE_ASPECT_RATIO = 1.4
MAX_PLATE_ASPECT_RATIO = 8.5
# Two-line plates (common on Indian bikes/trucks: state+RTO code on top row,
# series+number on bottom row) crop much closer to square than single-line plates.
DOUBLE_LINE_MIN_ASPECT_RATIO = 0.6
# Tried widening this to 2.2 (two genuinely two-line plates, KL35J4199 and
# AP39BD0606, had aspect ratios ~2.0-2.03, just above this 1.8 cutoff) --
# measured net WORSE on the held-out eval (247->246/253): recognize_candidates
# runs the CROP_VARIANTS single-line search regardless of is_double_line, so
# routing more crops into try_double_line=True just adds double-line
# candidates to the pool without stopping the (sometimes higher-scoring but
# wrong) single-line ones from winning -- and it misrouted a couple of
# previously-fine single-line plates (AP09BM5751, MP48BD1069) that also
# happened to sit in that 1.8-2.2 band. Reverted; see the double-line
# discussion around recognize_double_line for what actually needs fixing.
DOUBLE_LINE_MAX_ASPECT_RATIO = 1.8
MIN_PLATE_SHARPNESS = 18.0
OCR_TARGET_MIN_HEIGHT = 72
OCR_MAX_VARIANTS = 6
INDIAN_STANDARD_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$")
INDIAN_BH_PLATE_RE = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")
LETTER_SLOT_FIXES = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}
DIGIT_SLOT_FIXES = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
}


class CameraUnavailable(Exception):
    pass


def decode_image(image_bytes):
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_camera(name: str):
    for camera in settings.CAMERAS:
        if camera["name"] == name:
            return camera
    return None


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


def camera_url(camera: dict, path: str):
    base = camera["ip"].strip()
    if base.startswith("http://") or base.startswith("https://"):
        return f"{base.rstrip('/')}/{path.lstrip('/')}"
    return f"http://{base}/{path.lstrip('/')}"


def encode_jpeg(image) -> bytes | None:
    if image is None or image.size == 0:
        return None
    encoded, buffer = cv2.imencode(".jpg", image)
    if not encoded:
        return None
    return buffer.tobytes()


def save_capture_image(root_dir: str, camera_name: str, image, suffix: str = "") -> str | None:
    """Save `image` under root_dir/<YYYY-MM-DD>/, one file per capture, so
    plate crops and vehicle frames end up date-wise in their own top-level
    folders (captured_plates/ vs captured_vehicles/)."""
    if image is None or image.size == 0:
        return None

    now = datetime.now()
    date_dir = os.path.join(root_dir, now.strftime("%Y-%m-%d"))
    try:
        os.makedirs(date_dir, exist_ok=True)
    except OSError:
        return None

    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    name_parts = [camera_name, timestamp] + ([suffix] if suffix else [])
    file_path = os.path.join(date_dir, "_".join(name_parts) + ".jpg")
    if not cv2.imwrite(file_path, image):
        return None
    return file_path


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").strip().upper())


def unreadable_plate_marker() -> str:
    return f"UNREAD_{int(time.time() * 1000) % 1000000:06d}"


def plate_sharpness(image) -> float:
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def plate_crop_metrics(image) -> dict:
    if image is None or image.size == 0:
        return {
            "width": 0,
            "height": 0,
            "area": 0,
            "aspect_ratio": 0.0,
            "sharpness": 0.0,
            "is_double_line": False,
            "usable": False,
        }

    height, width = image.shape[:2]
    area = width * height
    aspect_ratio = width / float(max(height, 1))
    sharpness = plate_sharpness(image)
    is_double_line = DOUBLE_LINE_MIN_ASPECT_RATIO <= aspect_ratio <= DOUBLE_LINE_MAX_ASPECT_RATIO
    aspect_ok = (
        MIN_PLATE_ASPECT_RATIO <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO
        or is_double_line
    )
    usable = (
        width >= MIN_PLATE_CROP_WIDTH
        and height >= MIN_PLATE_CROP_HEIGHT
        and area >= MIN_PLATE_CROP_AREA
        and aspect_ok
        and sharpness >= MIN_PLATE_SHARPNESS
    )
    return {
        "width": width,
        "height": height,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "sharpness": sharpness,
        "is_double_line": is_double_line,
        "usable": usable,
    }


def plate_crop_quality_score(image, detection_confidence: float = 0.0) -> float:
    metrics = plate_crop_metrics(image)
    if not metrics["usable"]:
        return 0.0

    area_score = min(metrics["area"] / 12000.0, 1.0)
    sharpness_score = min(metrics["sharpness"] / 180.0, 1.0)
    aspect = metrics["aspect_ratio"]
    if metrics["is_double_line"]:
        aspect_score = max(0.0, 1.0 - abs(aspect - 1.0) / 1.2)
    else:
        aspect_score = max(0.0, 1.0 - abs(aspect - 4.5) / 4.0)
    return (
        float(detection_confidence) * 0.45
        + area_score * 0.20
        + sharpness_score * 0.25
        + aspect_score * 0.10
    )


def clahe_bgr(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((enhanced_l, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def sharpen_bgr(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.addWeighted(image, 1.6, blurred, -0.6, 0)


def upscale_for_ocr(image, min_height: int = OCR_TARGET_MIN_HEIGHT):
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image
    scale = max(1.0, min_height / float(height))
    if scale <= 1.05:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def deskew_plate_image(image):
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 20:
        return image

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 1.0 or abs(angle) > 18:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def find_double_line_split(image) -> int:
    """Locate the horizontal gap between a double-line plate's two text rows
    via a row-wise ink-density projection, instead of assuming the boundary
    sits at the exact vertical midpoint. Real two-line plates (state+RTO code
    over series+number) often have an uneven row height, e.g. a short top row
    over large bottom digits, so a fixed 50/50 cut bleeds one row's pixels
    into the other's crop."""
    height = image.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    row_ink = thresh.sum(axis=1).astype(np.float64)

    band_start = max(int(height * 0.25), 1)
    band_end = min(int(height * 0.75), height - 1)
    if band_end <= band_start:
        return height // 2

    band = row_ink[band_start:band_end]
    valley_offset = int(np.argmin(band))
    band_mean = float(band.mean()) if band.size else 0.0

    # No clear low-ink gap in the search band (e.g. a single-line plate
    # misclassified as double-line) -- fall back to the midpoint.
    if band_mean <= 0 or band[valley_offset] > band_mean * 0.5:
        return height // 2

    return band_start + valley_offset


def ocr_image_variants(image):
    if image is None or image.size == 0:
        return []

    variants = []
    seen = set()

    def add(name, candidate):
        if candidate is None or candidate.size == 0:
            return
        candidate = upscale_for_ocr(candidate)
        key = (candidate.shape[:2], int(candidate.mean()), int(plate_sharpness(candidate)))
        if key in seen:
            return
        seen.add(key)
        variants.append((name, candidate))

    add("original", image)
    add("clahe", clahe_bgr(image))
    add("sharp", sharpen_bgr(image))
    deskewed = deskew_plate_image(image)
    if deskewed is not image:
        add("deskew", deskewed)
        add("deskew_clahe", clahe_bgr(deskewed))
    return variants[:4]


def format_score(text: str) -> float:
    normalized = normalize_plate_text(text)
    if not normalized:
        return 0.0
    if INDIAN_STANDARD_PLATE_RE.match(normalized) or INDIAN_BH_PLATE_RE.match(normalized):
        return 0.98 if plate_state_code(normalized) else 0.9
    if 6 <= len(normalized) <= 12 and any(c.isalpha() for c in normalized) and any(c.isdigit() for c in normalized):
        return 0.45
    return 0.0


def plate_length_penalty(text: str) -> float:
    normalized = normalize_plate_text(text)
    if not normalized:
        return 0.05
    if INDIAN_STANDARD_PLATE_RE.match(normalized) or INDIAN_BH_PLATE_RE.match(normalized):
        return 0.0
    if 8 <= len(normalized) <= 11:
        return 0.0
    if 6 <= len(normalized) <= 13:
        return 0.02
    return 0.05


def fix_slots(text: str, pattern: str) -> tuple[str, int]:
    normalized = normalize_plate_text(text)
    if len(normalized) != len(pattern):
        return normalized, 999

    changes = 0
    corrected = []
    for char, slot in zip(normalized, pattern):
        if slot == "L":
            new_char = LETTER_SLOT_FIXES.get(char, char)
            if not new_char.isalpha():
                return normalized, 999
        else:
            new_char = DIGIT_SLOT_FIXES.get(char, char)
            if not new_char.isdigit():
                return normalized, 999
        if new_char != char:
            changes += 1
        corrected.append(new_char)
    return "".join(corrected), changes


INDIAN_SLOT_PATTERNS = (
    ("LLDLDDDD", "indian_8"),        # state(2) + RTO(1) + series(1) + number(4)
    ("LLDLLDDDD", "indian_9a"),      # state(2) + RTO(1) + series(2) + number(4)
    ("LLDDLDDDD", "indian_9b"),      # state(2) + RTO(2) + series(1) + number(4)
    ("LLDLLLDDDD", "indian_10a"),    # state(2) + RTO(1) + series(3) + number(4)
    ("LLDDLLDDDD", "indian_10b"),    # state(2) + RTO(2) + series(2) + number(4)
    ("LLDDLLLDDDD", "indian_11"),    # state(2) + RTO(2) + series(3) + number(4)
)


def corrected_plate_candidates(text: str) -> list[tuple[str, float, str]]:
    normalized = normalize_plate_text(text)
    if not normalized:
        return []

    candidates = [(normalized, format_score(normalized), "raw")]
    for pattern, tag in INDIAN_SLOT_PATTERNS:
        corrected, changes = fix_slots(normalized, pattern)
        if changes <= 2 and corrected != normalized:
            candidates.append((corrected, max(format_score(corrected) - changes * 0.05, 0.0), tag))

    for corrected, changes, tag in extra_char_removal_candidates(normalized):
        if corrected != normalized:
            candidates.append((corrected, max(format_score(corrected) - changes * 0.05, 0.0), tag))

    for value, score, tag in list(candidates):
        for state_fixed, state_changes in correct_state_code_candidates(value):
            if state_changes == 1 and state_fixed != value:
                # Score relative to `value`'s own (already change-penalised) score,
                # not a fresh format_score(state_fixed) -- that flat rescore let a
                # candidate that needed several other changes to reach a
                # plausible-but-wrong shape "reset" to a near-perfect score just by
                # also having a fixable state code, letting it outscore a cleaner
                # candidate that needed fewer changes overall.
                state_fix_gain = format_score(state_fixed) - format_score(value)
                candidates.append((state_fixed, max(score + state_fix_gain - 0.03, 0.0), f"{tag}+statefix"))

    best_by_text = {}
    for value, score, tag in candidates:
        if value not in best_by_text or score > best_by_text[value][0]:
            best_by_text[value] = (score, tag)
    return [(value, score, tag) for value, (score, tag) in best_by_text.items()]


def choose_corrected_plate(text: str) -> tuple[str, str]:
    candidates = corrected_plate_candidates(text)
    if not candidates:
        return normalize_plate_text(text), "raw"
    value, _, tag = max(candidates, key=lambda item: (item[1], -abs(len(item[0]) - len(normalize_plate_text(text)))))
    return value, tag


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def similar_plate_text(left: str, right: str) -> bool:
    left = normalize_plate_text(left)
    right = normalize_plate_text(right)
    if not left or not right:
        return False
    if abs(len(left) - len(right)) > 2:
        return False
    return levenshtein_distance(left, right) <= (1 if max(len(left), len(right)) <= 7 else 2)


def _align_to_reference(reference: str, other: str) -> list[str]:
    """Map `other` onto `reference`'s positions via sequence alignment, so a
    frame whose read is a character shorter/longer (e.g. one frame dropped
    a digit) can still contribute per-position votes instead of being
    excluded from consensus outright. Positions with no aligned character
    (an insertion/deletion) are left as "" and simply don't vote there."""
    aligned = [""] * len(reference)
    matcher = difflib.SequenceMatcher(None, reference, other, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("equal", "replace"):
            span = min(i2 - i1, j2 - j1)
            for offset in range(span):
                aligned[i1 + offset] = other[j1 + offset]
    return aligned


def consensus_plate_text(texts: list[str]) -> str:
    normalized = [normalize_plate_text(text) for text in texts if normalize_plate_text(text)]
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0]

    length_counts = Counter(len(text) for text in normalized)
    target_length = length_counts.most_common(1)[0][0]
    same_length = [text for text in normalized if len(text) == target_length]
    reference = same_length[0]

    consensus = []
    for index in range(target_length):
        votes = Counter()
        for text in normalized:
            if len(text) == target_length:
                votes[text[index]] += 1
            else:
                aligned_char = _align_to_reference(reference, text)[index]
                if aligned_char:
                    votes[aligned_char] += 1
        consensus.append(votes.most_common(1)[0][0] if votes else reference[index])
    corrected, _ = choose_corrected_plate("".join(consensus))
    return corrected


def _rerank_by_crop_agreement(candidates: list[dict]) -> list[dict]:
    """A single crop-trim ratio can occasionally clip or distort one
    character into a look-alike (e.g. X into Y) and still come back with
    very high model confidence -- high enough to outscore every other
    crop that reads the plate correctly, once every crop/enhancement
    combination gets tried instead of stopping at the first passable one.
    Multiple *color/contrast* variants (original/clahe/sharp) of that same
    crop framing all inherit its mistake, so they aren't independent
    confirmations of it. Reward candidates that distinct crop framings
    (not just distinct color variants of the same framing) agree on, so a
    lone high-confidence outlier doesn't beat a plate reading several
    differently-framed crops converge on."""
    if not candidates:
        return candidates

    groups = []
    for candidate in candidates:
        crop_variant = candidate["variant"].split(":", 2)[1] if ":" in candidate["variant"] else candidate["variant"]
        for group in groups:
            if similar_plate_text(candidate["text"], group["representative"]):
                group["members"].append(candidate)
                group["crop_variants"].add(crop_variant)
                break
        else:
            groups.append({
                "representative": candidate["text"],
                "members": [candidate],
                "crop_variants": {crop_variant},
            })

    def group_rank(group):
        best = max(group["members"], key=lambda c: c["score"])
        support_bonus = min(len(group["crop_variants"]) - 1, 4) * 0.03
        return best["score"] + support_bonus

    best_group = max(groups, key=group_rank)
    winner = max(best_group["members"], key=lambda c: c["score"])
    # Multiple color/contrast variants of the same crop framing aren't
    # independent votes, so raw majority-of-members voting can let a
    # same-length character mistake that most framings happen to share
    # outvote the single best-scoring (already confidence+format weighted)
    # reading. Only defer to consensus voting when the group actually mixes
    # different *lengths* -- i.e. some crop genuinely dropped/gained a
    # character (the CTC-collapse case) -- where recovering the fuller
    # reading via alignment is worth more than trusting one crop's score.
    member_lengths = {len(member["text"]) for member in best_group["members"]}
    if len(member_lengths) > 1:
        consensus_text = consensus_plate_text([member["text"] for member in best_group["members"]])
        if consensus_text and len(consensus_text) >= len(winner["text"]) and consensus_text != winner["text"]:
            winner = dict(winner)
            winner["text"] = consensus_text
    return [winner] + [c for c in candidates if c is not winner]


def consensus_label(labels: list[str]) -> str:
    known = [label for label in labels if label and label != "Unknown"]
    if not known:
        return "Unknown"
    return Counter(known).most_common(1)[0][0]


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


def open_video_stream(camera: dict):
    capture = open_video_capture(camera)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    fps = float(camera.get("video_fps") or source_fps or 25.0)
    delay = 1.0 / max(fps, 1.0)
    loop = bool(camera.get("video_loop", True))

    def generate():
        try:
            while True:
                frame = read_video_frame(capture, loop=loop)
                if frame is None:
                    break

                image_bytes = encode_jpeg(frame)
                if image_bytes is None:
                    continue

                yield (
                    f"--{MJPEG_BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(image_bytes)}\r\n\r\n"
                ).encode("ascii") + image_bytes + b"\r\n"
                time.sleep(delay)
        finally:
            capture.release()

    return f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}", generate()


def open_mjpeg_stream(name: str):
    camera = find_camera(name)
    if not camera:
        return None
    if is_video_source(camera):
        return open_video_stream(camera)

    try:
        upstream = requests.get(
            camera_url(camera, settings.CAMERA_STREAM_PATH),
            auth=digest_auth(),
            stream=True,
            timeout=10,
        )
        upstream.raise_for_status()
    except requests.RequestException as exc:
        raise CameraUnavailable(str(exc)) from exc

    media_type = upstream.headers.get("Content-Type", "multipart/x-mixed-replace")

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return media_type, generate()


def snapshot_params(camera: dict) -> dict:
    """Axis VAPIX params so snapshots come back at the resolution/quality we
    need for plate OCR instead of whatever the camera defaults to."""
    params = {}
    resolution = camera.get("snapshot_resolution", settings.CAMERA_SNAPSHOT_RESOLUTION)
    if resolution:
        params["resolution"] = resolution
    compression = camera.get("snapshot_compression", settings.CAMERA_SNAPSHOT_COMPRESSION)
    if compression is not None:
        params["compression"] = compression
    return params


def get_snapshot(name: str) -> bytes | None:
    camera = find_camera(name)
    if not camera:
        return None
    if is_video_source(camera):
        capture = open_video_capture(camera)
        try:
            frame = read_video_frame(capture, loop=True)
            return encode_jpeg(frame)
        finally:
            capture.release()

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


def open_audio_stream(name: str):
    camera = find_camera(name)
    if not camera:
        return None
    if is_video_source(camera):
        raise CameraUnavailable("Audio is not available for local video sources.")

    url = camera_url(camera, "/axis-cgi/audio/receive.cgi")
    auth = digest_auth()
    try:
        upstream = requests.get(url, auth=auth, stream=True, timeout=10)
        if upstream.status_code == 405:
            upstream.close()
            upstream = requests.post(url, auth=auth, stream=True, timeout=10)
        upstream.raise_for_status()
    except requests.RequestException as exc:
        raise CameraUnavailable(str(exc)) from exc

    media_type = upstream.headers.get("Content-Type", "audio/basic")

    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=2048):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return media_type, generate()


def ptz_move(name: str, rpan: float = 0, rtilt: float = 0, rzoom: float = 0) -> None:
    camera = find_camera(name)
    if not camera:
        raise ValueError("Unknown camera.")
    if is_video_source(camera):
        raise CameraUnavailable("PTZ is not available for local video sources.")

    params = {k: v for k, v in {"rpan": rpan, "rtilt": rtilt, "rzoom": rzoom}.items() if v}
    if not params:
        return

    try:
        response = requests.get(
            camera_url(camera, "/axis-cgi/com/ptz.cgi"),
            params=params,
            auth=digest_auth(),
            timeout=6,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CameraUnavailable(str(exc)) from exc


def ptz_continuous_move(name: str, pan: float = 0, tilt: float = 0) -> None:
    camera = find_camera(name)
    if not camera:
        raise ValueError("Unknown camera.")
    if is_video_source(camera):
        raise CameraUnavailable("PTZ is not available for local video sources.")

    try:
        response = requests.get(
            camera_url(camera, "/axis-cgi/com/ptz.cgi"),
            params={"continuouspantiltmove": f"{pan},{tilt}"},
            auth=digest_auth(),
            timeout=6,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CameraUnavailable(str(exc)) from exc


class PlateDetector:
    def __init__(self, model_path, confidence=0.35, classes=None, padding_ratio=0.15, device="cpu"):
        self.model_path = model_path
        self.confidence = confidence
        self.allowed_classes = {c.lower() for c in (classes or [])}
        self.padding_ratio = padding_ratio
        self.device = device
        self.model = YOLO(model_path) if os.path.exists(model_path) else None
        if self.model is None:
            print(f"Plate detector model not found: {model_path}")

    @property
    def available(self):
        return self.model is not None

    def detect(self, image):
        detections = self.detect_many(image)
        if not detections:
            return None
        return detections[0]

    def detect_many(self, image):
        if self.model is None or image is None or image.size == 0:
            return []

        results = self.model.predict(image, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        h, w = image.shape[:2]
        candidates = []
        for box in boxes:
            label = self.model.names[int(box.cls)]
            confidence = float(box.conf)
            if confidence < self.confidence:
                continue
            if self.allowed_classes and label.lower() not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            box_w = max(x2 - x1, 1)
            box_h = max(y2 - y1, 1)
            pad_x = int(box_w * self.padding_ratio)
            pad_y = int(box_h * self.padding_ratio)

            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w, x2 + pad_x)
            crop_y2 = min(h, y2 + pad_y)
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue

            metrics = plate_crop_metrics(crop)
            if not metrics["usable"]:
                continue

            quality_score = plate_crop_quality_score(crop, confidence)
            candidates.append({
                "crop": crop,
                "bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                "raw_bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "label": label,
                "sharpness": metrics["sharpness"],
                "crop_width": metrics["width"],
                "crop_height": metrics["height"],
                "aspect_ratio": metrics["aspect_ratio"],
                "is_double_line": metrics["is_double_line"],
                "score": quality_score,
            })

        if not candidates:
            return []

        candidates.sort(key=lambda item: item["score"], reverse=True)
        for candidate in candidates:
            candidate.pop("score", None)
        return candidates


class PlateOCR:
    IMG_H = 48
    IMG_W = 320
    CROP_VARIANTS = (
        ("original", 0.00, 1.00),
        ("trim10", 0.10, 0.90),
        ("trim18", 0.18, 0.88),
        ("trim25", 0.25, 0.86),
        ("center60", 0.20, 0.80),
        ("center50", 0.25, 0.75),
    )

    def __init__(self, model_dir, char_dict_path, use_space_char=True):
        model_file = os.path.join(model_dir, "inference.json")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "inference.pdmodel")
        params_file = os.path.join(model_dir, "inference.pdiparams")

        config = inference.Config(model_file, params_file)
        config.disable_gpu()
        config.disable_glog_info()
        try:
            config.set_cpu_math_library_num_threads(settings.RUNTIME_PADDLE_CPU_THREADS)
        except Exception:
            pass

        self.predictor = inference.create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])

        with open(char_dict_path, "rb") as f:
            chars = [line.decode("utf-8").strip("\r\n") for line in f.readlines()]
        if use_space_char:
            chars.append(" ")
        self.character = ["blank"] + chars

    def _resize_norm(self, img):
        h, w = img.shape[:2]
        ratio = w / float(h)
        max_wh_ratio = max(self.IMG_W / float(self.IMG_H), ratio)
        img_w = int(self.IMG_H * max_wh_ratio)
        resized_w = img_w if math.ceil(self.IMG_H * ratio) > img_w else math.ceil(self.IMG_H * ratio)

        resized = cv2.resize(img, (resized_w, self.IMG_H))
        resized = resized.astype("float32").transpose((2, 0, 1)) / 255.0
        resized -= 0.5
        resized /= 0.5

        padded = np.zeros((3, self.IMG_H, img_w), dtype=np.float32)
        padded[:, :, :resized_w] = resized
        return padded

    def recognize(self, img):
        if img is None or img.size == 0:
            return "", 0.0

        norm_img = self._resize_norm(img)[np.newaxis, :]
        self.input_handle.reshape(norm_img.shape)
        self.input_handle.copy_from_cpu(norm_img)
        self.predictor.run()
        preds = self.output_handle.copy_to_cpu()

        preds_idx = preds.argmax(axis=2)[0]
        preds_prob = preds.max(axis=2)[0]

        chars, confs = self._decode_ctc(preds_idx, preds_prob)
        return "".join(chars), float(np.mean(confs)) if confs else 0.0

    def _decode_ctc(self, preds_idx, preds_prob):
        """Greedy CTC decode with a repeated-character recovery step.

        Plain CTC decoding collapses every run of consecutive identical
        predicted classes into one character, which is correct when a run
        is just one character stretched across several timesteps -- but a
        genuinely doubled character (e.g. the "99" in "9924") can also
        render as a single unbroken run if the model never emits a blank
        between the two instances, and plain collapsing then silently
        drops one of the two digits/letters. Measured against real
        misreads, a true doubled-character run is consistently >=3
        timesteps long while an ordinary single character's run is 1
        (occasionally 2) -- so only runs at or above that length are
        treated as two repeated characters, keeping the common case
        (single character, possibly width-2) untouched.
        """
        runs = []
        n = len(preds_idx)
        i = 0
        while i < n:
            j = i
            while j < n and preds_idx[j] == preds_idx[i]:
                j += 1
            if preds_idx[i] != 0:
                # Match the original decoder's confidence exactly (probability
                # at the run's first timestep) so the only behavior change is
                # the repeated-character recovery below -- using the run's
                # mean instead shifts confidence for every character, which
                # can flip which candidate crop/variant wins downstream.
                runs.append((preds_idx[i], j - i, float(preds_prob[i])))
            i = j

        # A fixed ">=3 timesteps" cutoff assumes every image resizes to
        # roughly the same per-character timestep width, but _resize_norm's
        # width scales with the crop's own aspect ratio -- a wide crop gives
        # every ordinary single character more timesteps too, so the same
        # fixed cutoff over- or under-fires depending on image size. Scaling
        # the cutoff to this sequence's own typical (median) run length
        # instead means "this run is unusually long FOR THIS IMAGE", which is
        # the actual signal a doubled character leaves.
        non_blank_lengths = sorted(length for _, length, _ in runs)
        if non_blank_lengths:
            median_length = non_blank_lengths[len(non_blank_lengths) // 2]
            repeat_threshold = max(3, round(median_length * 1.7))
        else:
            repeat_threshold = 3

        chars = []
        confs = []
        for class_idx, length, prob in runs:
            repeats = 2 if length >= repeat_threshold else 1
            char = self.character[class_idx]
            chars.extend([char] * repeats)
            confs.extend([prob] * repeats)
        return chars, confs

    def recognize_double_line(self, img):
        """Split a two-line plate (bike/truck: state+RTO on top, series+number on bottom)
        into its two rows and OCR each row separately, then concatenate."""
        if img is None or img.size == 0:
            return "", 0.0, None, None

        height = img.shape[0]
        if height < 20:
            return "", 0.0, None, None

        mid = find_double_line_split(img)
        # A bigger overlap avoids clipping a character that straddles the
        # split, but the same overlap band can get OCR'd twice -- once as
        # the tail of top_row, once as the head of bottom_row -- producing a
        # duplicated/hallucinated extra character right at the join (e.g.
        # "GJ07CF7956" misread as "GJ07GCF7956"). Halved from 0.08 since that
        # failure mode showed up more often than genuine clipping in the
        # held-out eval -- re-check ocr_training/eval_ocr_accuracy.py's
        # double-line examples if tuning this further.
        overlap = max(int(height * 0.04), 2)
        top_row = img[0:min(mid + overlap, height), :]
        bottom_row = img[max(mid - overlap, 0):, :]

        top_text, top_confidence = self.recognize(upscale_for_ocr(top_row))
        bottom_text, bottom_confidence = self.recognize(upscale_for_ocr(bottom_row))
        top_text = normalize_plate_text(top_text)
        bottom_text = normalize_plate_text(bottom_text)
        if not top_text and not bottom_text:
            return "", 0.0, top_row, bottom_row

        confidences = [c for c in (top_confidence, bottom_confidence) if c > 0]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return top_text + bottom_text, confidence, top_row, bottom_row

    def recognize_candidates(self, img):
        if img is None or img.size == 0:
            return []

        candidates = []
        seen = set()
        attempts = 0
        crop_metrics = plate_crop_metrics(img)
        try_double_line = crop_metrics["is_double_line"]

        def add_candidate(raw_text, confidence, variant_tag, crop_image=None):
            text = normalize_plate_text(raw_text)
            if not text:
                return
            for corrected_text, candidate_format_score, correction_tag in corrected_plate_candidates(text):
                key = (corrected_text, variant_tag, correction_tag)
                if key in seen:
                    continue
                seen.add(key)
                validity_score = 1.0 if is_valid_plate_format(corrected_text) else 0.0
                score = (
                    confidence * 0.68
                    + candidate_format_score * 0.22
                    + validity_score * 0.10
                    - plate_length_penalty(corrected_text)
                )
                candidates.append({
                    "text": corrected_text,
                    "raw_text": text,
                    "confidence": confidence,
                    "variant": f"{variant_tag}:{correction_tag}",
                    "format_score": candidate_format_score,
                    "valid": bool(validity_score),
                    "score": score,
                    # Kept only so recognize_best can hand the SAR refiner the
                    # exact crop CTC's fast search already picked as best --
                    # not used anywhere else, and never on double-line crops
                    # (those aren't a single rectangle SAR can read directly).
                    "crop": crop_image,
                })

        # Try every image-enhancement x crop-trim combination (bounded by
        # OCR_MAX_VARIANTS) instead of stopping at the first "confident
        # enough" hit -- a tighter/looser crop tried later is often
        # meaningfully cleaner than an earlier one that happened to score
        # just high enough to look done (e.g. a wide trim can misread one
        # character with very high confidence while a tighter trim tried a
        # few attempts later reads the whole plate perfectly). The final
        # score-sorted pick at the end already favours the best candidate
        # seen, so nothing is lost by looking at all of them.
        for image_variant, variant_image in ocr_image_variants(img):
            if try_double_line:
                attempts += 1
                raw_text, confidence, top_row, bottom_row = self.recognize_double_line(variant_image)
                # crop_image carries a (top_row, bottom_row) tuple here instead
                # of a single crop -- recognize_best tells the two apart with
                # isinstance() to know whether to hand the SAR refiner one
                # rectangle or a top/bottom pair.
                add_candidate(raw_text, confidence, f"{image_variant}:double_line", crop_image=(top_row, bottom_row))

            for crop_variant, top_ratio, bottom_ratio in self.CROP_VARIANTS:
                if attempts >= OCR_MAX_VARIANTS:
                    break

                h = variant_image.shape[0]
                top = int(h * top_ratio)
                bottom = int(h * bottom_ratio)
                crop = variant_image[top:bottom, :]
                if crop.size == 0:
                    continue

                attempts += 1
                raw_text, confidence = self.recognize(crop)
                add_candidate(raw_text, confidence, f"{image_variant}:{crop_variant}", crop_image=crop)
            if attempts >= OCR_MAX_VARIANTS:
                break

        candidates.sort(key=lambda item: item["score"], reverse=True)
        return _rerank_by_crop_agreement(candidates)

    def recognize_best(self, img):
        candidates = self.recognize_candidates(img)
        if not candidates:
            return "", 0.0, ""
        best = candidates[0]
        crop = best.get("crop")

        # SAR refinement: CTC's fast multi-crop-variant search above already
        # found the best-framed crop; re-read just that one (much slower, run
        # once, not per-variant) with the attention-based SAR head, which
        # doesn't have CTC's blank-collapse failure mode -- a 253-image
        # held-out comparison found SAR exact-matched 98.81% vs this CTC
        # pipeline's 87.75% on single-line crops, and never lost to CTC on a
        # single image. `crop` is a (top_row, bottom_row) tuple for a
        # double-line winner (see recognize_double_line) and a single ndarray
        # otherwise -- route to whichever SAR method matches.
        is_double_line_crop = isinstance(crop, tuple)
        crop_usable = crop is not None and not (is_double_line_crop and (crop[0] is None or crop[1] is None))
        if sar_refiner is not None and crop_usable:
            try:
                if is_double_line_crop:
                    sar_text, sar_confidence = sar_refiner.recognize_double_line(*crop)
                else:
                    sar_text, sar_confidence = sar_refiner.recognize(crop)
                sar_text = normalize_plate_text(sar_text)
            except Exception as exc:
                print(f"SAR refine error (falling back to CTC reading): {exc}")
                sar_text, sar_confidence = "", 0.0
            if sar_text:
                # Grammar-clean only the double-line reading (plain
                # concatenation never gets a correction pass otherwise) --
                # NOT the single-line one, which measured worse (247->246/253
                # held-out) when this was applied unconditionally, likely
                # because a still-correct raw SAR reading occasionally got
                # "corrected" into something else.
                if is_double_line_crop:
                    sar_text, _correction_tag = choose_corrected_plate(sar_text)
                return sar_text, sar_confidence, f"{best['variant']}+sar"

        return best["text"], best["confidence"], best["variant"]


def classify_vehicle_color_bgr(bgr):
    """Name a single BGR colour the way colour_detection.py's classify_color did."""
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = map(int, hsv)

    if v < 50:
        return "Black"

    if s < 35:
        if v > 200:
            return "White"
        elif v > 130:
            return "Silver / Light Grey"
        return "Grey"

    if h < 10 or h >= 170:
        return "Red"
    elif h < 22:
        return "Orange"
    elif h < 35:
        return "Yellow"
    elif h < 85:
        return "Green"
    elif h < 130:
        return "Blue"
    elif h < 160:
        return "Purple"
    return "Red"


def dominant_vehicle_color(pixels):
    """Median-BGR colour name for a set of vehicle-body pixels, ignoring very
    dark pixels (tyres/windows/shadows) same as colour_detection.py."""
    pixels = pixels.reshape(-1, 3)
    if len(pixels) == 0:
        return "Unknown"

    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lit = pixels[hsv_pixels[:, 2] > 50]
    if len(lit) == 0:
        return "Unknown"

    median_bgr = np.median(lit, axis=0).astype(np.uint8)
    return classify_vehicle_color_bgr(median_bgr)


class VehicleColorDetector:
    VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

    def __init__(self, detector_model_path, segmentation_model_path, type_classifier_model_path, confidence=0.5, device="cpu"):
        self.detector = YOLO(detector_model_path)
        self.segmenter = YOLO(segmentation_model_path)
        self.type_classifier = YOLO(type_classifier_model_path)
        self.confidence = confidence
        self.device = device

    def _segmentation_color(self, image):
        """Segment the largest vehicle in `image` (yolo11n-seg.pt COCO masks)
        and take the median colour of just its body pixels; falls back to the
        whole crop when no mask is found, same as colour_detection.py."""
        seg_results = self.segmenter(image, device=self.device, verbose=False)
        result = seg_results[0]

        best_mask = None
        largest_area = 0

        if result.boxes is not None and result.masks is not None:
            for i, box in enumerate(result.boxes):
                class_name = self.segmenter.names[int(box.cls[0])]
                if class_name not in self.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    best_mask = result.masks.data[i].cpu().numpy()

        if best_mask is None:
            return dominant_vehicle_color(image)

        mask = cv2.resize(
            best_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = mask > 0.5

        vehicle_pixels = image[mask]
        if len(vehicle_pixels) == 0:
            return dominant_vehicle_color(image)

        return dominant_vehicle_color(vehicle_pixels)

    def vehicle_candidates(self, image, confidence: float | None = None, min_area_ratio: float = 0.0):
        if image is None or image.size == 0:
            return []

        results = self.detector.predict(image, device=self.device, verbose=False)
        detections = results[0].boxes
        if detections is None:
            return []

        min_confidence = self.confidence if confidence is None else confidence
        frame_area = max(image.shape[0] * image.shape[1], 1)
        candidates = []
        for det in detections:
            label = self.detector.names[int(det.cls)]
            det_confidence = float(det.conf)
            if label in self.VEHICLE_CLASSES and det_confidence >= min_confidence:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                area = max(x2 - x1, 1) * max(y2 - y1, 1)
                if area / frame_area < min_area_ratio:
                    continue
                candidates.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "confidence": det_confidence,
                    "area": area,
                })
        return candidates

    def classify_roi(self, image):
        if image is None or image.size == 0:
            return "Unknown", "Unknown"

        color_label = self._segmentation_color(image)

        roi_resized = cv2.resize(image, (480, 480))
        type_result = self.type_classifier.predict(source=roi_resized, imgsz=480, device=self.device, verbose=False)
        type_boxes = type_result[0].boxes
        type_label = "Unknown"
        if type_boxes is not None and len(type_boxes) > 0:
            best_box = max(type_boxes, key=lambda b: float(b.conf))
            type_label = self.type_classifier.names[int(best_box.cls)]

        return type_label, color_label

    def detect(self, image):
        candidates = self.vehicle_candidates(image)
        if not candidates:
            return "Unknown", "Unknown"

        best = max(candidates, key=lambda item: item["confidence"] * item["area"])
        x1, y1, x2, y2 = best["bbox"]
        return self.classify_roi(image[y1:y2, x1:x2])

    def classify_for_plate(self, image, plate_bbox):
        if image is None or image.size == 0:
            return "Unknown", "Unknown", "none"
        if not plate_bbox:
            vehicle_type, vehicle_color = self.detect(image)
            return vehicle_type, vehicle_color, "vehicle_detector"

        h, w = image.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in plate_bbox]
        plate_cx = (px1 + px2) / 2
        plate_cy = (py1 + py2) / 2
        plate_w = max(px2 - px1, 1)
        plate_h = max(py2 - py1, 1)

        matched = []
        for candidate in self.vehicle_candidates(image):
            x1, y1, x2, y2 = candidate["bbox"]
            contains_center = x1 <= plate_cx <= x2 and y1 <= plate_cy <= y2
            ix1, iy1 = max(x1, px1), max(y1, py1)
            ix2, iy2 = min(x2, px2), min(y2, py2)
            overlap = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
            if contains_center or overlap > 0:
                matched.append((candidate, overlap))

        if matched:
            candidate = max(matched, key=lambda item: (item[1], item[0]["confidence"] * item[0]["area"]))[0]
            x1, y1, x2, y2 = candidate["bbox"]
            vehicle_type, vehicle_color = self.classify_roi(image[y1:y2, x1:x2])
            return vehicle_type, vehicle_color, "matched_vehicle"

        roi_x1 = max(0, int(px1 - plate_w * 2.5))
        roi_x2 = min(w, int(px2 + plate_w * 2.5))
        roi_y1 = max(0, int(py1 - plate_h * 5.0))
        roi_y2 = min(h, int(py2 + plate_h * 2.5))
        vehicle_type, vehicle_color = self.classify_roi(image[roi_y1:roi_y2, roi_x1:roi_x2])
        return vehicle_type, vehicle_color, "plate_anchor_roi"

    def detect_many(self, image, plate_detections):
        detections = []
        for plate in plate_detections:
            vehicle_type, vehicle_color, vehicle_source = self.classify_for_plate(image, plate.get("bbox"))
            detections.append({
                **plate,
                "vehicle_type": vehicle_type,
                "vehicle_color": vehicle_color,
                "vehicle_source": vehicle_source,
            })
        return detections



INDIAN_STATE_CODES = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PN": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TG": "Telangana",
    "TR": "Tripura",
    "TS": "Telangana",
    "UA": "Uttarakhand",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}
def plate_state_code(text: str) -> str:
    normalized = normalize_plate_text(text)
    if len(normalized) >= 4 and normalized[2:4] == "BH":
        return "BH"

    state_code = normalized[:2]
    return state_code if state_code in INDIAN_STATE_CODES else ""


DIGIT_TO_LETTER_CANDIDATES = {
    "0": ("O", "D", "Q"),
    "1": ("I", "L"),
    "2": ("Z",),
    "4": ("A",),
    "5": ("S",),
    "6": ("G",),
    "8": ("B",),
}

# Same idea, but for a letter misread as a different letter rather than as a
# digit -- e.g. Delhi's "DL" prefix routinely comes back "OL" because a
# slightly rounded D reads as O. DIGIT_TO_LETTER_CANDIDATES can't catch this
# since neither character involved is a digit.
LETTER_TO_LETTER_CANDIDATES = {
    "O": ("D", "Q", "C"),
    "D": ("O",),
    "Q": ("O",),
    "C": ("G", "O"),
    "G": ("C",),
    "I": ("L",),
    "L": ("I",),
    "V": ("Y",),
    "Y": ("V",),
}


def _state_code_char_candidates(ch: str) -> tuple[str, ...]:
    options = {ch}
    options.update(DIGIT_TO_LETTER_CANDIDATES.get(ch, ()))
    options.update(LETTER_TO_LETTER_CANDIDATES.get(ch, ()))
    return tuple(options)


def _confusable_state_code_fixes(prefix: str) -> list[str]:
    """Try correcting a 2-char state-code prefix using known look-alikes
    (digit-vs-letter, e.g. '1' for 'L' in "D1"; or letter-vs-letter, e.g. 'O'
    for 'D' in "OL") instead of blind edit-distance. Plain edit-distance
    treats every substitution as equally likely, so a prefix like "OL" is
    just as "close" to DL/OD/OL-typo'd-a-dozen-ways -- but O only visually
    resembles a handful of letters, so trying only the visually-plausible
    substitutions narrows the field a lot, even when it doesn't narrow to
    one."""
    candidate_chars = [_state_code_char_candidates(ch) for ch in prefix]
    return sorted({
        first + second
        for first in candidate_chars[0]
        for second in candidate_chars[1]
        if (first + second) in INDIAN_STATE_CODES and (first + second) != prefix
    })


def correct_state_code_candidates(text: str) -> list[tuple[str, int]]:
    """Return every plausible state-code fix for `text`'s first two
    characters -- there can be more than one. A prefix like "OL" is one
    visual substitution away from both "DL" (Delhi) and "OD" (Odisha); rather
    than refuse to guess (the old behaviour), hand both back as separate
    candidates and let the normal whole-plate scoring -- grammar match on the
    rest of the plate, OCR confidence, multi-crop/multi-frame agreement --
    pick the winner, the same way it already disambiguates any other
    same-length misread."""
    normalized = normalize_plate_text(text)
    if len(normalized) < 4 or plate_state_code(normalized):
        return []

    prefix = normalized[:2]
    fixes = _confusable_state_code_fixes(prefix)
    if not fixes:
        fixes = [code for code in INDIAN_STATE_CODES if levenshtein_distance(prefix, code) == 1]
    return [(fix + normalized[2:], 1) for fix in fixes]


def extra_char_removal_candidates(text: str) -> list[tuple[str, int, str]]:
    """Recover a reading that gained one spurious character -- e.g. a single
    wide/stretched 'L' coming back as two narrow strokes, "11" -- by trying
    every single-character removal and keeping the ones that turn into a
    fully valid Indian plate. Unlike a genuinely dropped character, removing
    one never requires guessing an unknown value, so trying every position is
    safe (each still has to pass the same slot-pattern + state-code checks as
    everything else)."""
    normalized = normalize_plate_text(text)
    candidates = []
    for pattern, tag in INDIAN_SLOT_PATTERNS:
        if len(normalized) != len(pattern) + 1:
            continue
        for i in range(len(normalized)):
            trimmed = normalized[:i] + normalized[i + 1:]
            fixed, changes = fix_slots(trimmed, pattern)
            if changes > 1:
                continue
            # The removal + type-fix alone is enough for most cases (e.g.
            # "D11 2CT0820" -> drop a '1' -> "D112CT0820"), but a compound
            # error -- extra char *and* a state-code look-alike, like "D111..."
            # -- also needs the state-code fix layered on top before it looks
            # like a real state; try both.
            if plate_state_code(fixed):
                candidates.append((fixed, changes + 1, f"{tag}_trim{i}"))
            else:
                for state_fixed, _ in correct_state_code_candidates(fixed):
                    candidates.append((state_fixed, changes + 2, f"{tag}_trim{i}+statefix"))
    return candidates


def is_valid_plate_format(text: str) -> bool:
    normalized = normalize_plate_text(text)
    return (
        6 <= len(normalized) <= 12
        and any(char.isalpha() for char in normalized)
        and any(char.isdigit() for char in normalized)
    )


def detect_plate_color(plate_image):
    """HSV mask-scoring plate colour classifier, ported from plate_colour.py.
    Trims the outer edges (bumper/plate-frame bleed into the padded crop),
    then scores each candidate colour by the fraction of ROI pixels that
    fall in its HSV range and returns the best-scoring one."""
    if plate_image is None or plate_image.size == 0:
        return "Unknown"

    h, w = plate_image.shape[:2]

    roi = plate_image[
        int(h * 0.18): int(h * 0.82),
        int(w * 0.12): int(w * 0.88),
    ]
    if roi.size == 0:
        roi = plate_image

    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0:
        return "Unknown"

    white_mask = (S < 70) & (V > 80)
    yellow_mask = (H >= 15) & (H <= 40) & (S > 60) & (V > 70)
    green_mask = (H >= 35) & (H <= 95) & (S > 50) & (V > 50)
    blue_mask = (H >= 90) & (H <= 140) & (S > 50) & (V > 50)
    red_mask = (
        ((H >= 0) & (H <= 10)) | ((H >= 165) & (H <= 179))
    ) & (S > 70) & (V > 50)
    black_mask = V < 55

    scores = {
        "White": np.count_nonzero(white_mask) / total_pixels,
        "Yellow": np.count_nonzero(yellow_mask) / total_pixels,
        "Green": np.count_nonzero(green_mask) / total_pixels,
        "Blue": np.count_nonzero(blue_mask) / total_pixels,
        "Red": np.count_nonzero(red_mask) / total_pixels,
        "Black": np.count_nonzero(black_mask) / total_pixels,
    }

    return max(scores, key=scores.get)


inference_semaphore = threading.Semaphore(settings.RUNTIME_MAX_CONCURRENT_INFERENCE)

vehicle_color_detector = VehicleColorDetector(
    detector_model_path=settings.VEHICLE_DETECTOR_MODEL_PATH,
    segmentation_model_path=settings.VEHICLE_SEGMENTATION_MODEL_PATH,
    type_classifier_model_path=settings.TYPE_CLASSIFIER_MODEL_PATH,
    confidence=settings.VEHICLE_COLOR_CONFIDENCE,
    device=settings.VEHICLE_COLOR_DEVICE,
)

plate_detector = (
    PlateDetector(
        model_path=settings.PLATE_DETECTOR_MODEL_PATH,
        confidence=settings.PLATE_DETECTOR_CONFIDENCE,
        classes=settings.PLATE_DETECTOR_CLASSES,
        padding_ratio=settings.PLATE_DETECTOR_PADDING_RATIO,
        device=settings.VEHICLE_COLOR_DEVICE,
    )
    if settings.PLATE_DETECTOR_ENABLED
    else None
)

plate_ocr = (
    PlateOCR(
        model_dir=settings.PLATE_OCR_MODEL_DIR,
        char_dict_path=settings.PLATE_OCR_CHAR_DICT_PATH,
    )
    if settings.PLATE_OCR_ENABLED
    else None
)

sar_refiner = None
if settings.PLATE_OCR_ENABLED and settings.PLATE_OCR_SAR_REFINE_ENABLED:
    try:
        from sar_refiner import SARRefiner

        sar_refiner = SARRefiner(
            checkpoint_path=settings.PLATE_OCR_SAR_REFINE_CHECKPOINT_PATH,
            char_dict_path=settings.PLATE_OCR_CHAR_DICT_PATH,
        )
    except Exception as exc:
        print(f"SAR refiner disabled (failed to load): {exc}")
        sar_refiner = None


def _process_frame_impl(overview_image=None, plate_image=None, include_plate_crop=False):
    vehicle_type, vehicle_color, plate_color = "Unknown", "Unknown", "Unknown"
    plate_text, plate_confidence, plate_text_valid = "", 0.0, False
    plate_bbox, plate_detection_confidence, plate_detection_label = None, 0.0, ""
    plate_ocr_variant, plate_state, vehicle_source = "", "", ""
    plate_crop_sharpness, plate_crop_quality = 0.0, 0.0
    plate_detector_available = bool(plate_detector and plate_detector.available)
    plate_candidates = []

    if plate_image is not None:
        metrics = plate_crop_metrics(plate_image)
        if metrics["usable"]:
            plate_candidates.append({
                "crop": plate_image,
                "bbox": plate_bbox,
                "confidence": plate_detection_confidence,
                "label": plate_detection_label,
                "sharpness": metrics["sharpness"],
                "crop_width": metrics["width"],
                "crop_height": metrics["height"],
                "aspect_ratio": metrics["aspect_ratio"],
                "quality": plate_crop_quality_score(plate_image, plate_detection_confidence),
            })
    elif overview_image is not None and plate_detector_available:
        for detected_plate in plate_detector.detect_many(overview_image)[:3]:
            detected_plate["quality"] = plate_crop_quality_score(
                detected_plate["crop"],
                detected_plate.get("confidence", 0.0),
            )
            plate_candidates.append(detected_plate)

    if plate_candidates:
        selected_plate = plate_candidates[0]
        if plate_ocr is not None:
            scored_plates = []
            for candidate in plate_candidates:
                candidate_text, candidate_confidence, candidate_variant = plate_ocr.recognize_best(candidate["crop"])
                candidate_valid = is_valid_plate_format(candidate_text)
                score = (
                    (1.0 if candidate_valid else 0.0) * 1.2
                    + format_score(candidate_text) * 0.8
                    + float(candidate_confidence) * 0.8
                    + float(candidate.get("quality", 0.0)) * 0.5
                )
                scored_plates.append((score, candidate, candidate_text, candidate_confidence, candidate_variant))
            _, selected_plate, plate_text, plate_confidence, plate_ocr_variant = max(scored_plates, key=lambda item: item[0])

        plate_image = selected_plate["crop"]
        plate_bbox = selected_plate.get("bbox")
        plate_detection_confidence = selected_plate.get("confidence", 0.0)
        plate_detection_label = selected_plate.get("label", "")
        plate_crop_sharpness = selected_plate.get("sharpness", plate_sharpness(plate_image))
        plate_crop_quality = selected_plate.get("quality", plate_crop_quality_score(plate_image, plate_detection_confidence))

    if overview_image is not None:
        vehicle_type, vehicle_color, vehicle_source = vehicle_color_detector.classify_for_plate(overview_image, plate_bbox)

    if plate_image is not None and settings.PLATE_COLOR_ENABLED:
        plate_color = detect_plate_color(plate_image)

    if plate_image is not None and plate_ocr is not None and not plate_text:
        plate_text, plate_confidence, plate_ocr_variant = plate_ocr.recognize_best(plate_image)

    plate_state = plate_state_code(plate_text)
    plate_text_valid = is_valid_plate_format(plate_text)

    result = {
        "vehicle_type": vehicle_type,
        "vehicle_color": vehicle_color,
        "vehicle_source": vehicle_source,
        "plate_color": plate_color,
        "plate_text": plate_text,
        "plate_confidence": plate_confidence,
        "plate_text_valid": plate_text_valid,
        "plate_state_code": plate_state,
        "plate_state_name": INDIAN_STATE_CODES.get(plate_state, "Bharat Series" if plate_state == "BH" else ""),
        "plate_bbox": plate_bbox,
        "plate_detection_confidence": plate_detection_confidence,
        "plate_detection_label": plate_detection_label,
        "plate_ocr_variant": plate_ocr_variant,
        "plate_crop_sharpness": plate_crop_sharpness,
        "plate_crop_quality": plate_crop_quality,
        "plate_candidate_count": len(plate_candidates),
        "plate_detector_available": plate_detector_available,
    }
    if include_plate_crop and plate_image is not None:
        result["plate_crop_image"] = plate_image
    return result


def process_frame(overview_image=None, plate_image=None, include_plate_crop=False):
    # Bounds how many camera threads run the heavy CPU models (plate detector,
    # OCR, vehicle color/type) at once. Without this, all camera worker
    # threads can hit these models simultaneously and oversubscribe the CPU,
    # which shows up as growing lag under load rather than a clean error.
    with inference_semaphore:
        return _process_frame_impl(
            overview_image=overview_image,
            plate_image=plate_image,
            include_plate_crop=include_plate_crop,
        )


def select_aggregated_result(results: list[dict]) -> dict:
    candidates = [
        result for result in results
        if result.get("plate_bbox") is not None and normalize_plate_text(result.get("plate_text", ""))
    ]
    if not candidates:
        detected_results = [result for result in results if result.get("plate_bbox") is not None]
        if detected_results:
            selected = dict(max(
                detected_results,
                key=lambda result: (
                    float(result.get("plate_crop_quality") or 0.0),
                    float(result.get("plate_detection_confidence") or 0.0),
                    float(result.get("plate_crop_sharpness") or 0.0),
                ),
            ))
        else:
            selected = dict(results[0]) if results else {}
        selected.update({
            "aggregation_enabled": True,
            "aggregation_window_size": len(results),
            "aggregation_candidate_count": 0,
            "aggregation_text_votes": 0,
            "aggregation_min_candidates_met": False,
            "aggregation_score": 0.0,
        })
        return selected

    groups = []
    for result in candidates:
        text = normalize_plate_text(result.get("plate_text", ""))
        for group in groups:
            if similar_plate_text(text, group["representative"]):
                group["results"].append(result)
                group["texts"].append(text)
                break
        else:
            groups.append({
                "representative": text,
                "results": [result],
                "texts": [text],
            })

    def result_score(result):
        return (
            float(result.get("plate_confidence") or 0.0),
            float(result.get("plate_crop_quality") or 0.0),
            min(float(result.get("plate_crop_sharpness") or 0.0) / 200.0, 1.0),
            format_score(result.get("plate_text", "")),
            1 if result.get("vehicle_type") != "Unknown" else 0,
            1 if result.get("vehicle_color") != "Unknown" else 0,
        )

    scored_groups = []
    for group in groups:
        group_results = group["results"]
        group_text = consensus_plate_text(group["texts"])
        group_count = len(group_results)
        avg_confidence = sum(float(result.get("plate_confidence") or 0.0) for result in group_results) / group_count
        avg_quality = sum(float(result.get("plate_crop_quality") or 0.0) for result in group_results) / group_count
        best_result = max(group_results, key=result_score)
        group_score = (
            group_count * 1.25
            + avg_confidence * 0.75
            + avg_quality * 0.45
            + format_score(group_text) * 0.85
            + min(float(best_result.get("plate_crop_sharpness") or 0.0) / 200.0, 1.0) * 0.30
        )
        scored_groups.append((group_score, group_text, best_result, avg_confidence, group_count))

    group_score, selected_text, best_result, avg_confidence, selected_count = max(scored_groups, key=lambda item: item[0])
    selected = dict(best_result)
    state_code = plate_state_code(selected_text)
    min_candidates_met = selected_count >= settings.PLATE_OCR_AGGREGATION_MIN_CANDIDATES

    selected.update({
        "plate_text": selected_text,
        "plate_frame_confidence": float(selected.get("plate_confidence") or 0.0),
        "plate_confidence": avg_confidence,
        "plate_text_valid": is_valid_plate_format(selected_text) and min_candidates_met,
        "plate_state_code": state_code,
        "plate_state_name": INDIAN_STATE_CODES.get(state_code, "Bharat Series" if state_code == "BH" else ""),
        "aggregation_enabled": True,
        "aggregation_window_size": len(results),
        "aggregation_candidate_count": len(candidates),
        "aggregation_cluster_count": len(groups),
        "aggregation_text_votes": selected_count,
        "aggregation_min_candidates_met": min_candidates_met,
        "aggregation_score": group_score,
    })
    return selected


def process_frame_batch(frames: list, include_plate_crop=False) -> dict:
    results = []
    for frame in frames:
        if frame is None or frame.size == 0:
            continue

        results.append(process_frame(overview_image=frame, include_plate_crop=include_plate_crop))

    if not results:
        return {}
    return select_aggregated_result(results)


def post_anpr_event(camera_name: str, lane_type: str, result: dict, snapshot_bytes: bytes):
    data = {
        "camera_name": camera_name,
        "lane_type": lane_type,
        "event_type": result.get("event_type", lane_type),
        "camera_group": result.get("camera_group", camera_name),
        "camera_role": result.get("camera_role", ""),
        "ramp_name": result.get("ramp_name", ""),
        "basement_name": result.get("basement_name", ""),
        "lane_name": result.get("lane_name", ""),
        "plate_text": result.get("plate_text", ""),
        "plate_confidence": str(result.get("plate_confidence", 0.0)),
        "plate_text_valid": str(result.get("plate_text_valid", False)).lower(),
        "plate_state_code": result.get("plate_state_code", ""),
        "plate_state_name": result.get("plate_state_name", ""),
        "plate_color": result.get("plate_color", "Unknown"),
        "vehicle_type": result.get("vehicle_type", "Unknown"),
        "vehicle_color": result.get("vehicle_color", "Unknown"),
        "vehicle_source": result.get("vehicle_source", ""),
        "plate_detection_confidence": str(result.get("plate_detection_confidence", 0.0)),
        "plate_detection_label": result.get("plate_detection_label", ""),
        "plate_crop_sharpness": str(result.get("plate_crop_sharpness", 0.0)),
        "plate_crop_quality": str(result.get("plate_crop_quality", 0.0)),
        "aggregation_text_votes": str(result.get("aggregation_text_votes", 0)),
        "aggregation_candidate_count": str(result.get("aggregation_candidate_count", result.get("plate_candidate_count", 0))),
        "aggregation_cluster_count": str(result.get("aggregation_cluster_count", 0)),
        "plate_ocr_fallback": str(result.get("plate_ocr_fallback", False)).lower(),
        "plate_ocr_fallback_reason": result.get("plate_ocr_fallback_reason", ""),
        "plate_ocr_variant": result.get("plate_ocr_variant", ""),
    }
    if result.get("plate_bbox") is not None:
        data["plate_bbox"] = ",".join(str(v) for v in result["plate_bbox"])

    headers = {}
    if settings.BACKEND_EVENT_TOKEN:
        headers["X-AI-Event-Token"] = settings.BACKEND_EVENT_TOKEN

    files = {"snapshot": (f"{camera_name}.jpg", snapshot_bytes, "image/jpeg")}
    response = requests.post(
        settings.BACKEND_EVENT_URL,
        data=data,
        files=files,
        headers=headers,
        timeout=settings.BACKEND_EVENT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


class CameraStatusRegistry:
    """Tracks whether each camera's worker is still running, or - for a
    non-looping local video source - has played through to the end. Read by
    /api/cameras/ (frontend polls this to show "Streaming completed") and by
    the stream endpoint (so a finished video isn't silently replayed from
    frame 0 on the next live-view request)."""

    def __init__(self):
        self._status = {}
        self._lock = threading.Lock()

    def set(self, camera_name: str, status: str):
        with self._lock:
            self._status[camera_name] = status

    def get(self, camera_name: str, default: str = "running") -> str:
        with self._lock:
            return self._status.get(camera_name, default)


camera_status_registry = CameraStatusRegistry()


class DuplicateFilter:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._seen = {}
        self._lock = threading.Lock()

    def should_skip(self, camera_name: str, plate_text: str):
        if not plate_text:
            return True

        key = (camera_name, plate_text)
        now = time.time()
        with self._lock:
            last_seen = self._seen.get(key)
            if last_seen and now - last_seen < self.seconds:
                return True
            self._seen[key] = now
        return False


def motion_prefilter_signature(frame):
    """Small grayscale copy of a frame, cheap to diff against the next one.
    No model involved - this is plain pixel comparison."""
    width = settings.VEHICLE_TRIGGER_MOTION_PREFILTER_WIDTH
    height = max(1, int(frame.shape[0] * (width / frame.shape[1])))
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def motion_prefilter_changed(prev_signature, curr_signature) -> bool:
    """True if enough pixels changed between two signatures to be worth
    running the real (model-based) vehicle detector. No previous signature
    (first poll) always counts as "changed" so we never skip the first
    check."""
    if prev_signature is None or prev_signature.shape != curr_signature.shape:
        return True
    diff = cv2.absdiff(prev_signature, curr_signature)
    changed_pixels = np.count_nonzero(diff > settings.VEHICLE_TRIGGER_MOTION_PREFILTER_PIXEL_THRESHOLD)
    changed_ratio = changed_pixels / diff.size
    return changed_ratio >= settings.VEHICLE_TRIGGER_MOTION_PREFILTER_MIN_CHANGE_RATIO


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """Standard ray-casting point-in-polygon test. `polygon` is a list of
    (x, y) points sharing one coordinate space with (x, y) -- this project
    uses fractions of frame width/height (0..1) throughout so a zone drawn
    once keeps working if snapshot_resolution ever changes."""
    if len(polygon) < 3:
        return True
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_at_y = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def polygon_bbox_px(polygon: list, width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel bounding box of a normalized-fraction polygon, clamped to the frame."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1 = max(0, int(min(xs) * width))
    y1 = max(0, int(min(ys) * height))
    x2 = min(width, int(max(xs) * width) + 1)
    y2 = min(height, int(max(ys) * height) + 1)
    return x1, y1, x2, y2


class VehicleTriggerGate:
    def __init__(self, camera_name: str, trigger_zone: list | None = None):
        self.camera_name = camera_name
        self.last_vehicle_seen = 0.0
        self.last_no_vehicle_log = 0.0
        self.last_motion_signature = None
        # Normalized-fraction (0..1) polygon marking the lane a vehicle must
        # actually be in for this camera to fire -- e.g. the plate-visible
        # patch of a ramp entry, not the whole field of view. None/too-few
        # points means "whole frame", the old (still supported) behaviour.
        self.trigger_zone = trigger_zone if trigger_zone and len(trigger_zone) >= 3 else None

    def _zone_crop(self, frame):
        """Crop to the trigger zone's bounding box so the motion pre-filter
        only ever looks at pixels near the lane -- movement elsewhere in the
        camera's view (people walking past, an adjacent lane, background
        flicker) can no longer count as "something changed here"."""
        if not self.trigger_zone:
            return frame
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = polygon_bbox_px(self.trigger_zone, width, height)
        cropped = frame[y1:y2, x1:x2]
        return cropped if cropped.size else frame

    def should_process(self, frame) -> bool:
        if not settings.VEHICLE_TRIGGER_ENABLED:
            return True

        now = time.time()
        if now - self.last_vehicle_seen <= settings.VEHICLE_TRIGGER_HOLD_SECONDS:
            return True

        # Cheap pixel-diff pre-check: with no physical presence sensor, this
        # is what lets us poll frequently without hitting the vehicle
        # detector model on every idle frame - an empty, unchanged lane
        # never reaches the model at all.
        if settings.VEHICLE_TRIGGER_MOTION_PREFILTER_ENABLED:
            signature = motion_prefilter_signature(self._zone_crop(frame))
            moved = motion_prefilter_changed(self.last_motion_signature, signature)
            self.last_motion_signature = signature
            if not moved:
                return False

        with inference_semaphore:
            candidates = vehicle_color_detector.vehicle_candidates(
                frame,
                confidence=settings.VEHICLE_TRIGGER_CONFIDENCE,
                min_area_ratio=settings.VEHICLE_TRIGGER_MIN_AREA_RATIO,
            )
        if self.trigger_zone:
            height, width = frame.shape[:2]
            candidates = [
                c for c in candidates
                if point_in_polygon(
                    ((c["bbox"][0] + c["bbox"][2]) / 2) / width,
                    ((c["bbox"][1] + c["bbox"][3]) / 2) / height,
                    self.trigger_zone,
                )
            ]
        if candidates:
            best = max(candidates, key=lambda item: item["confidence"] * item["area"])
            self.last_vehicle_seen = now
            print(
                f"[{self.camera_name}] vehicle_gate=vehicle "
                f"{best['label']} confidence={best['confidence']:.3f}"
            )
            return True

        if now - self.last_no_vehicle_log >= settings.VEHICLE_TRIGGER_NO_VEHICLE_LOG_INTERVAL_SECONDS:
            print(f"[{self.camera_name}] vehicle_gate=no_vehicle skip_anpr")
            self.last_no_vehicle_log = now
        return False


class WorkerHandle:
    def __init__(self, stop_event: threading.Event, threads: list[threading.Thread]):
        self.stop_event = stop_event
        self.threads = threads

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=5)


def handle_camera_frames(camera: dict, frames: list, duplicate_filter: DuplicateFilter) -> None:
    if not frames:
        return

    camera_name = camera["name"]
    fallback_snapshot_image = next((frame for frame in reversed(frames) if frame is not None and frame.size > 0), None)
    if settings.PLATE_OCR_AGGREGATION_ENABLED:
        result = process_frame_batch(frames, include_plate_crop=True)
    else:
        result = process_frame(overview_image=frames[0], include_plate_crop=True)

    plate_crop_image = result.pop("plate_crop_image", None)
    snapshot_image = plate_crop_image
    result.update({
        "event_type": camera.get("event_type", camera.get("lane_type", "")),
        "lane_type": camera.get("lane_type", ""),
        "camera_group": camera.get("camera_group", camera_name),
        "camera_role": camera.get("camera_role", ""),
        "ramp_name": camera.get("ramp_name", ""),
        "basement_name": camera.get("basement_name", ""),
        "lane_name": camera.get("lane_name", ""),
    })

    plate_text = result.get("plate_text", "")
    plate_valid = bool(result.get("plate_text_valid", False))
    plate_confidence = float(result.get("plate_confidence") or 0.0)
    should_post = plate_valid and plate_confidence >= settings.PLATE_OCR_MIN_CONFIDENCE
    fallback_reason = ""
    has_detected_plate = plate_crop_image is not None and result.get("plate_bbox") is not None
    if not should_post and settings.PLATE_OCR_POST_UNCERTAIN:
        fallback_text = normalize_plate_text(plate_text)
        if has_detected_plate and fallback_text and plate_confidence >= settings.PLATE_OCR_FALLBACK_MIN_CONFIDENCE:
            result["plate_text"] = fallback_text
            result["plate_text_valid"] = False
            result["plate_ocr_fallback"] = True
            result["plate_ocr_fallback_reason"] = "uncertain-ocr"
            fallback_reason = "uncertain-ocr"
            should_post = True
            snapshot_image = plate_crop_image

    if not should_post and settings.PLATE_OCR_POST_UNREADABLE:
            result["plate_text"] = unreadable_plate_marker()
            result["plate_confidence"] = max(
                float(result.get("plate_detection_confidence") or 0.0),
                float(result.get("plate_crop_quality") or 0.0),
            )
            result["plate_text_valid"] = False
            result["plate_ocr_fallback"] = True
            result["plate_ocr_fallback_reason"] = "unreadable-ocr"
            fallback_reason = "unreadable-ocr"
            should_post = True
            snapshot_image = plate_crop_image if has_detected_plate else fallback_snapshot_image

    if not should_post:
        print(
            f"[{camera_name}] skipped plate={plate_text or '-'} "
            f"valid={plate_valid} confidence={plate_confidence:.3f} "
            f"votes={result.get('aggregation_text_votes', 0)} "
            f"candidates={result.get('aggregation_candidate_count', result.get('plate_candidate_count', 0))} "
            f"quality={float(result.get('plate_crop_quality') or 0.0):.3f}"
        )
        return
    plate_text = result.get("plate_text", "")
    if duplicate_filter.should_skip(camera_name, plate_text):
        return

    if snapshot_image is None:
        print(f"[{camera_name}] skipped plate={plate_text} reason=no_snapshot")
        return
    snapshot_bytes = encode_jpeg(snapshot_image)
    if snapshot_bytes is None:
        return

    if settings.STORAGE_ENABLED:
        save_capture_image(
            settings.PLATE_IMAGE_DIR,
            camera_name,
            plate_crop_image,
            suffix=normalize_plate_text(plate_text),
        )
        save_capture_image(
            settings.VEHICLE_IMAGE_DIR,
            camera_name,
            fallback_snapshot_image,
        )

    saved = post_anpr_event(camera_name, result["lane_type"], result, snapshot_bytes)
    selected = "selected" if saved.get("selected", True) else "merged-lower-confidence"
    fallback = f" fallback={fallback_reason}" if fallback_reason else ""
    print(f"[{camera_name}] sent {plate_text} -> backend id={saved.get('id')} {selected}{fallback}")


def handle_camera_frame(camera: dict, frame, duplicate_filter: DuplicateFilter) -> None:
    handle_camera_frames(camera, [frame], duplicate_filter)


def collect_ip_frame_burst(camera_name: str, first_frame, stop_event: threading.Event) -> list:
    frames = [first_frame] if first_frame is not None and first_frame.size > 0 else []
    if not settings.PLATE_OCR_AGGREGATION_ENABLED:
        return frames

    window_size = max(1, settings.PLATE_OCR_AGGREGATION_WINDOW_SIZE)
    delay = max(0.0, settings.PLATE_OCR_AGGREGATION_FRAME_DELAY_SECONDS)
    while len(frames) < window_size and not stop_event.is_set():
        if delay:
            stop_event.wait(delay)
        if stop_event.is_set():
            break

        snapshot = get_snapshot(camera_name)
        if not snapshot:
            continue
        frame = decode_image(snapshot)
        if frame is not None and frame.size > 0:
            frames.append(frame)
    return frames


def collect_video_frame_burst(capture, first_frame, loop: bool, stop_event: threading.Event) -> list:
    frames = [first_frame] if first_frame is not None and first_frame.size > 0 else []
    if not settings.PLATE_OCR_AGGREGATION_ENABLED:
        return frames

    window_size = max(1, settings.PLATE_OCR_AGGREGATION_WINDOW_SIZE)
    while len(frames) < window_size and not stop_event.is_set():
        frame = read_video_frame(capture, loop=loop)
        if frame is None:
            break
        frames.append(frame)
    return frames


def plate_frame_visibility_score(frame) -> float:
    if frame is None or frame.size == 0 or not (plate_detector and plate_detector.available):
        return 0.0

    detections = plate_detector.detect_many(frame)
    if not detections:
        return 0.0

    height, width = frame.shape[:2]
    best_score = 0.0
    for plate in detections[:3]:
        quality = plate_crop_quality_score(plate.get("crop"), plate.get("confidence", 0.0))
        x1, y1, x2, y2 = [int(value) for value in plate.get("bbox", [0, 0, 0, 0])]
        edge_clearance = min(x1, y1, width - x2, height - y2)
        edge_score = 0.65 if edge_clearance <= 2 else 1.0
        best_score = max(best_score, quality * edge_score)
    return best_score


def add_best_track_frame(track_entries: list, frame, sequence: int) -> None:
    score = plate_frame_visibility_score(frame)
    track_entries.append({
        "frame": frame,
        "score": score,
        "sequence": sequence,
    })
    # Keep only the best plate-visible frames. When no plate is detected yet
    # (score=0), newer frames win so the fallback still follows the vehicle.
    track_entries.sort(key=lambda entry: (entry["score"], entry["sequence"]), reverse=True)
    del track_entries[settings.PLATE_TRACK_MAX_FRAMES:]


def selected_track_frames(track_entries: list) -> list:
    limit = max(1, settings.PLATE_OCR_AGGREGATION_WINDOW_SIZE)
    return [entry["frame"] for entry in sorted(
        track_entries,
        key=lambda entry: (entry["score"], entry["sequence"]),
        reverse=True,
    )[:limit]]


def ip_camera_loop(
    camera: dict,
    duplicate_filter: DuplicateFilter,
    vehicle_gate: VehicleTriggerGate,
    stop_event: threading.Event,
):
    camera_name = camera["name"]
    idle_interval = settings.CAMERA_SNAPSHOT_INTERVAL_SECONDS
    tracking_interval = max(0.05, settings.PLATE_OCR_AGGREGATION_FRAME_DELAY_SECONDS)
    track_frames: list = []
    frame_sequence = 0
    track_finalized = False
    print(f"[{camera_name}] started event={camera.get('event_type')} ip={camera['ip']}")

    def add_track_frame(frame) -> None:
        nonlocal frame_sequence
        frame_sequence += 1
        add_best_track_frame(track_frames, frame, frame_sequence)

    def finalize_track() -> None:
        nonlocal track_frames, track_finalized
        if track_frames:
            try:
                handle_camera_frames(camera, selected_track_frames(track_frames), duplicate_filter)
            except requests.RequestException as exc:
                print(f"[{camera_name}] HTTP error: {exc}")
            except Exception as exc:
                print(f"[{camera_name}] worker error: {exc}")
        track_frames = []
        track_finalized = True

    def reset_track() -> None:
        nonlocal track_frames, frame_sequence, track_finalized
        track_frames = []
        frame_sequence = 0
        track_finalized = False

    try:
        while not stop_event.is_set():
            wait_interval = idle_interval
            try:
                snapshot = get_snapshot(camera_name)
                if not snapshot:
                    wait_interval = tracking_interval if track_frames else idle_interval
                    continue

                frame = decode_image(snapshot)
                if frame is not None and frame.size > 0 and vehicle_gate.should_process(frame):
                    # Score only the first track_max_frames samples for this vehicle,
                    # then OCR the best aggregation window from that small pool.
                    if not track_finalized:
                        add_track_frame(frame)
                        if frame_sequence >= settings.PLATE_TRACK_MAX_FRAMES:
                            finalize_track()
                    wait_interval = tracking_interval
                else:
                    if not track_finalized:
                        finalize_track()
                    reset_track()
            except CameraUnavailable as exc:
                print(f"[{camera_name}] camera unavailable: {exc}")
            except requests.RequestException as exc:
                print(f"[{camera_name}] HTTP error: {exc}")
            except Exception as exc:
                print(f"[{camera_name}] worker error: {exc}")
            finally:
                stop_event.wait(wait_interval)
    finally:
        if not track_finalized:
            finalize_track()


def video_camera_loop(
    camera: dict,
    duplicate_filter: DuplicateFilter,
    vehicle_gate: VehicleTriggerGate,
    stop_event: threading.Event,
):
    camera_name = camera["name"]
    path = video_path(camera)
    realtime = bool(camera.get("video_realtime", False))

    try:
        capture = open_video_capture(camera)
    except CameraUnavailable as exc:
        print(f"[{camera_name}] video unavailable: {exc}")
        return

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_delay = 1.0 / max(source_fps, 1.0)
    process_every = int(camera.get("process_every_n_frames") or max(1, round(source_fps * settings.CAMERA_SNAPSHOT_INTERVAL_SECONDS)))
    loop = bool(camera.get("video_loop", True))
    frame_index = 0
    track_sequence = 0
    track_frames: list = []
    track_finalized = False

    def finalize_track():
        nonlocal track_frames, track_finalized
        if track_frames:
            try:
                handle_camera_frames(camera, selected_track_frames(track_frames), duplicate_filter)
            except requests.RequestException as exc:
                print(f"[{camera_name}] HTTP error: {exc}")
            except Exception as exc:
                print(f"[{camera_name}] worker error: {exc}")
        track_frames = []
        track_finalized = True

    def reset_track():
        nonlocal track_frames, track_sequence, track_finalized
        track_frames = []
        track_sequence = 0
        track_finalized = False

    print(f"[{camera_name}] started event={camera.get('event_type')} video={path}")
    camera_status_registry.set(camera_name, "running")
    ended_naturally = False
    try:
        while not stop_event.is_set():
            iteration_started = time.time()
            frame = read_video_frame(capture, loop=loop)
            if frame is None:
                ended_naturally = True
                break

            # While a vehicle track is open we process every frame (dense sampling across
            # its full dwell time); otherwise we only sample every `process_every` frames.
            if track_frames or track_finalized or frame_index % process_every == 0:
                try:
                    if vehicle_gate.should_process(frame):
                        if not track_finalized:
                            track_sequence += 1
                            add_best_track_frame(track_frames, frame, track_sequence)
                            if track_sequence >= settings.PLATE_TRACK_MAX_FRAMES:
                                finalize_track()
                    else:
                        if not track_finalized:
                            finalize_track()
                        reset_track()
                except requests.RequestException as exc:
                    print(f"[{camera_name}] HTTP error: {exc}")
                except Exception as exc:
                    print(f"[{camera_name}] worker error: {exc}")

            frame_index += 1
            if realtime:
                # Only throttle when we're ahead of the video's natural pace. Never skip
                # frames to "catch up" when processing falls behind -- OCR/detection is
                # CPU-heavy and dropping frames to stay in sync means silently missing
                # vehicles that pass through those skipped frames.
                elapsed = time.time() - iteration_started
                remaining = frame_delay - elapsed
                if remaining > 0:
                    stop_event.wait(remaining)
    finally:
        if not track_finalized:
            finalize_track()
        capture.release()
        if ended_naturally and not loop:
            print(f"[{camera_name}] video playback completed - one full pass done, not looping")
            camera_status_registry.set(camera_name, "completed")
        elif stop_event.is_set():
            camera_status_registry.set(camera_name, "stopped")


def camera_loop(
    camera: dict,
    duplicate_filter: DuplicateFilter,
    vehicle_gate: VehicleTriggerGate,
    stop_event: threading.Event,
):
    if is_video_source(camera):
        video_camera_loop(camera, duplicate_filter, vehicle_gate, stop_event)
        return
    ip_camera_loop(camera, duplicate_filter, vehicle_gate, stop_event)


def start_camera_workers() -> WorkerHandle | None:
    cameras = list(enabled_cameras())
    if not cameras:
        print("No enabled cameras found.")
        return None

    if not os.path.exists(settings.PLATE_DETECTOR_MODEL_PATH):
        raise RuntimeError(
            f"Plate detector model missing: {settings.PLATE_DETECTOR_MODEL_PATH}. "
            "Place the trained YOLO plate detector at that path before running cameras."
        )

    duplicate_filter = DuplicateFilter(settings.CAMERA_DUPLICATE_SECONDS)
    stop_event = threading.Event()
    threads = []
    for camera in cameras:
        vehicle_gate = VehicleTriggerGate(camera["name"], trigger_zone=camera.get("trigger_zone"))
        thread = threading.Thread(
            target=camera_loop,
            args=(camera, duplicate_filter, vehicle_gate, stop_event),
            daemon=True,
            name=f"anpr-camera-{camera['name']}",
        )
        thread.start()
        threads.append(thread)

    print(f"Running {len(threads)} camera worker(s). Press Ctrl+C to stop.")
    return WorkerHandle(stop_event, threads)


_worker_handle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker_handle
    if settings.CAMERA_AUTO_START_WORKERS:
        _worker_handle = start_camera_workers()
    try:
        yield
    finally:
        if _worker_handle is not None:
            _worker_handle.stop()


app = FastAPI(title="ANPR AI Classification Service", lifespan=lifespan)


@app.post("/classify")
async def classify(
    overview_image: Optional[UploadFile] = File(None),
    plate_image: Optional[UploadFile] = File(None),
    plate_text_poly: Optional[str] = Form(None),
):
    del plate_text_poly
    decoded_overview = decode_image(await overview_image.read()) if overview_image is not None else None
    decoded_plate = decode_image(await plate_image.read()) if plate_image is not None else None
    return process_frame(overview_image=decoded_overview, plate_image=decoded_plate)


@app.get("/api/cameras/")
def list_cameras():
    return [
        {
            "name": cam["name"],
            "ip": cam["ip"],
            "source_type": camera_source_type(cam),
            "process_enabled": cam.get("process_enabled", True),
            "event_type": cam.get("event_type", ""),
            "lane_type": cam.get("lane_type", ""),
            "ramp_name": cam.get("ramp_name", ""),
            "basement_name": cam.get("basement_name", ""),
            "lane_name": cam.get("lane_name", ""),
            "camera_group": cam.get("camera_group", cam["name"]),
            "camera_role": cam.get("camera_role", ""),
            "enabled": cam.get("enabled", False),
            "status": camera_status_registry.get(cam["name"], "running" if cam.get("enabled", False) else "stopped"),
        }
        for cam in settings.CAMERAS
    ]


@app.get("/api/streams/{name}")
def camera_stream(name: str):
    if camera_status_registry.get(name) == "completed":
        raise HTTPException(status_code=410, detail="Video playback completed - this source will not loop.")
    try:
        result = open_mjpeg_stream(name)
    except CameraUnavailable as exc:
        raise HTTPException(status_code=502, detail=f"Camera did not respond: {exc}") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown camera.")
    media_type, generator = result
    return StreamingResponse(generator, media_type=media_type)


@app.get("/api/cameras/{name}/snapshot")
def camera_snapshot(name: str):
    try:
        image = get_snapshot(name)
    except CameraUnavailable as exc:
        raise HTTPException(status_code=502, detail=f"Camera did not respond: {exc}") from exc
    if image is None:
        raise HTTPException(status_code=404, detail="Unknown camera.")
    return Response(content=image, media_type="image/jpeg")


@app.get("/api/cameras/{name}/audio")
def camera_audio(name: str):
    try:
        result = open_audio_stream(name)
    except CameraUnavailable as exc:
        raise HTTPException(status_code=502, detail=f"Camera audio did not respond: {exc}") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown camera.")
    media_type, generator = result
    return StreamingResponse(generator, media_type=media_type)


@app.post("/api/cameras/{name}/ptz")
async def camera_ptz(name: str, payload: dict):
    try:
        ptz_move(name, rpan=payload.get("rpan", 0), rtilt=payload.get("rtilt", 0), rzoom=payload.get("rzoom", 0))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CameraUnavailable as exc:
        raise HTTPException(status_code=502, detail=f"Camera did not respond: {exc}") from exc
    return {"ok": True}


@app.post("/api/cameras/{name}/ptz/continuous")
async def camera_ptz_continuous(name: str, payload: dict):
    try:
        ptz_continuous_move(name, pan=payload.get("pan", 0), tilt=payload.get("tilt", 0))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except CameraUnavailable as exc:
        raise HTTPException(status_code=502, detail=f"Camera did not respond: {exc}") from exc
    return {"ok": True}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "plate_detector_available": bool(plate_detector and plate_detector.available),
        "plate_detector_model": settings.PLATE_DETECTOR_MODEL_PATH,
        "plate_ocr_enabled": settings.PLATE_OCR_ENABLED,
        "plate_ocr_aggregation_enabled": settings.PLATE_OCR_AGGREGATION_ENABLED,
        "plate_ocr_aggregation_window_size": settings.PLATE_OCR_AGGREGATION_WINDOW_SIZE,
        "plate_ocr_aggregation_min_candidates": settings.PLATE_OCR_AGGREGATION_MIN_CANDIDATES,
        "runtime_opencv_threads": settings.RUNTIME_OPENCV_THREADS,
        "runtime_paddle_cpu_threads": settings.RUNTIME_PADDLE_CPU_THREADS,
        "runtime_torch_threads": settings.RUNTIME_TORCH_THREADS,
        "runtime_torch_interop_threads": settings.RUNTIME_TORCH_INTEROP_THREADS,
        "vehicle_trigger_enabled": settings.VEHICLE_TRIGGER_ENABLED,
        "vehicle_trigger_confidence": settings.VEHICLE_TRIGGER_CONFIDENCE,
        "vehicle_trigger_min_area_ratio": settings.VEHICLE_TRIGGER_MIN_AREA_RATIO,
        "vehicle_trigger_hold_seconds": settings.VEHICLE_TRIGGER_HOLD_SECONDS,
        "camera_count": len(settings.CAMERAS),
        "stream_camera_count": sum(1 for camera in settings.CAMERAS if camera.get("enabled", False)),
        "processing_camera_count": len(list(enabled_cameras())),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.SERVICE_HOST, port=settings.SERVICE_PORT)
