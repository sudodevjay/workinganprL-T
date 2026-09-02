"""Run the actual production tracking pipeline (VehicleTriggerGate + per-track frame
buffering + process_frame_batch aggregation) against a real video file and report what
the AI/OCR pipeline itself detected and read -- no ground-truth file involved.

This intentionally mirrors main.video_camera_loop's tracking logic (dense sampling
while a vehicle is present, finalize when it's gone) but skips the backend HTTP POST
so it can run standalone against a video file.
"""
import json
import os
import sys
import time

import cv2

import main
from config import settings


def run(video_path: str, camera_name: str = "real_video_test", max_frames: int = None,
        start_frame: int = 0, track_offset: int = 0, jsonl_path: str = None):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    process_every = max(1, round(source_fps * settings.CAMERA_SNAPSHOT_INTERVAL_SECONDS))
    vehicle_gate = main.VehicleTriggerGate(camera_name)

    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"video={video_path} fps={source_fps} total_frames={total_frames} "
          f"process_every={process_every} track_max_frames={settings.PLATE_TRACK_MAX_FRAMES} "
          f"start_frame={start_frame} track_offset={track_offset}")

    track_frames = []
    track_start_frame = None
    frame_index = start_frame
    tracks = []
    started = time.time()
    jsonl_file = open(jsonl_path, "a", encoding="utf-8") if jsonl_path else None

    def finalize(end_frame):
        nonlocal track_frames, track_start_frame
        if not track_frames:
            return
        result = main.process_frame_batch(track_frames, include_plate_crop=False)
        plate_text = result.get("plate_text", "")
        plate_valid = bool(result.get("plate_text_valid", False))
        plate_confidence = float(result.get("plate_confidence") or 0.0)
        accepted = plate_valid and plate_confidence >= settings.PLATE_OCR_MIN_CONFIDENCE
        track = {
            "track_index": track_offset + len(tracks) + 1,
            "start_frame": track_start_frame,
            "end_frame": end_frame,
            "start_seconds": round(track_start_frame / source_fps, 2),
            "end_seconds": round(end_frame / source_fps, 2),
            "frames_buffered": len(track_frames),
            "plate_text": plate_text,
            "plate_text_valid": plate_valid,
            "plate_confidence": round(plate_confidence, 4),
            "accepted": accepted,
            "vehicle_type": result.get("vehicle_type", "Unknown"),
            "vehicle_color": result.get("vehicle_color", "Unknown"),
            "plate_color": result.get("plate_color", "Unknown"),
            "plate_state_code": result.get("plate_state_code", ""),
            "plate_detection_confidence": round(float(result.get("plate_detection_confidence") or 0.0), 4),
            "aggregation_text_votes": result.get("aggregation_text_votes", 0),
            "aggregation_candidate_count": result.get("aggregation_candidate_count", 0),
        }
        tracks.append(track)
        if jsonl_file:
            jsonl_file.write(json.dumps(track, ensure_ascii=False) + "\n")
            jsonl_file.flush()
        print(
            f"[track {track['track_index']:03d}] "
            f"t={track['start_seconds']}-{track['end_seconds']}s "
            f"frames={track['frames_buffered']} "
            f"plate='{plate_text}' valid={plate_valid} conf={plate_confidence:.3f} "
            f"accepted={accepted} type={track['vehicle_type']} color={track['vehicle_color']} "
            f"plate_color={track['plate_color']}",
            flush=True,
        )
        track_frames = []
        track_start_frame = None

    while True:
        if max_frames is not None and frame_index >= max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break

        if track_frames or frame_index % process_every == 0:
            present = vehicle_gate.should_process(frame)
            if present:
                if not track_frames:
                    track_start_frame = frame_index
                track_frames.append(frame)
                if len(track_frames) >= settings.PLATE_TRACK_MAX_FRAMES:
                    finalize(frame_index)
            else:
                finalize(frame_index)

        frame_index += 1
        if frame_index % 300 == 0:
            elapsed = time.time() - started
            print(f"... {frame_index}/{total_frames} frames, {len(tracks)} tracks so far, "
                  f"{elapsed:.1f}s elapsed", flush=True)

    finalize(frame_index)
    capture.release()
    if jsonl_file:
        jsonl_file.close()
    return tracks


def summarize(tracks):
    total = len(tracks)
    read = [t for t in tracks if main.normalize_plate_text(t["plate_text"])]
    missed = total - len(read)
    valid_format = [t for t in read if t["plate_text_valid"]]
    accepted = [t for t in read if t["accepted"]]
    avg_conf = sum(t["plate_confidence"] for t in read) / len(read) if read else 0.0

    type_known = [t for t in tracks if t["vehicle_type"] != "Unknown"]
    color_known = [t for t in tracks if t["vehicle_color"] != "Unknown"]
    plate_color_known = [t for t in tracks if t["plate_color"] != "Unknown"]

    return {
        "total_tracks_detected": total,
        "read_nonempty_text": len(read),
        "missed_no_text": missed,
        "read_rate_pct": round(len(read) / total * 100, 1) if total else 0.0,
        "valid_format_count": len(valid_format),
        "valid_format_rate_among_read_pct": round(len(valid_format) / len(read) * 100, 1) if read else 0.0,
        "accepted_would_post_count": len(accepted),
        "average_confidence_among_read": round(avg_conf, 4),
        "vehicle_type_known_count": len(type_known),
        "vehicle_color_known_count": len(color_known),
        "plate_color_known_count": len(plate_color_known),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default=os.path.join("..", "test_videos", "testplate.mp4"))
    parser.add_argument("max_frames", nargs="?", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--track-offset", type=int, default=0)
    parser.add_argument("--jsonl", default=None, help="Append each finalized track as one JSON line to this file (for resumable runs)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    video_path = main.resolve_local_path(args.video_path)

    tracks = run(
        video_path,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        track_offset=args.track_offset,
        jsonl_path=args.jsonl,
    )
    summary = summarize(tracks)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("test_outputs", f"real_video_test_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tracks.json"), "w", encoding="utf-8") as f:
        json.dump(tracks, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== SUMMARY (this run only) ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved: {out_dir}/tracks.json, {out_dir}/summary.json")
