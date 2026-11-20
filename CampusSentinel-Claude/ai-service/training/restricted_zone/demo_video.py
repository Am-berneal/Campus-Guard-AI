"""Demo: detect + track people in a video, check zones and time rules, save results.

Outputs in ai-service/runs/restricted_zone/<video-name>_<run-time>/
    annotated.mp4      boxes, track IDs, zones (red = restricted now, green = allowed)
    incidents.json     POTENTIAL incidents (require administrator review)
    evidence/*.jpg     annotated frame at the moment each incident was raised

Timestamps: a recorded video has no reliable wall-clock time, so frame time is
    --start-time  +  frame_index / fps
interpreted in the campus time zone. Use --start-time to test night/day rules.
For a webcam (--video 0) the real current time is used.

Run from the ai-service folder:
    python -m training.restricted_zone.demo_video --video path\\to\\clip.mp4 --start-time "2026-11-16 21:30"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from training.restricted_zone.config import AI_SERVICE_ROOT, ConfigError, load_config
from training.restricted_zone.detector import PersonDetector
from training.restricted_zone.rules import RuleEngine

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "zones.sample.yaml"
RED, GREEN, WHITE, YELLOW = (0, 0, 255), (0, 170, 0), (255, 255, 255), (0, 220, 255)


def draw(frame, cfg, result, detections, when_local):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for zone in cfg.zones:
        color = RED if result.zone_restricted[zone.zone_id] else GREEN
        pts = zone.polygon_pixels(w, h)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)
        label = f"{zone.name}: {'RESTRICTED' if result.zone_restricted[zone.zone_id] else 'allowed'}"
        x, y = pts[:, 0].min(), max(pts[:, 1].min() - 8, 15)
        cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)  # translucent zone fill

    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        violating = det.track_id in result.violating
        color = RED if violating else YELLOW
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1)  # ground point used for zone checks
        tid = "?" if det.track_id is None else det.track_id
        cv2.putText(frame, f"ID {tid} {det.confidence:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.rectangle(frame, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(frame, f"{when_local:%a %Y-%m-%d %H:%M:%S} ({cfg.timezone_name})  |  POTENTIAL incidents need review",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return frame


def main() -> int:
    p = argparse.ArgumentParser(description="Restricted-zone demo on a video file or webcam.")
    p.add_argument("--video", required=True, help="video file path, or 0 for the default webcam")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--start-time", help='campus-local start time of the video, e.g. "2026-11-16 21:30"')
    p.add_argument("--out-dir", type=Path, default=AI_SERVICE_ROOT / "runs" / "restricted_zone")
    p.add_argument("--show", action="store_true", help="show a live window (press q to stop)")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = whole video)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], help="override config detector.device")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1

    webcam = args.video.isdigit()
    if not webcam and not Path(args.video).is_file():
        print(f"ERROR: video not found: {args.video}")
        return 1
    if args.start_time:
        try:
            start = datetime.fromisoformat(args.start_time).replace(tzinfo=cfg.timezone)
        except ValueError:
            print('ERROR: --start-time must look like "2026-11-16 21:30"')
            return 1
    else:
        start = datetime.now(cfg.timezone)

    cap = cv2.VideoCapture(int(args.video) if webcam else args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video source {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 1 or fps > 240:
        fps = 25.0  # some files/webcams report 0; assume 25 fps for timestamps
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    name = "webcam" if webcam else Path(args.video).stem
    run_dir = args.out_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "evidence").mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(run_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    det_cfg = cfg.detector
    print(f"Loading person detector {det_cfg.weights.name} (pretrained, Ultralytics) ...")
    detector = PersonDetector(det_cfg.weights, det_cfg.confidence, det_cfg.iou, det_cfg.image_size,
                              args.device or det_cfg.device, det_cfg.half, cfg.tracker_config)
    print(f"Device: {detector.device} (half precision: {detector.half}) | video {w}x{h} @ {fps:.1f} fps")
    print(f"Video start (campus time): {start:%a %Y-%m-%d %H:%M:%S %Z}")

    engine = RuleEngine(cfg)
    incidents, frame_idx, t0 = [], 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            when = datetime.now(cfg.timezone) if webcam else start + timedelta(seconds=frame_idx / fps)
            detections = detector.track(frame)
            result = engine.update(detections, w, h, when)
            annotated = draw(frame, cfg, result, detections, when.astimezone(cfg.timezone))

            for inc in result.incidents:
                if cfg.incidents.save_evidence_frames:
                    ev = run_dir / "evidence" / f"{inc['incident_id']}.jpg"
                    cv2.imwrite(str(ev), annotated)
                    inc["evidence_frame"] = str(ev.relative_to(AI_SERVICE_ROOT)).replace("\\", "/")
                inc["frame_index"] = frame_idx
                incidents.append(inc)
                print(f"  POTENTIAL INCIDENT  frame {frame_idx}  {inc['zone_name']}  track {inc['track_id']}  "
                      f"{inc['timestamp_local']}  conf {inc['confidence']:.2f}")

            writer.write(annotated)
            if args.show:
                cv2.imshow("CampusSentinel restricted zones (q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        detector.close()

    elapsed = time.perf_counter() - t0
    (run_dir / "incidents.json").write_text(json.dumps(incidents, indent=2), encoding="utf-8")
    print(f"\nFrames processed: {frame_idx} in {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps)")
    print(f"Potential incidents: {len(incidents)} (administrator review required)")
    print(f"Output: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
