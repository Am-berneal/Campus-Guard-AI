"""After-hours person detection (stored with the existing `restricted_zone` incident type).

Rule (official, Asia/Kolkata, closing time inclusive):
    COS 20:00 · Outside Library 22:30 · B Block 22:30 · Kravings 22:30

    if person_detected and frame_recording_time >= closing_time:
        potential incident

- The person may be ANYWHERE in the frame. Zone polygons, feet position and dwell time are
  NOT used (the polygons stay in the camera YAML for possible later use, but are ignored here).
- One person detection (YOLO person class, confidence ≥ the camera's `detector.confidence`)
  in one sampled frame after closing is sufficient.
- Detection uses the existing PersonDetector in plain detection mode, so tracking can never
  suppress a valid detection. Duplicates are prevented by grouping: consecutive after-hours
  frames with a person (gaps ≤ EVENT_GAP_S) form ONE event with a deterministic
  event_id, and the backend upserts by that id.
- Frame time = recording start (`recorded_at`) + frame_index / fps. Upload/analysis time is
  never used. Missing or future recording time → status "time_unknown", no incidents.

This reports after-hours PERSON PRESENCE in the camera view. It does not establish that
anyone entered a particular restricted area, and every event needs administrator review.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2

from inference.settings import Settings
from inference.video_io import VideoInfo, group_times, iter_frames, label, save_jpeg
from training.restricted_zone.config import ConfigError, closing_time_schedule, load_config
from training.restricted_zone.schedule import parse_schedule

LOG = logging.getLogger("inference.after_hours")
FUTURE_TOLERANCE = timedelta(minutes=10)   # a recording cannot start in the future
MAX_EVIDENCE = 3



EVENT_GAP_S = 10.0   # person frames closer than this (seconds) form ONE potential incident

def zone_config_path(settings: Settings, camera_id: str) -> Path:
    return settings.zone_config_dir / f"{camera_id}.yaml"


def clock_12h(dt_or_hhmm) -> str:
    """'20:00' or a datetime → '8:00 PM'."""
    if isinstance(dt_or_hhmm, str):
        h, m = (int(v) for v in dt_or_hhmm.split(":"))
        if h == 24:
            return "12:00 AM (midnight)"
        dt_or_hhmm = datetime(2000, 1, 1, h, m)
    return dt_or_hhmm.strftime("%I:%M %p").lstrip("0")


def normalize_recorded_at(recorded_at: datetime | None, tz: ZoneInfo,
                          now: datetime | None = None) -> tuple[datetime | None, str]:
    """Return (recording start in campus time, explanation) or (None, reason it is unusable).

    - timezone-aware value → converted to campus time (any offset/UTC is handled correctly)
    - value WITHOUT offset → interpreted as campus time, and the result says so explicitly
    - missing or in the future → None: the time rule cannot be verified
    """
    if recorded_at is None:
        return None, "no recording time was provided"
    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        local = recorded_at.replace(tzinfo=tz)
        note = f"recording time had no UTC offset; interpreted as campus time ({tz.key})"
    else:
        local = recorded_at.astimezone(tz)
        note = f"recording time converted from UTC{recorded_at.strftime('%z')} to {tz.key}"
    now = now or datetime.now(timezone.utc)
    if local - now > FUTURE_TOLERANCE:
        return None, f"recording time {local.isoformat(timespec='seconds')} is in the future"
    return local, note


def _evidence_image(frame, dets, local: datetime, closing_display: str):
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"person {d.confidence:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
    return label(img, f"AFTER-HOURS PERSON  {clock_12h(local)} IST {local:%d %b %Y}  (closing {closing_display})")


def analyze(video: Path, info: VideoInfo, detector, camera_id: str, settings: Settings, evidence_dir: Path,
            recorded_at: datetime | None) -> dict:
    cfg_path = zone_config_path(settings, camera_id)
    if not cfg_path.is_file():
        return {"available": False, "status": "not_configured",
                "reason": f"no configuration for camera '{camera_id}' ({cfg_path.name})",
                "potential_incident": False, "events": []}
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        raise ValueError(f"invalid camera configuration {cfg_path.name}: {exc}") from exc
    if not cfg.closing_time:
        return {"available": False, "status": "not_configured",
                "reason": f"no closing_time set for camera '{camera_id}' ({cfg_path.name})",
                "potential_incident": False, "events": []}

    closing_display = clock_12h(cfg.closing_time)
    rule = {"mode": "after_hours_person_presence", "closing_time": cfg.closing_time,
            "restricted_until": cfg.restricted_until, "closing_time_display": closing_display,
            "campus_timezone": cfg.timezone_name, "person_confidence_threshold": cfg.detector.confidence}

    start, time_note = normalize_recorded_at(recorded_at, cfg.timezone)
    if start is None:
        # Safe fallback: without a trustworthy recording time the rule cannot be checked.
        return {"available": True, "status": "time_unknown",
                "reason": (f"After-hours check not evaluated: {time_note}. Provide the recording start time "
                           f"(campus time, {cfg.timezone_name}) to check for persons after {closing_display}."),
                "potential_incident": False, "events": [], "rule": rule}

    # Restricted period = [closing_time, restricted_until) every day, via the existing schedule logic.
    period = parse_schedule(closing_time_schedule(cfg.closing_time, cfg.restricted_until))
    tz = cfg.timezone
    step = max(1, round(info.fps / settings.zone_sample_fps))
    sampled = list(range(0, info.frame_count, step))
    after_hours = {i for i in sampled
                   if period.status_at(start + timedelta(seconds=i / info.fps), tz) == "restricted"}

    # Only frames after closing need the detector (cheaper; earlier frames can never produce an incident).
    detector.confidence, detector.iou, detector.image_size = cfg.detector.confidence, cfg.detector.iou, cfg.detector.image_size
    hits: list[dict] = []        # one entry per after-hours frame containing ≥1 person
    for i, frame in iter_frames(video, after_hours):
        dets = detector.detect(frame)                  # plain detection: no tracking prerequisite
        if not dets:
            continue
        local = (start + timedelta(seconds=i / info.fps)).astimezone(tz)
        hits.append({"frame_index": i, "time_s": round(i / info.fps, 2), "local": local, "dets": dets,
                     "max_conf": max(d.confidence for d in dets)})

    # Group frames into events; pick evidence frames (first, most confident, last) per event.
    groups = []
    for g in group_times([h["time_s"] for h in hits], EVENT_GAP_S):
        group = [hits[k] for k in g]
        picks = []
        for h in (group[0], max(group, key=lambda h: h["max_conf"]), group[-1]):
            if h not in picks:
                picks.append(h)
        groups.append((group, picks[:MAX_EVIDENCE]))
    # One extra sequential pass reads only the evidence frames (no raw frames kept in memory meanwhile).
    needed = {h["frame_index"] for _, picks in groups for h in picks}
    raw = dict(iter_frames(video, needed)) if needed else {}

    events = []
    for group, picks in groups:
        first, best, last = group[0], max(group, key=lambda h: h["max_conf"]), group[-1]
        evidence = []
        for hit in picks:
            frame = raw.get(hit["frame_index"])
            image = None
            if frame is not None:
                image = f"afterhours_{first['frame_index']}_{hit['frame_index']}.jpg"
                save_jpeg(_evidence_image(frame, hit["dets"], hit["local"], closing_display), evidence_dir / image)
            evidence.append({"time_s": hit["time_s"], "frame_index": hit["frame_index"],
                             "confidence": round(hit["max_conf"], 4), "image": image,
                             "details": {"persons": len(hit["dets"]),
                                         # list of objects (same shape as smoking's cigarette_boxes)
                                         "person_boxes": [{"bbox_xyxy_px": [round(v, 1) for v in d.bbox],
                                                           "confidence": round(d.confidence, 4)}
                                                          for d in hit["dets"][:10]],
                                         "timestamp_local": hit["local"].isoformat(timespec="seconds"),
                                         "closing_time": cfg.closing_time}})
        local = first["local"]
        name = cfg.camera_name or camera_id
        events.append({
            "event_id": f"restricted_zone-afterhours-{first['frame_index']}",   # deterministic → no duplicates
            "type": "restricted_zone",
            "detection_mode": "after_hours_person_presence",
            "start_s": first["time_s"], "end_s": last["time_s"],
            "confidence": round(best["max_conf"], 4),          # highest person-detection confidence in the event
            "camera_id": camera_id, "camera_name": cfg.camera_name,
            "closing_time": cfg.closing_time, "closing_time_display": closing_display,
            "recorded_local_time": local.isoformat(timespec="seconds"),
            "recorded_utc_time": local.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "persons_max_in_frame": max(len(h["dets"]) for h in group),
            "frames_with_person": len(group),
            "description": (f"Potential after-hours person detected in {name} at {clock_12h(local)} IST. "
                            f"Configured closing time: {closing_display}."),
            "evidence": evidence,
        })

    end_local = start + timedelta(seconds=info.duration_s)
    return {
        "available": True,
        "status": "completed",
        "potential_incident": bool(events),
        "events": events,
        "rule": rule,
        "summary": {"frames_sampled": len(sampled), "sample_fps": settings.zone_sample_fps,
                    "frames_after_closing": len(after_hours),
                    "frames_in_restricted_period": len(after_hours),   # same value, key the dashboard already shows
                    "frames_after_closing_with_person": len(hits),
                    "max_people_in_frame": max((len(h["dets"]) for h in hits), default=0),
                    "recorded_at_local": start.isoformat(timespec="seconds"),
                    "video_end_local": end_local.isoformat(timespec="seconds"),
                    "recorded_at_note": time_note,
                    "campus_timezone": cfg.timezone_name, "config": cfg_path.name},
    }
