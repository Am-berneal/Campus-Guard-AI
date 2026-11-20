"""Restricted-zone / closing-time analysis using the EXISTING module
(training.restricted_zone: config, PersonDetector + ByteTrack, zones, schedule, RuleEngine).

A potential incident needs ALL of:
  1. a tracked person detected (YOLO + ByteTrack),
  2. the person's ground point inside the camera's restricted zone,
  3. the RECORDING time of that frame inside the restricted period
     (closing time inclusive, e.g. COS from 20:00, others from 22:30, Asia/Kolkata),
  4. held for `min_dwell_seconds`; one incident per person and zone per cooldown.

Time comes from the recording, never from the upload/analysis clock:
  frame time = recorded_at (start of the video) + frame_index / fps.
If recorded_at is missing or implausible, the module returns status "time_unknown" and
creates no incidents, because the restriction cannot be verified.

Zones come from training/restricted_zone/configs/cameras/<camera_id>.yaml.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from inference.settings import Settings
from inference.video_io import VideoInfo, iter_frames, save_jpeg
from training.restricted_zone.config import ConfigError, load_config
from training.restricted_zone.demo_video import draw
from training.restricted_zone.rules import RuleEngine

LOG = logging.getLogger("inference.restricted_zone")
FUTURE_TOLERANCE = timedelta(minutes=10)   # a recording cannot start in the future


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


def analyze(video: Path, info: VideoInfo, detector, camera_id: str, settings: Settings, evidence_dir: Path,
            recorded_at: datetime | None) -> dict:
    cfg_path = zone_config_path(settings, camera_id)
    if not cfg_path.is_file():
        return {"available": False, "status": "not_configured",
                "reason": f"no zone configuration for camera '{camera_id}' ({cfg_path.name})",
                "potential_incident": False, "events": []}
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        raise ValueError(f"invalid zone configuration {cfg_path.name}: {exc}") from exc

    rule = {"closing_time": cfg.closing_time, "restricted_until": cfg.restricted_until,
            "closing_time_display": clock_12h(cfg.closing_time) if cfg.closing_time else None,
            "campus_timezone": cfg.timezone_name}
    zones_info = [{"zone_id": z.zone_id, "name": z.name, "calibrated": z.calibrated} for z in cfg.zones]

    start, time_note = normalize_recorded_at(recorded_at, cfg.timezone)
    if start is None:
        # Safe fallback: without a trustworthy recording time the restriction cannot be checked.
        return {"available": True, "status": "time_unknown",
                "reason": (f"Restricted-time rule not evaluated: {time_note}. Provide the recording start time "
                           f"(campus time, {cfg.timezone_name}) to check closing-time entry."),
                "potential_incident": False, "events": [], "rule": rule, "zones": zones_info}

    video_end = start + timedelta(seconds=info.duration_s)
    step = max(1, round(info.fps / settings.zone_sample_fps))
    wanted = set(range(0, info.frame_count, step))

    # Detector thresholds from this camera's config (the detector object is shared).
    detector.confidence, detector.iou, detector.image_size = cfg.detector.confidence, cfg.detector.iou, cfg.detector.image_size
    engine = RuleEngine(cfg)
    events, max_people, first = [], 0, True
    zone_restricted_seen = {z.zone_id: False for z in cfg.zones}
    zone_by_id = {z.zone_id: z for z in cfg.zones}
    restricted_frames = 0
    for i, frame in iter_frames(video, wanted):
        when = start + timedelta(seconds=i / info.fps)          # recording time of THIS frame
        dets = detector.track(frame, persist=not first)          # first frame: fresh tracker for this video
        first = False
        max_people = max(max_people, len(dets))
        result = engine.update(dets, info.width, info.height, when)
        if any(result.zone_restricted.values()):
            restricted_frames += 1
        for zid, r in result.zone_restricted.items():
            zone_restricted_seen[zid] |= r
        for inc in result.incidents:
            zone = zone_by_id[inc["zone_id"]]
            local = when.astimezone(cfg.timezone)
            name = f"zone_{inc['zone_id']}_{inc['track_id']}_{i}.jpg"
            save_jpeg(draw(frame.copy(), cfg, result, dets, local), evidence_dir / name)
            closing_txt = f" The configured closing time is {rule['closing_time_display']}." if cfg.closing_time else ""
            geometry_txt = "" if zone.calibrated else " Zone boundary is an uncalibrated placeholder."
            events.append({
                "event_id": f"restricted_zone-{inc['zone_id']}-{inc['track_id']}-{i}",
                "type": "restricted_zone",
                "start_s": round(max(0.0, i / info.fps - inc["dwell_seconds"]), 2), "end_s": round(i / info.fps, 2),
                "confidence": inc["confidence"],
                "zone_id": inc["zone_id"], "zone_name": inc["zone_name"], "zone_calibrated": zone.calibrated,
                "track_id": inc["track_id"],
                "camera_name": cfg.camera_name,
                "closing_time": cfg.closing_time,
                "closing_time_display": rule["closing_time_display"],
                "recorded_local_time": local.isoformat(timespec="seconds"),
                "recorded_utc_time": local.astimezone(timezone.utc).isoformat(timespec="seconds"),
                "description": (f"Potential restricted-zone entry detected in {cfg.camera_name or camera_id} at "
                                f"{clock_12h(local)} on {local.strftime('%d %b %Y')} (campus time).{closing_txt} "
                                f"A person (track {inc['track_id']}) was inside '{inc['zone_name']}' for at least "
                                f"{inc['dwell_seconds']:.1f} s.{geometry_txt} Requires administrator review."),
                "evidence": [{"time_s": round(i / info.fps, 2), "frame_index": i, "confidence": inc["confidence"],
                              "image": name,
                              "details": {"zone_id": inc["zone_id"], "zone_name": inc["zone_name"],
                                          "track_id": inc["track_id"], "bbox_xyxy_px": inc["bbox_xyxy_px"],
                                          "timestamp_local": inc["timestamp_local"],
                                          "timestamp_utc": inc["timestamp_utc"],
                                          "closing_time": cfg.closing_time}}],
            })

    return {
        "available": True,
        "status": "completed",
        "potential_incident": bool(events),
        "events": events,
        "rule": rule,
        "zones": [{**z, "restricted_during_video": zone_restricted_seen[z["zone_id"]]} for z in zones_info],
        "summary": {"frames_sampled": len(wanted), "sample_fps": settings.zone_sample_fps,
                    "frames_in_restricted_period": restricted_frames,
                    "max_people_in_frame": max_people,
                    "recorded_at_local": start.isoformat(timespec="seconds"),
                    "video_end_local": video_end.isoformat(timespec="seconds"),
                    "recorded_at_note": time_note,
                    "recorded_at_source": "request",
                    "campus_timezone": cfg.timezone_name, "config": cfg_path.name},
    }
