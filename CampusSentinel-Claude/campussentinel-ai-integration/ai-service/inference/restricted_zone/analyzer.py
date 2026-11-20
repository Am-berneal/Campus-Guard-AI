"""Restricted-zone / restricted-time analysis using the EXISTING module
(training.restricted_zone: config, PersonDetector + ByteTrack, zones, schedule, RuleEngine).

Zones come from training/restricted_zone/configs/cameras/<camera_id>.yaml (same format as
zones.sample.yaml). A recorded upload has no trustworthy clock, so frame time is
`recorded_at + frame_time` (recorded_at defaults to the moment of analysis, in campus time).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from inference.settings import Settings
from inference.video_io import VideoInfo, iter_frames, save_jpeg
from training.restricted_zone.config import ConfigError, load_config
from training.restricted_zone.demo_video import draw
from training.restricted_zone.rules import RuleEngine

LOG = logging.getLogger("inference.restricted_zone")


def zone_config_path(settings: Settings, camera_id: str) -> Path:
    return settings.zone_config_dir / f"{camera_id}.yaml"


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

    if recorded_at is not None and recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=cfg.timezone)   # a time without offset = campus local time
    start = recorded_at.astimezone(cfg.timezone) if recorded_at else datetime.now(cfg.timezone)
    step = max(1, round(info.fps / settings.zone_sample_fps))
    wanted = set(range(0, info.frame_count, step))

    # Detector settings from this camera's config (the detector object is shared).
    detector.confidence, detector.iou, detector.image_size = cfg.detector.confidence, cfg.detector.iou, cfg.detector.image_size
    engine = RuleEngine(cfg)
    events, max_people, first = [], 0, True
    zone_restricted_seen = {z.zone_id: False for z in cfg.zones}
    for i, frame in iter_frames(video, wanted):
        when = start + timedelta(seconds=i / info.fps)
        dets = detector.track(frame, persist=not first)   # first frame: fresh tracker for this video
        first = False
        max_people = max(max_people, len(dets))
        result = engine.update(dets, info.width, info.height, when)
        for zid, r in result.zone_restricted.items():
            zone_restricted_seen[zid] |= r
        for inc in result.incidents:
            name = f"zone_{inc['zone_id']}_{inc['track_id']}_{i}.jpg"
            save_jpeg(draw(frame.copy(), cfg, result, dets, when.astimezone(cfg.timezone)), evidence_dir / name)
            events.append({
                "event_id": f"restricted_zone-{inc['zone_id']}-{inc['track_id']}-{i}",
                "type": "restricted_zone",
                "start_s": round(i / info.fps - inc["dwell_seconds"], 2), "end_s": round(i / info.fps, 2),
                "confidence": inc["confidence"],
                "zone_id": inc["zone_id"], "zone_name": inc["zone_name"], "track_id": inc["track_id"],
                "description": (f"Person (track {inc['track_id']}) inside restricted zone '{inc['zone_name']}' "
                                f"for ≥{inc['dwell_seconds']:.1f}s at {inc['timestamp_local']} (campus time)."),
                "evidence": [{"time_s": round(i / info.fps, 2), "frame_index": i, "confidence": inc["confidence"],
                              "image": name,
                              "details": {"zone_id": inc["zone_id"], "zone_name": inc["zone_name"],
                                          "track_id": inc["track_id"], "bbox_xyxy_px": inc["bbox_xyxy_px"],
                                          "timestamp_local": inc["timestamp_local"],
                                          "timestamp_utc": inc["timestamp_utc"]}}],
            })

    return {
        "available": True,
        "status": "completed",
        "potential_incident": bool(events),
        "events": events,
        "zones": [{"zone_id": z.zone_id, "name": z.name, "restricted_during_video": zone_restricted_seen[z.zone_id]}
                  for z in cfg.zones],
        "summary": {"frames_sampled": len(wanted), "sample_fps": settings.zone_sample_fps,
                    "max_people_in_frame": max_people,
                    "recorded_at_local": start.isoformat(timespec="seconds"),
                    "recorded_at_source": "request" if recorded_at else "analysis time (no recording time given)",
                    "campus_timezone": cfg.timezone_name, "config": cfg_path.name},
    }
