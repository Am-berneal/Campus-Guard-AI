"""Load and validate the restricted-zone YAML configuration.

All problems are reported as ConfigError with the zone and field that is wrong,
so a typo in the YAML never silently disables a zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from training.restricted_zone.schedule import DAY_NAMES, STATUSES, parse_hhmm, parse_schedule
from training.restricted_zone.zones import Zone

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DetectorSettings:
    weights: Path
    confidence: float = 0.4
    iou: float = 0.5
    image_size: int = 640
    device: str = "auto"      # auto | cuda | cpu
    half: bool = True         # float16 on CUDA only


@dataclass(frozen=True)
class IncidentSettings:
    min_dwell_seconds: float = 1.0      # must be inside a restricted zone this long before an incident
    cooldown_seconds: float = 300.0     # no second incident for the same person + zone within this time
    save_evidence_frames: bool = True


@dataclass(frozen=True)
class AppConfig:
    timezone: ZoneInfo
    timezone_name: str
    camera_id: str
    detector: DetectorSettings
    tracker_config: Path
    incidents: IncidentSettings
    zones: tuple[Zone, ...]
    camera_name: str | None = None
    closing_time: str | None = None      # "HH:MM" campus time; entry prohibited from this minute (inclusive)
    restricted_until: str | None = None  # "HH:MM" or "24:00"; end of the daily restricted period


def closing_time_schedule(closing: str, until: str) -> dict:
    """Schedule (existing format) for 'no entry from <closing> until <until>', every day.

    Start is inclusive, so the closing minute itself is restricted. until="24:00" ends the
    period at midnight; an earlier time (e.g. "06:00") makes it an overnight window.
    """
    return {"default": "allowed",
            "restricted": [{"days": list(DAY_NAMES), "start": closing, "end": until}]}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (AI_SERVICE_ROOT / p)


def _number(section: dict, key: str, default, lo, hi, ctx: str):
    value = section.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not (lo <= value <= hi):
        raise ConfigError(f"{ctx}.{key} must be a number between {lo} and {hi}, got {value!r}")
    return value


def _parse_zone(raw: dict, i: int, closing_rule: dict | None = None) -> Zone:
    ctx = f"zones[{i}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{ctx} must be a mapping")
    for key in ("id", "name", "polygon"):
        if key not in raw:
            raise ConfigError(f"{ctx} is missing '{key}'")
    zid = str(raw["id"]).strip()
    ctx = f"zone '{zid}'"
    if not zid:
        raise ConfigError(f"zones[{i}].id must not be empty")

    status = raw.get("status", "restricted")
    if status not in STATUSES:
        raise ConfigError(f"{ctx}: status must be one of {STATUSES}, got {status!r}")

    poly = raw["polygon"]
    if not isinstance(poly, list) or len(poly) < 3:
        raise ConfigError(f"{ctx}: polygon needs at least 3 [x, y] points")
    points = []
    for j, pt in enumerate(poly):
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2 and all(isinstance(v, (int, float)) for v in pt)):
            raise ConfigError(f"{ctx}: polygon point {j} must be [x, y], got {pt!r}")
        x, y = float(pt[0]), float(pt[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ConfigError(f"{ctx}: polygon point {j} {pt} is outside 0..1 (coordinates are normalised)")
        points.append((x, y))

    # A zone without its own schedule uses the camera's closing-time rule (if one is configured).
    raw_schedule = raw.get("schedule")
    if raw_schedule is None and closing_rule is not None:
        raw_schedule = closing_rule
    try:
        schedule = parse_schedule(raw_schedule)
    except ValueError as exc:
        raise ConfigError(f"{ctx}: schedule: {exc}") from exc

    calibrated = raw.get("calibrated", False)
    if not isinstance(calibrated, bool):
        raise ConfigError(f"{ctx}: calibrated must be true or false")
    return Zone(zid, str(raw["name"]), status, tuple(points), schedule, bool(raw.get("enabled", True)), calibrated)


def load_config_dict(raw: dict) -> AppConfig:
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be a YAML mapping")

    tz_name = raw.get("campus_timezone")
    if not tz_name:
        raise ConfigError("campus_timezone is required, e.g. Asia/Kolkata (the machine's local time zone is NOT used)")
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ConfigError(
            f"unknown campus_timezone {tz_name!r}. On Windows install time-zone data with: pip install tzdata"
        ) from exc

    det = raw.get("detector", {}) or {}
    detector = DetectorSettings(
        weights=_resolve(det.get("weights", "models/person/yolo26n.pt")),
        confidence=_number(det, "confidence", 0.4, 0.01, 0.99, "detector"),
        iou=_number(det, "iou", 0.5, 0.05, 0.95, "detector"),
        image_size=int(_number(det, "image_size", 640, 160, 1920, "detector")),
        device=str(det.get("device", "auto")),
        half=bool(det.get("half", True)),
    )
    if detector.device not in ("auto", "cuda", "cpu"):
        raise ConfigError(f"detector.device must be auto, cuda or cpu, got {detector.device!r}")

    trk = raw.get("tracker", {}) or {}
    tracker_config = _resolve(trk.get("config", "training/restricted_zone/trackers/bytetrack_campus.yaml"))

    inc = raw.get("incidents", {}) or {}
    incidents = IncidentSettings(
        min_dwell_seconds=float(_number(inc, "min_dwell_seconds", 1.0, 0, 3600, "incidents")),
        cooldown_seconds=float(_number(inc, "cooldown_seconds", 300, 0, 86400, "incidents")),
        save_evidence_frames=bool(inc.get("save_evidence_frames", True)),
    )

    # Optional camera closing time (e.g. "20:00"): no entry from that minute onwards.
    closing = raw.get("closing_time")
    until = raw.get("restricted_until", "24:00")
    closing_rule = None
    if closing is not None:
        try:
            parse_hhmm(str(closing), allow_24=False)
            parse_hhmm(str(until), allow_24=True)
        except ValueError as exc:
            raise ConfigError(f"closing_time / restricted_until: {exc}") from exc
        if str(until) == str(closing):
            raise ConfigError("restricted_until must differ from closing_time")
        closing_rule = closing_time_schedule(str(closing), str(until))

    zones_raw = raw.get("zones")
    if not isinstance(zones_raw, list) or not zones_raw:
        raise ConfigError("'zones' must be a non-empty list")
    zones = tuple(_parse_zone(z, i, closing_rule) for i, z in enumerate(zones_raw))
    ids = [z.zone_id for z in zones]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ConfigError(f"duplicate zone ids: {dupes}")

    return AppConfig(tz, tz_name, str(raw.get("camera_id", "camera-unknown")), detector, tracker_config, incidents, zones,
                     camera_name=str(raw["camera_name"]) if raw.get("camera_name") else None,
                     closing_time=str(closing) if closing is not None else None,
                     restricted_until=str(until) if closing is not None else None)


def load_config(path: Path | str) -> AppConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML syntax error in {path}: {exc}") from exc
    return load_config_dict(raw)
