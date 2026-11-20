"""Rule engine: tracked people + zones + time → POTENTIAL incidents.

An incident is created when a tracked person's ground point has been inside a zone
that is restricted at that moment for at least `min_dwell_seconds`. After that, the
same (track, zone) pair stays silent for `cooldown_seconds`, even if the person
steps out and back in. Detections without a track ID are ignored.

Every incident is a POTENTIAL violation that requires administrator review.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from training.restricted_zone.config import AppConfig
from training.restricted_zone.detector import PersonDetection
from training.restricted_zone.zones import bottom_center


@dataclass
class _PairState:
    entered_at: datetime | None = None       # start of the current continuous stay inside
    last_incident_at: datetime | None = None
    last_seen: datetime | None = None


@dataclass
class FrameResult:
    incidents: list[dict]
    zone_restricted: dict[str, bool]                     # zone_id → restricted right now
    person_zones: dict[int, list[str]] = field(default_factory=dict)   # track_id → zones it is inside
    violating: set[int] = field(default_factory=set)     # track_ids inside a restricted zone


class RuleEngine:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self._state: dict[tuple[int, str], _PairState] = {}

    def update(self, detections: list[PersonDetection], frame_w: int, frame_h: int, when: datetime) -> FrameResult:
        if when.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        tz = self.cfg.timezone
        restricted_now = {z.zone_id: z.is_restricted_at(when, tz) for z in self.cfg.zones}
        result = FrameResult([], restricted_now)
        inc = self.cfg.incidents

        for det in detections:
            if det.track_id is None:
                continue
            point = bottom_center(det.bbox, frame_w, frame_h)
            for zone in self.cfg.zones:
                if not zone.enabled or not zone.contains(point):
                    # Left the zone (or never in it): end the continuous stay.
                    st = self._state.get((det.track_id, zone.zone_id))
                    if st:
                        st.entered_at = None
                    continue

                result.person_zones.setdefault(det.track_id, []).append(zone.zone_id)
                st = self._state.setdefault((det.track_id, zone.zone_id), _PairState())
                st.last_seen = when
                if not restricted_now[zone.zone_id]:
                    st.entered_at = None
                    continue

                result.violating.add(det.track_id)
                if st.entered_at is None:
                    st.entered_at = when
                dwell = (when - st.entered_at).total_seconds()
                cooling = (st.last_incident_at is not None
                           and (when - st.last_incident_at).total_seconds() < inc.cooldown_seconds)
                if dwell >= inc.min_dwell_seconds and not cooling:
                    st.last_incident_at = when
                    result.incidents.append(self._incident(det, zone, point, when, dwell, frame_w, frame_h))

        self._prune(when)
        return result

    def _incident(self, det, zone, point, when, dwell, w, h) -> dict:
        x1, y1, x2, y2 = det.bbox
        return {
            "incident_id": uuid.uuid4().hex,
            "type": "restricted_zone_entry",
            "status": "pending_review",          # requires administrator review; no automatic action
            "camera_id": self.cfg.camera_id,
            "zone_id": zone.zone_id,
            "zone_name": zone.name,
            "track_id": det.track_id,
            "timestamp_utc": when.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
            "timestamp_local": when.astimezone(self.cfg.timezone).isoformat(timespec="milliseconds"),
            "campus_timezone": self.cfg.timezone_name,
            "confidence": round(det.confidence, 4),
            "bbox_xyxy_px": [round(v, 1) for v in det.bbox],
            "bbox_xyxy_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
            "ground_point_norm": [round(point[0], 4), round(point[1], 4)],
            "dwell_seconds": round(dwell, 2),
            "frame_size": [w, h],
            "evidence_frame": None,              # filled in by the caller if a frame is saved
            "detector": "pretrained YOLO person detector + ByteTrack (not trained by us)",
        }

    def _prune(self, now: datetime) -> None:
        """Forget pairs not seen for longer than the cooldown (bounded memory on long streams)."""
        horizon = self.cfg.incidents.cooldown_seconds + 10
        stale = [k for k, s in self._state.items()
                 if s.last_seen is not None and (now - s.last_seen).total_seconds() > horizon]
        for k in stale:
            del self._state[k]
