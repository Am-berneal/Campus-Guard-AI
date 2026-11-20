"""Synthetic tests for the restricted-zone module (no model, no video, no GPU).

Run from the ai-service folder:
    python -m unittest discover -s training\\restricted_zone\\tests -t . -v

Calendar used: 2026-11-16 = Monday … 2026-11-20 = Friday, 21 = Saturday, 22 = Sunday.
"""

from __future__ import annotations

import copy
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from training.restricted_zone.config import ConfigError, load_config, load_config_dict
from training.restricted_zone.detector import PersonDetection, parse_boxes
from training.restricted_zone.rules import RuleEngine
from training.restricted_zone.schedule import parse_schedule
from training.restricted_zone.zones import Zone, bottom_center

IST = ZoneInfo("Asia/Kolkata")
SAMPLE = Path(__file__).resolve().parents[1] / "configs" / "zones.sample.yaml"
W, H = 1000, 1000  # synthetic frame size → pixel / 1000 = normalised


def at(y, mo, d, h, mi=0, s=0.0):
    return datetime(y, mo, d, h, mi, tzinfo=IST) + timedelta(seconds=s)


BASE = {
    "campus_timezone": "Asia/Kolkata",
    "camera_id": "test-cam",
    "incidents": {"min_dwell_seconds": 1.0, "cooldown_seconds": 60},
    "zones": [
        {   # left half: allowed Mon–Fri 06–20, otherwise restricted
            "id": "A", "name": "Zone A", "status": "restricted",
            "polygon": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]],
            "schedule": {"default": "restricted",
                         "allowed": [{"days": ["mon", "tue", "wed", "thu", "fri"], "start": "06:00", "end": "20:00"}]},
        },
        {   # right half: restricted every night 22–06, otherwise allowed
            "id": "B", "name": "Zone B", "status": "restricted",
            "polygon": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]],
            "schedule": {"default": "allowed",
                         "restricted": [{"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                                         "start": "22:00", "end": "06:00"}]},
        },
    ],
}


def person(track_id, foot_x, foot_y=900, conf=0.9):
    """Synthetic 100×300 px person whose bottom-centre (feet) is at (foot_x, foot_y)."""
    return PersonDetection((foot_x - 50, foot_y - 300, foot_x + 50, foot_y), conf, track_id)


# --------------------------------------------------------------------------- detection output
class TestDetectionParsing(unittest.TestCase):
    def test_keeps_only_persons_above_threshold(self):
        dets = parse_boxes([[0, 0, 10, 20], [5, 5, 15, 25], [1, 1, 2, 2]], [0.9, 0.8, 0.2], [0, 2, 0],
                           ids=[7, 8, 9], min_conf=0.4)
        self.assertEqual(len(dets), 1)  # class 2 (car) and low-confidence person dropped
        self.assertEqual(dets[0].track_id, 7)
        self.assertEqual(dets[0].bbox, (0.0, 0.0, 10.0, 20.0))

    def test_missing_track_ids(self):
        dets = parse_boxes([[0, 0, 10, 20]], [0.9], [0], ids=None)
        self.assertIsNone(dets[0].track_id)

    def test_empty_and_mismatch(self):
        self.assertEqual(parse_boxes([], [], []), [])
        with self.assertRaises(ValueError):
            parse_boxes([[0, 0, 1, 1]], [0.9, 0.8], [0])


# --------------------------------------------------------------------------- geometry
class TestGeometry(unittest.TestCase):
    def setUp(self):
        sched = parse_schedule({"default": "restricted"})
        self.square = Zone("sq", "Square", "restricted", ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)), sched)
        # L-shape: the top-right notch (0.75, 0.25) is OUTSIDE.
        self.ell = Zone("L", "L", "restricted",
                        ((0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.9, 0.5), (0.9, 0.9), (0.1, 0.9)), sched)

    def test_inside_outside_edge(self):
        self.assertTrue(self.square.contains((0.5, 0.5)))
        self.assertFalse(self.square.contains((0.1, 0.5)))
        self.assertTrue(self.square.contains((0.2, 0.5)))  # on the edge counts as inside

    def test_concave_polygon(self):
        self.assertTrue(self.ell.contains((0.3, 0.3)))
        self.assertFalse(self.ell.contains((0.75, 0.25)))
        self.assertTrue(self.ell.contains((0.75, 0.75)))

    def test_bottom_center(self):
        self.assertEqual(bottom_center((100, 200, 300, 800), 1000, 1000), (0.2, 0.8))
        with self.assertRaises(ValueError):
            bottom_center((0, 0, 1, 1), 0, 100)

    def test_person_standing_in_front_of_zone_is_outside(self):
        # Box overlaps the square, but feet are below it → outside.
        self.assertFalse(self.square.contains(bottom_center((400, 300, 600, 950), 1000, 1000)))


# --------------------------------------------------------------------------- schedules
class TestSchedule(unittest.TestCase):
    def setUp(self):
        self.library = parse_schedule({"default": "restricted",
                                       "allowed": [{"days": ["mon", "tue", "wed", "thu", "fri"],
                                                    "start": "06:00", "end": "20:00"}]})
        self.nightly = parse_schedule({"default": "allowed",
                                       "restricted": [{"days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
                                                       "start": "22:00", "end": "06:00"}]})
        self.friday_night = parse_schedule({"default": "allowed",
                                            "restricted": [{"days": ["fri"], "start": "22:00", "end": "06:00"}]})

    def s(self, sched, when):
        return sched.status_at(when, IST)

    def test_allowed_vs_restricted_time(self):
        self.assertEqual(self.s(self.library, at(2026, 11, 16, 10)), "allowed")      # Mon 10:00
        self.assertEqual(self.s(self.library, at(2026, 11, 16, 21)), "restricted")   # Mon 21:00
        self.assertEqual(self.s(self.library, at(2026, 11, 17, 5, 59)), "restricted")  # Tue 05:59
        self.assertEqual(self.s(self.library, at(2026, 11, 17, 6, 0)), "allowed")    # start inclusive
        self.assertEqual(self.s(self.library, at(2026, 11, 17, 20, 0)), "restricted")  # end exclusive

    def test_weekday_restriction(self):
        self.assertEqual(self.s(self.library, at(2026, 11, 21, 10)), "restricted")   # Saturday
        self.assertEqual(self.s(self.library, at(2026, 11, 22, 10)), "restricted")   # Sunday

    def test_overnight(self):
        self.assertEqual(self.s(self.nightly, at(2026, 11, 16, 21, 59)), "allowed")
        self.assertEqual(self.s(self.nightly, at(2026, 11, 16, 23, 30)), "restricted")
        self.assertEqual(self.s(self.nightly, at(2026, 11, 17, 3, 0)), "restricted")
        self.assertEqual(self.s(self.nightly, at(2026, 11, 17, 6, 0)), "allowed")

    def test_overnight_belongs_to_start_day(self):
        self.assertEqual(self.s(self.friday_night, at(2026, 11, 20, 23)), "restricted")  # Fri 23:00
        self.assertEqual(self.s(self.friday_night, at(2026, 11, 21, 2)), "restricted")   # Sat 02:00 (Fri window)
        self.assertEqual(self.s(self.friday_night, at(2026, 11, 22, 2)), "allowed")      # Sun 02:00
        self.assertEqual(self.s(self.friday_night, at(2026, 11, 20, 2)), "allowed")      # Fri 02:00 (Thu not listed)

    def test_restricted_overrides_allowed(self):
        sched = parse_schedule({"default": "allowed",
                                "allowed": [{"days": ["mon"], "start": "00:00", "end": "24:00"}],
                                "restricted": [{"days": ["mon"], "start": "12:00", "end": "13:00"}]})
        self.assertEqual(self.s(sched, at(2026, 11, 16, 12, 30)), "restricted")
        self.assertEqual(self.s(sched, at(2026, 11, 16, 23, 59)), "allowed")

    def test_timezone_conversion_and_naive_rejected(self):
        # 15:30 UTC Monday = 21:00 IST Monday → library restricted (after 20:00 local).
        utc_time = datetime(2026, 11, 16, 15, 30, tzinfo=timezone.utc)
        self.assertEqual(self.s(self.library, utc_time), "restricted")
        # 04:00 UTC Monday = 09:30 IST → allowed.
        self.assertEqual(self.s(self.library, datetime(2026, 11, 16, 4, 0, tzinfo=timezone.utc)), "allowed")
        with self.assertRaises(ValueError):
            self.library.status_at(datetime(2026, 11, 16, 10), IST)


# --------------------------------------------------------------------------- rule engine
class TestRuleEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RuleEngine(load_config_dict(BASE))

    def run_frames(self, dets_per_frame, start, fps=10):
        out = []
        for i, dets in enumerate(dets_per_frame):
            out += self.engine.update(dets, W, H, start + timedelta(seconds=i / fps)).incidents
        return out

    def test_inside_restricted_zone_creates_one_incident(self):
        # Zone A at Monday 21:00 is restricted. 3 s inside at 10 fps → exactly one incident.
        incs = self.run_frames([[person(1, 200)]] * 30, at(2026, 11, 16, 21))
        self.assertEqual(len(incs), 1)
        inc = incs[0]
        self.assertEqual((inc["type"], inc["zone_id"], inc["track_id"]), ("restricted_zone_entry", "A", 1))
        self.assertEqual(inc["status"], "pending_review")
        for key in ("zone_name", "timestamp_utc", "timestamp_local", "bbox_xyxy_px", "confidence", "evidence_frame"):
            self.assertIn(key, inc)
        self.assertGreaterEqual(inc["dwell_seconds"], 1.0)

    def test_dwell_time_required(self):
        # Only 0.5 s inside (5 frames at 10 fps) → no incident.
        self.assertEqual(self.run_frames([[person(1, 200)]] * 5, at(2026, 11, 16, 21)), [])

    def test_outside_zones_no_incident(self):
        self.assertEqual(self.run_frames([[person(1, 200, foot_y=1200)]] * 30, at(2026, 11, 16, 21)), [])

    def test_inside_during_allowed_time_no_incident(self):
        self.assertEqual(self.run_frames([[person(1, 200)]] * 30, at(2026, 11, 16, 10)), [])

    def test_multiple_zones(self):
        # Monday 23:00: A restricted (night), B restricted (22–06). Two people, one per zone.
        incs = self.run_frames([[person(1, 200), person(2, 800)]] * 20, at(2026, 11, 16, 23))
        self.assertEqual(sorted((i["track_id"], i["zone_id"]) for i in incs), [(1, "A"), (2, "B")])
        # Monday 21:00: A restricted, B allowed → only track 1.
        self.engine = RuleEngine(load_config_dict(BASE))
        incs = self.run_frames([[person(1, 200), person(2, 800)]] * 20, at(2026, 11, 16, 21))
        self.assertEqual([(i["track_id"], i["zone_id"]) for i in incs], [(1, "A")])

    def test_duplicate_suppression_and_cooldown(self):
        start = at(2026, 11, 16, 21)
        inside, outside = [person(1, 200)], [person(1, 200, foot_y=1200)]
        # inside 2 s → incident; out 1 s; back in 2 s (within 60 s cooldown) → no new incident.
        frames = [inside] * 20 + [outside] * 10 + [inside] * 20
        self.assertEqual(len(self.run_frames(frames, start)), 1)
        # 70 s later (after cooldown) still inside → second incident.
        later = self.engine.update(inside, W, H, start + timedelta(seconds=70)).incidents
        later += self.engine.update(inside, W, H, start + timedelta(seconds=71.5)).incidents
        self.assertEqual(len(later), 1)

    def test_untracked_detections_ignored(self):
        self.assertEqual(self.run_frames([[person(None, 200)]] * 30, at(2026, 11, 16, 21)), [])

    def test_allowed_status_zone_never_triggers(self):
        cfg = copy.deepcopy(BASE)
        cfg["zones"][0]["status"] = "allowed"
        self.engine = RuleEngine(load_config_dict(cfg))
        self.assertEqual(self.run_frames([[person(1, 200)]] * 30, at(2026, 11, 16, 21)), [])

    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.update([person(1, 200)], W, H, datetime(2026, 11, 16, 21))


# --------------------------------------------------------------------------- configuration
class TestConfig(unittest.TestCase):
    def test_sample_config_loads(self):
        cfg = load_config(SAMPLE)
        self.assertEqual(len(cfg.zones), 2)
        self.assertEqual(cfg.timezone_name, "Asia/Kolkata")

    def bad(self, mutate, fragment):
        cfg = copy.deepcopy(BASE)
        mutate(cfg)
        with self.assertRaises(ConfigError) as ctx:
            load_config_dict(cfg)
        self.assertIn(fragment, str(ctx.exception))

    def test_invalid_configs(self):
        self.bad(lambda c: c["zones"][0].update(polygon=[[0, 0], [1, 1]]), "at least 3")
        self.bad(lambda c: c["zones"][0].update(polygon=[[0, 0], [1.5, 0], [1, 1]]), "outside 0..1")
        self.bad(lambda c: c["zones"][1].update(id="A"), "duplicate zone ids")
        self.bad(lambda c: c["zones"][0]["schedule"]["allowed"][0].update(start="25:00"), "invalid time")
        self.bad(lambda c: c["zones"][0]["schedule"]["allowed"][0].update(days=["funday"]), "unknown day")
        self.bad(lambda c: c["zones"][0]["schedule"]["allowed"][0].update(end="06:00"), "equal")
        self.bad(lambda c: c["zones"][0].update(status="maybe"), "status")
        self.bad(lambda c: c.update(campus_timezone="Mars/Olympus"), "unknown campus_timezone")
        self.bad(lambda c: c.pop("campus_timezone"), "campus_timezone is required")
        self.bad(lambda c: c.update(zones=[]), "non-empty list")
        self.bad(lambda c: c["incidents"].update(cooldown_seconds=-5), "cooldown_seconds")
        self.bad(lambda c: c["zones"][0].pop("polygon"), "missing 'polygon'")

    def test_missing_file(self):
        with self.assertRaises(ConfigError):
            load_config("does/not/exist.yaml")


if __name__ == "__main__":
    unittest.main()
