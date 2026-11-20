"""Closing-time / restricted-zone tests using the REAL camera configs in
training/restricted_zone/configs/cameras/ (official times: COS 20:00, others 22:30, Asia/Kolkata).

Rule-engine tests use synthetic detections (no model). The analyzer tests at the end use the
COCO person detector if its weights are on disk, otherwise they are skipped.

Run from the ai-service folder:
    python -m unittest discover -s inference\\tests -t . -v
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from inference.restricted_zone.analyzer import clock_12h, normalize_recorded_at
from inference.settings import AI_SERVICE_ROOT, Settings
from training.restricted_zone.config import ConfigError, load_config, load_config_dict
from training.restricted_zone.detector import PersonDetection
from training.restricted_zone.rules import RuleEngine

IST = ZoneInfo("Asia/Kolkata")
CAMERA_DIR = AI_SERVICE_ROOT / "training" / "restricted_zone" / "configs" / "cameras"
W = H = 1000
OFFICIAL = {"cos": ("COS", "20:00"), "library": ("Outside Library", "22:30"),
            "b_block": ("B Block", "22:30"), "kravings": ("Kravings", "22:30")}
COCO = next((p for p in (AI_SERVICE_ROOT / "models/person/yolo26n.pt",
                         AI_SERVICE_ROOT / "models/smoking/yolo/base/yolo26n.pt") if p.is_file()), None)


def cfg(camera: str):
    return load_config(CAMERA_DIR / f"{camera}.yaml")


def ist(h, m, s=0, day=16):
    return datetime(2026, 11, day, h, m, s, tzinfo=IST)  # 16 Nov 2026 = Monday


def person_at(track: int, x_norm: float, y_norm: float) -> PersonDetection:
    """Synthetic person whose feet (bottom-centre) are at the given normalised point."""
    x, y = x_norm * W, y_norm * H
    return PersonDetection((x - 40, y - 250, x + 40, y), 0.9, track)


def inside_point(camera: str) -> tuple[float, float]:
    poly = cfg(camera).zones[0].polygon
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))  # centroid (convex placeholders)


def run(engine: RuleEngine, start: datetime, seconds: float, dets_fn, fps: int = 10) -> list[dict]:
    out = []
    for k in range(int(seconds * fps)):
        when = start + timedelta(seconds=k / fps)
        out += engine.update(dets_fn(when), W, H, when).incidents
    return out


class TestCameraConfigs(unittest.TestCase):
    def test_all_four_cameras_match_official_rules(self):
        for cam, (name, closing) in OFFICIAL.items():
            c = cfg(cam)
            self.assertEqual((c.camera_id, c.camera_name, c.closing_time, c.timezone_name),
                             (cam, name, closing, "Asia/Kolkata"), cam)
            self.assertEqual(len(c.zones), 1)
            self.assertFalse(c.zones[0].calibrated, f"{cam}: placeholder geometry must be marked uncalibrated")

    def test_cos_boundaries(self):
        z, tz = cfg("cos").zones[0], IST
        self.assertFalse(z.is_restricted_at(ist(19, 59), tz))     # 7:59 PM allowed
        self.assertFalse(z.is_restricted_at(ist(19, 59, 59), tz))
        self.assertTrue(z.is_restricted_at(ist(20, 0), tz))       # 8:00 PM restricted (inclusive)
        self.assertTrue(z.is_restricted_at(ist(20, 1), tz))       # 8:01 PM restricted
        self.assertTrue(z.is_restricted_at(ist(23, 59, 59), tz))

    def test_other_locations_boundaries(self):
        for cam in ("library", "b_block", "kravings"):
            z = cfg(cam).zones[0]
            self.assertFalse(z.is_restricted_at(ist(22, 29), IST), cam)   # 10:29 PM allowed
            self.assertTrue(z.is_restricted_at(ist(22, 30), IST), cam)    # 10:30 PM restricted
            self.assertTrue(z.is_restricted_at(ist(22, 31), IST), cam)    # 10:31 PM restricted
            self.assertFalse(z.is_restricted_at(ist(20, 15), IST), cam)   # COS's time does not apply here

    def test_every_weekday_and_after_midnight(self):
        z = cfg("cos").zones[0]
        for day in range(16, 23):                                         # Mon 16 .. Sun 22 Nov
            self.assertTrue(z.is_restricted_at(ist(20, 0, day=day), IST))
        # Rule as given ("from 8 PM onwards", restricted_until 24:00): after midnight it is NOT restricted.
        self.assertFalse(z.is_restricted_at(ist(0, 0, day=17), IST))
        self.assertFalse(z.is_restricted_at(ist(7, 0, day=17), IST))

    def test_restricted_until_overnight_option(self):
        raw = yaml.safe_load((CAMERA_DIR / "cos.yaml").read_text(encoding="utf-8"))
        raw["restricted_until"] = "06:00"
        z = load_config_dict(raw).zones[0]
        self.assertTrue(z.is_restricted_at(ist(2, 0, day=17), IST))      # 2 AM after a 8 PM closing
        self.assertFalse(z.is_restricted_at(ist(6, 0, day=17), IST))
        self.assertFalse(z.is_restricted_at(ist(19, 59, day=17), IST))

    def test_invalid_closing_configs(self):
        raw = yaml.safe_load((CAMERA_DIR / "cos.yaml").read_text(encoding="utf-8"))
        bad = dict(raw, closing_time=1200)                                # what YAML makes of unquoted 20:00
        with self.assertRaises(ConfigError):
            load_config_dict(bad)
        with self.assertRaises(ConfigError):
            load_config_dict(dict(raw, closing_time="25:00"))
        with self.assertRaises(ConfigError):
            load_config_dict(dict(raw, restricted_until="20:00"))


class TestZoneAndTime(unittest.TestCase):
    def test_four_cases_cos(self):
        inside, outside = inside_point("cos"), (0.95, 0.2)
        cases = [  # (start time, foot point or None, expected incident?)
            (ist(20, 30), outside, False),   # outside zone after closing
            (ist(19, 30), inside, False),    # inside zone before closing
            (ist(20, 30), inside, True),     # inside zone after closing
            (ist(20, 30), None, False),      # nobody detected
        ]
        for start, point, expected in cases:
            engine = RuleEngine(cfg("cos"))
            incs = run(engine, start, 3, lambda _w, p=point: [person_at(1, *p)] if p else [])
            self.assertEqual(bool(incs), expected, (start, point))

    def test_other_cameras_inside_after_closing(self):
        for cam in ("library", "b_block", "kravings"):
            pt = inside_point(cam)
            self.assertTrue(run(RuleEngine(cfg(cam)), ist(22, 45), 3, lambda _w: [person_at(1, *pt)]), cam)
            self.assertFalse(run(RuleEngine(cfg(cam)), ist(22, 0), 3, lambda _w: [person_at(1, *pt)]), cam)

    def test_video_crossing_closing_time(self):
        """Video starts 7:59:50 PM; person inside throughout → incident only from 8:00 PM (+1 s dwell)."""
        pt = inside_point("cos")
        incs = run(RuleEngine(cfg("cos")), ist(19, 59, 50), 20, lambda _w: [person_at(1, *pt)])
        self.assertEqual(len(incs), 1)
        t = datetime.fromisoformat(incs[0]["timestamp_local"])
        self.assertGreaterEqual(t, ist(20, 0, 1))
        self.assertLess(t, ist(20, 0, 2))

    def test_no_duplicates_for_same_person(self):
        pt = inside_point("cos")
        incs = run(RuleEngine(cfg("cos")), ist(20, 30), 60, lambda _w: [person_at(7, *pt)])
        self.assertEqual(len(incs), 1)      # 60 s inside → still one incident (cooldown 300 s)

    def test_video_crossing_midnight(self):
        """Starts 11:59:50 PM: restricted until midnight only (rule as given)."""
        pt = inside_point("cos")
        incs = run(RuleEngine(cfg("cos")), ist(23, 59, 50), 20, lambda _w: [person_at(1, *pt)])
        self.assertEqual(len(incs), 1)
        self.assertLess(datetime.fromisoformat(incs[0]["timestamp_local"]), ist(0, 0, day=17))


class TestTimestamps(unittest.TestCase):
    NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)

    def test_aware_utc_converted_to_ist(self):
        local, note = normalize_recorded_at(datetime(2026, 11, 16, 14, 45, tzinfo=timezone.utc), IST, self.NOW)
        self.assertEqual(local, ist(20, 15))
        self.assertEqual(local.utcoffset(), timedelta(hours=5, minutes=30))
        self.assertIn("converted", note)

    def test_other_offset_crosses_date(self):
        ny = timezone(timedelta(hours=-5))
        local, _ = normalize_recorded_at(datetime(2026, 11, 16, 12, 0, tzinfo=ny), IST, self.NOW)  # 17:00 UTC
        self.assertEqual(local, ist(22, 30))
        local2, _ = normalize_recorded_at(datetime(2026, 11, 16, 19, 0, tzinfo=timezone.utc), IST, self.NOW)
        self.assertEqual(local2, ist(0, 30, day=17))                      # next calendar day in IST

    def test_naive_is_campus_time_and_says_so(self):
        local, note = normalize_recorded_at(datetime(2026, 11, 16, 20, 15), IST, self.NOW)
        self.assertEqual(local, ist(20, 15))
        self.assertIn("no UTC offset", note)

    def test_missing_and_future(self):
        self.assertEqual(normalize_recorded_at(None, IST, self.NOW)[0], None)
        future, reason = normalize_recorded_at(datetime(2027, 1, 1, tzinfo=timezone.utc), IST, self.NOW)
        self.assertIsNone(future)
        self.assertIn("future", reason)

    def test_clock_format(self):
        self.assertEqual(clock_12h("20:00"), "8:00 PM")
        self.assertEqual(clock_12h("22:30"), "10:30 PM")
        self.assertEqual(clock_12h(ist(20, 15)), "8:15 PM")


@unittest.skipUnless(COCO, "no COCO person weights on disk")
class TestAnalyzerWithPersonDetector(unittest.TestCase):
    """Real YOLO person detector + ByteTrack on Ultralytics' sample photo (people standing in a street)."""

    @classmethod
    def setUpClass(cls):
        import cv2
        import ultralytics
        from inference.model_manager import ModelManager
        from inference.video_io import probe

        cls.tmp = Path(tempfile.mkdtemp(prefix="cs_close_"))
        img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
        cls.video = cls.tmp / "people.avi"
        w = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 720))
        for _ in range(40):                                               # 4 s, static scene
            w.write(cv2.resize(img[0:900, 50:750], (640, 720)))
        w.release()
        cls.info = probe(cls.video)
        cls.settings = Settings(device="cpu", person_weights=COCO, evidence_dir=cls.tmp / "ev",
                                fight_checkpoint=cls.tmp / "x", smoking_cnn_checkpoint=cls.tmp / "x",
                                cigarette_checkpoint=cls.tmp / "x")
        cls.det = ModelManager(cls.settings).get("person_yolo")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)   # removes only this test's own temp folder

    def analyze(self, recorded_at):
        from inference.restricted_zone.analyzer import analyze
        return analyze(self.video, self.info, self.det, "cos", self.settings, self.tmp / "ev", recorded_at)

    def test_time_unknown_without_recording_time(self):
        r = self.analyze(None)
        self.assertEqual(r["status"], "time_unknown")
        self.assertEqual(r["events"], [])
        self.assertIn("not evaluated", r["reason"])

    def test_before_and_after_closing(self):
        before = self.analyze(datetime(2026, 11, 16, 19, 30, tzinfo=IST))
        self.assertEqual(before["status"], "completed")
        self.assertEqual(before["events"], [])                            # people present, but before 8 PM
        self.assertGreater(before["summary"]["max_people_in_frame"], 0)
        after = self.analyze(datetime(2026, 11, 16, 20, 15, tzinfo=IST))
        self.assertTrue(after["events"], "people inside the COS placeholder zone after 8 PM")
        ev = after["events"][0]
        self.assertEqual((ev["closing_time"], ev["closing_time_display"], ev["camera_name"]), ("20:00", "8:00 PM", "COS"))
        self.assertTrue(ev["recorded_local_time"].startswith("2026-11-16T20:15"))
        self.assertIn("Potential restricted-zone entry detected in COS at 8:15 PM", ev["description"])
        self.assertIn("closing time is 8:00 PM", ev["description"])
        self.assertFalse(ev["zone_calibrated"])
        self.assertTrue((self.tmp / "ev" / ev["evidence"][0]["image"]).is_file())


if __name__ == "__main__":
    unittest.main()
