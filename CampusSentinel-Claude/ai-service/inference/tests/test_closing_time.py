"""After-hours person detection tests (official closing times, REAL camera configs).

Most tests drive the real analyzer with a synthetic video and a FAKE person detector that
returns scripted detections, so the time logic is tested exactly and without a model.
The last class uses the real COCO person detector if its weights are on disk.

Run from the ai-service folder:
    python -m unittest discover -s inference\\tests -t . -v
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import yaml

from inference.restricted_zone.analyzer import analyze, clock_12h, normalize_recorded_at
from inference.settings import AI_SERVICE_ROOT, Settings
from inference.video_io import probe
from training.restricted_zone.config import ConfigError, load_config, load_config_dict
from training.restricted_zone.detector import PersonDetection

IST = ZoneInfo("Asia/Kolkata")
CAMERA_DIR = AI_SERVICE_ROOT / "training" / "restricted_zone" / "configs" / "cameras"
OFFICIAL = {"cos": ("COS", "20:00"), "library": ("Outside Library", "22:30"),
            "b_block": ("B Block", "22:30"), "kravings": ("Kravings", "22:30")}
COCO = next((p for p in (AI_SERVICE_ROOT / "models/person/yolo26n.pt",
                         AI_SERVICE_ROOT / "models/smoking/yolo/base/yolo26n.pt") if p.is_file()), None)
FPS, W, H = 10, 320, 240


def ist(h, m, s=0, day=16):
    return datetime(2026, 11, day, h, m, s, tzinfo=IST)   # 16 Nov 2026 = Monday


class FakeDetector:
    """Stand-in for PersonDetector.detect(): returns scripted detections per frame index.

    Frames encode their index in pixel (0,0) so the fake knows which frame it was given."""

    def __init__(self, script):
        self.script = script                  # callable(frame_index) -> list[PersonDetection]
        self.calls = 0
        self.confidence = self.iou = self.image_size = None

    def detect(self, frame):
        self.calls += 1
        return self.script(decode_index(frame))

    def track(self, *a, **k):                 # must NOT be needed any more
        raise AssertionError("tracking must not be required for after-hours detection")


BITS, BLOCK = 12, 16


def encode_index(frame, i):
    """Write frame index i as 12 black/white 16x16 blocks (survives lossy MJPG compression)."""
    for b in range(BITS):
        frame[0:BLOCK, b * BLOCK:(b + 1) * BLOCK] = 255 if (i >> b) & 1 else 0


def decode_index(frame):
    return sum(1 << b for b in range(BITS) if frame[0:BLOCK, b * BLOCK:(b + 1) * BLOCK].mean() > 127)


def person(x1=10, y1=10, x2=60, y2=120, conf=0.8, track_id=None):
    return PersonDetection((x1, y1, x2, y2), conf, track_id)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cs_afterhours_"))
        cls.video = cls.tmp / "v.avi"
        w = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
        for i in range(FPS * 20):                                     # 20 s
            f = np.full((H, W, 3), 90, np.uint8)
            encode_index(f, i)                                        # frame index for the fake detector
            w.write(f)
        w.release()
        cls.info = probe(cls.video)
        cls.settings = Settings(device="cpu", zone_sample_fps=5.0, evidence_dir=cls.tmp / "ev")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)   # only this test's own temp folder

    def run_cam(self, camera, start, script=lambda i: [person()], settings=None, video=None):
        det = FakeDetector(script)
        r = analyze(video or self.video, self.info, det, camera, settings or self.settings, self.tmp / "ev", start)
        return r, det

    def short(self, seconds):
        """A short clip so a start time like 19:59:00 stays entirely before closing."""
        p = self.tmp / f"short_{seconds}.avi"
        if not p.exists():
            w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
            for i in range(int(FPS * seconds)):
                f = np.full((H, W, 3), 90, np.uint8)
                encode_index(f, i)
                w.write(f)
            w.release()
        return p


# --------------------------------------------------------------------------- configs + time
class TestConfigsAndTime(unittest.TestCase):
    def test_official_closing_times_unchanged(self):
        for cam, (name, closing) in OFFICIAL.items():
            c = load_config(CAMERA_DIR / f"{cam}.yaml")
            self.assertEqual((c.camera_id, c.camera_name, c.closing_time, c.timezone_name, c.restricted_until),
                             (cam, name, closing, "Asia/Kolkata", "24:00"))

    def test_invalid_closing_configs(self):
        raw = yaml.safe_load((CAMERA_DIR / "cos.yaml").read_text(encoding="utf-8"))
        for bad in ({"closing_time": 1200}, {"closing_time": "25:00"}, {"restricted_until": "20:00"}):
            with self.assertRaises(ConfigError):
                load_config_dict({**raw, **bad})

    def test_timestamps(self):
        now = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(normalize_recorded_at(datetime(2026, 11, 16, 14, 45, tzinfo=timezone.utc), IST, now)[0], ist(20, 15))
        ny = timezone(timedelta(hours=-5))
        self.assertEqual(normalize_recorded_at(datetime(2026, 11, 16, 12, 0, tzinfo=ny), IST, now)[0], ist(22, 30))
        self.assertEqual(normalize_recorded_at(datetime(2026, 11, 16, 19, 0, tzinfo=timezone.utc), IST, now)[0],
                         ist(0, 30, day=17))                          # next calendar day in IST
        local, note = normalize_recorded_at(datetime(2026, 11, 16, 20, 15), IST, now)
        self.assertEqual(local, ist(20, 15))
        self.assertIn("no UTC offset", note)
        self.assertIsNone(normalize_recorded_at(None, IST, now)[0])
        self.assertIsNone(normalize_recorded_at(datetime(2027, 1, 1, tzinfo=timezone.utc), IST, now)[0])

    def test_clock(self):
        self.assertEqual((clock_12h("20:00"), clock_12h("22:30"), clock_12h(ist(20, 15))),
                         ("8:00 PM", "10:30 PM", "8:15 PM"))


# --------------------------------------------------------------------------- required scenarios
class TestAfterHoursRule(Base):
    def assert_case(self, cam, start, has_person, expected, clip_s=1.0):
        script = (lambda i: [person()]) if has_person else (lambda i: [])
        r, _ = self.run_cam(cam, start, script, video=self.short(clip_s))
        self.assertEqual(r["status"], "completed")
        self.assertEqual(bool(r["events"]), expected, (cam, start, has_person))
        return r

    def test_cos(self):
        self.assert_case("cos", ist(19, 59), True, False)            # 7:59 PM + person → no incident
        self.assert_case("cos", ist(20, 0), True, True)              # 8:00 PM + person → incident (inclusive)
        r = self.assert_case("cos", ist(20, 15), True, True)         # 8:15 PM + person → incident
        self.assert_case("cos", ist(20, 15), False, False)           # 8:15 PM, nobody → no incident
        ev = r["events"][0]
        self.assertEqual(ev["description"],
                         "Potential after-hours person detected in COS at 8:15 PM IST. Configured closing time: 8:00 PM.")
        self.assertEqual((ev["type"], ev["camera_id"], ev["camera_name"], ev["closing_time"]),
                         ("restricted_zone", "cos", "COS", "20:00"))
        self.assertTrue(ev["recorded_local_time"].startswith("2026-11-16T20:15:00+05:30"))
        self.assertEqual(ev["confidence"], 0.8)
        self.assertTrue((self.tmp / "ev" / ev["evidence"][0]["image"]).is_file())

    def test_other_three_cameras(self):
        for cam in ("library", "b_block", "kravings"):
            self.assert_case(cam, ist(22, 29), True, False)          # 10:29 PM → none
            self.assert_case(cam, ist(22, 30), True, True)           # 10:30 PM → incident
            self.assert_case(cam, ist(22, 45), True, True)           # 10:45 PM → incident
            self.assert_case(cam, ist(22, 45), False, False)         # 10:45 PM, nobody → none
            self.assert_case(cam, ist(20, 15), True, False)          # COS's closing time does not apply

    def test_person_anywhere_in_frame(self):
        corners = [(0, 0, 30, 60), (W - 30, 0, W, 60), (0, H - 60, 30, H), (W - 30, H - 60, W, H), (140, 90, 180, 150)]
        for box in corners:
            r, _ = self.run_cam("cos", ist(20, 15), lambda i, b=box: [person(*b)], video=self.short(1))
            self.assertTrue(r["events"], box)

    def test_polygon_geometry_is_ignored(self):
        # Move the (unused) zone polygon to a tiny corner far from the person: still an incident.
        raw = yaml.safe_load((CAMERA_DIR / "cos.yaml").read_text(encoding="utf-8"))
        raw["zones"][0]["polygon"] = [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]]
        zdir = self.tmp / "zones_tiny"
        zdir.mkdir(exist_ok=True)
        (zdir / "cos.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
        r, _ = self.run_cam("cos", ist(20, 15), lambda i: [person(250, 150, 300, 230)],
                            settings=replace(self.settings, zone_config_dir=zdir), video=self.short(1))
        self.assertTrue(r["events"])

    def test_single_detection_is_enough_and_untracked_counts(self):
        r, _ = self.run_cam("cos", ist(20, 15), lambda i: [person(track_id=None)] if i == 40 else [])
        self.assertEqual(len(r["events"]), 1)                          # one frame, no dwell, no track ID
        self.assertEqual(r["events"][0]["frames_with_person"], 1)
        self.assertEqual(r["events"][0]["start_s"], 4.0)

    def test_low_confidence_is_filtered_by_detector_threshold(self):
        # The analyzer passes the camera's confidence (0.4) to the detector; the real detector filters.
        r, det = self.run_cam("cos", ist(20, 15), lambda i: [], video=self.short(1))
        self.assertEqual(det.confidence, 0.4)
        self.assertEqual(r["events"], [])

    def test_missing_recording_time(self):
        r, det = self.run_cam("cos", None)
        self.assertEqual((r["status"], r["events"]), ("time_unknown", []))
        self.assertEqual(det.calls, 0)                                 # nothing evaluated, nothing guessed

    def test_video_crossing_closing_time(self):
        """Starts 7:59:55 PM; person throughout → event begins exactly at 8:00:00 PM (frame 50)."""
        r, det = self.run_cam("cos", ist(19, 59, 55))
        self.assertEqual(len(r["events"]), 1)
        self.assertTrue(r["events"][0]["recorded_local_time"].startswith("2026-11-16T20:00:00"))
        self.assertEqual(r["summary"]["frames_after_closing"], 75)      # 15 s × 5 fps
        self.assertEqual(det.calls, 75)                                 # detector not run before closing

    def test_crossing_midnight(self):
        r, _ = self.run_cam("cos", ist(23, 59, 55))                    # rule as given ends at midnight
        self.assertEqual(r["summary"]["frames_after_closing"], 25)
        self.assertTrue(r["events"])

    def test_grouping_and_deterministic_ids(self):
        # Person 0–4 s and 16–20 s (gap 12 s > 10 s) → two events; re-running gives the same IDs.
        script = lambda i: [person()] if i < 40 or i >= 160 else []
        r1, _ = self.run_cam("cos", ist(20, 15), script)
        r2, _ = self.run_cam("cos", ist(20, 15), script)
        self.assertEqual(len(r1["events"]), 2)
        self.assertEqual([e["event_id"] for e in r1["events"]], [e["event_id"] for e in r2["events"]])
        # Continuous presence for 20 s → ONE event, not one per frame.
        r3, _ = self.run_cam("cos", ist(20, 15))
        self.assertEqual(len(r3["events"]), 1)
        self.assertEqual(r3["events"][0]["frames_with_person"], 100)
        self.assertLessEqual(len(r3["events"][0]["evidence"]), 3)


@unittest.skipUnless(COCO, "no COCO person weights on disk")
class TestRealPersonDetector(unittest.TestCase):
    """Real YOLO26n person detector on Ultralytics' sample street photo (people standing)."""

    @classmethod
    def setUpClass(cls):
        import ultralytics
        from inference.model_manager import ModelManager

        cls.tmp = Path(tempfile.mkdtemp(prefix="cs_afterhours_real_"))
        img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
        cls.video = cls.tmp / "people.avi"
        w = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 720))
        for _ in range(30):
            w.write(cv2.resize(img[0:900, 50:750], (640, 720)))
        w.release()
        cls.info = probe(cls.video)
        cls.settings = Settings(device="cpu", person_weights=COCO, evidence_dir=cls.tmp / "ev",
                                fight_checkpoint=cls.tmp / "x", smoking_cnn_checkpoint=cls.tmp / "x",
                                cigarette_checkpoint=cls.tmp / "x")
        cls.det = ModelManager(cls.settings).get("person_yolo")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_before_and_after_closing(self):
        before = analyze(self.video, self.info, self.det, "cos", self.settings, self.tmp / "ev", ist(19, 30))
        self.assertEqual(before["events"], [])
        after = analyze(self.video, self.info, self.det, "cos", self.settings, self.tmp / "ev", ist(20, 15))
        self.assertEqual(len(after["events"]), 1)
        ev = after["events"][0]
        self.assertGreaterEqual(ev["persons_max_in_frame"], 1)
        self.assertGreaterEqual(ev["confidence"], 0.4)
        self.assertIn("Potential after-hours person detected in COS at 8:15 PM IST", ev["description"])
        self.assertTrue((self.tmp / "ev" / ev["evidence"][0]["image"]).is_file())


if __name__ == "__main__":
    unittest.main()
