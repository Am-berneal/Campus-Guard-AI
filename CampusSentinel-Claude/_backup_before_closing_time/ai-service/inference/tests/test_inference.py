"""Tests for the AI inference layer and the analysis API (synthetic data, CPU).

Models used here are RANDOMLY initialised stand-ins created in a temp folder, so the tests
check wiring, shapes, statuses and error handling — never accuracy.
The person-detector test needs a COCO YOLO file on disk and is skipped otherwise.

Run from the ai-service folder:
    python -m unittest discover -s inference\\tests -t . -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from inference.fighting.analyzer import plan_windows
from inference.model_manager import ModelManager
from inference.pipeline import analyze_video
from inference.settings import AI_SERVICE_ROOT, Settings
from inference.smoking.analyzer import _frame_score, sample_indices
from inference.video_io import VideoInfo, VideoReadError, group_times, probe

COCO_CANDIDATES = [AI_SERVICE_ROOT / "models/person/yolo26n.pt", AI_SERVICE_ROOT / "models/smoking/yolo/base/yolo26n.pt"]
COCO = next((p for p in COCO_CANDIDATES if p.is_file()), None)


def write_video(path: Path, n: int = 90, size=(320, 240), fps: float = 30.0) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n):
        frame = np.full((size[1], size[0], 3), (i * 2) % 255, np.uint8)
        cv2.rectangle(frame, (20 + i, 40), (60 + i, 200), (0, 0, 255), -1)
        w.write(frame)
    w.release()
    return path


def make_models(root: Path) -> dict:
    """Random-weight checkpoints in the real formats (fighting + smoking CNN)."""
    from training.fighting.checkpoint import save_checkpoint
    from training.fighting.config import ModelConfig
    from training.fighting.model import CNNLSTM
    from training.smoking.cnn.checkpoint import build_metadata, save_best
    from training.smoking.cnn.model import SmokingCNN

    torch.manual_seed(0)
    fight = root / "fight.pt"
    save_checkpoint(fight, CNNLSTM(ModelConfig(pretrained=False, hidden_size=32)), None, 1, 0.6, {"threshold": 0.5})
    cnn = root / "cnn.pt"
    save_best(cnn, SmokingCNN(), build_metadata(1, {"val_loss": 0.6}, {}))
    return {"fight": fight, "cnn": cnn}


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cs_inf_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def settings(self, **kw) -> Settings:
        base = Settings(device="cpu", upload_dir=self.tmp / "uploads", results_dir=self.tmp / "results",
                        evidence_dir=self.tmp / "evidence", fight_checkpoint=self.tmp / "missing_fight.pt",
                        smoking_cnn_checkpoint=self.tmp / "missing_cnn.pt", cigarette_checkpoint=self.tmp / "missing_yolo.pt",
                        person_weights=self.tmp / "missing_person.pt", zone_config_dir=self.tmp / "zones")
        return replace(base, **kw)


# --------------------------------------------------------------------------- pure helpers
class TestHelpers(unittest.TestCase):
    def test_plan_windows(self):
        self.assertEqual(plan_windows(150, 30, 5, 24), [(0, 150)])            # one RWF-sized clip
        self.assertEqual(plan_windows(60, 30, 5, 24), [(0, 60)])              # short video
        w = plan_windows(600, 30, 5, 24)                                      # 20 s → four 5 s windows
        self.assertEqual(w, [(0, 150), (150, 300), (300, 450), (450, 600)])
        long = plan_windows(30 * 600, 30, 5, 24)                              # 10 min → capped
        self.assertEqual(len(long), 24)
        self.assertEqual(long[0][0], 0)
        self.assertEqual(long[-1][1], 30 * 600)
        self.assertEqual(plan_windows(0, 30, 5, 24), [])

    def test_sample_indices(self):
        info = VideoInfo(30.0, 300, 320, 240)
        self.assertEqual(sample_indices(info, 0.5, 240)[:3], [0, 15, 30])
        self.assertEqual(len(sample_indices(info, 0.5, 240)), 20)
        self.assertEqual(len(sample_indices(VideoInfo(30.0, 30000, 320, 240), 0.5, 100)), 100)

    def test_fusion_rules(self):
        s = Settings(smoking_cnn_threshold=0.5, cigarette_confidence=0.35)
        self.assertEqual(_frame_score("both", 0.8, 0.6, s), 0.6)
        self.assertIsNone(_frame_score("both", 0.8, 0.1, s))                  # cigarette alone missing
        self.assertIsNone(_frame_score("both", 0.2, 0.9, s))                  # cigarette alone ≠ smoking
        self.assertEqual(_frame_score("either", 0.2, 0.9, s), 0.9)
        self.assertEqual(_frame_score("cnn", 0.7, 0.0, s), 0.7)
        self.assertIsNone(_frame_score("yolo", 0.99, 0.2, s))

    def test_group_times(self):
        self.assertEqual(group_times([0, 0.5, 1.0, 5.0, 5.5], 2.0), [[0, 1, 2], [3, 4]])
        self.assertEqual(group_times([], 2.0), [])

    def test_settings_overrides(self):
        s = Settings()
        self.assertEqual(s.with_overrides({"smoking_rule": "either"}).smoking_rule, "either")
        with self.assertRaises(ValueError):
            s.with_overrides({"smoking_rule": "magic"})
        with self.assertRaises(ValueError):
            s.with_overrides({"fight_checkpoint": "/etc/passwd"})            # paths are not overridable
        with self.assertRaises(ValueError):
            s.with_overrides({"fight_threshold": 1.5})


# --------------------------------------------------------------------------- manager + pipeline
class TestManagerAndPipeline(TempCase):
    def test_missing_models_are_unavailable_not_downloaded(self):
        m = ModelManager(self.settings())
        m.preload()
        st = m.status()["models"]
        self.assertTrue(all(v["status"] == "unavailable" for v in st.values()), st)
        self.assertFalse((self.tmp / "missing_person.pt").exists())          # nothing was downloaded
        self.assertEqual(m.device, "cpu")

    def test_pipeline_with_available_models(self):
        models = make_models(self.tmp)
        m = ModelManager(self.settings(fight_checkpoint=models["fight"], smoking_cnn_checkpoint=models["cnn"]))
        video = write_video(self.tmp / "v.avi", 90)
        r = analyze_video(m, video, "abc123", "cos", "COS")
        a = r["analysis"]
        self.assertEqual(a["fighting"]["status"], "completed")
        self.assertAlmostEqual(a["fighting"]["fight_probability"] + a["fighting"]["nonfight_probability"], 1, 3)
        self.assertEqual(a["fighting"]["sampling"]["frames_per_window"], 16)
        self.assertEqual(a["smoking"]["status"], "completed")
        self.assertEqual(a["smoking"]["rule"], "cnn")                       # YOLO missing → CNN-only, noted
        self.assertTrue(a["smoking"]["notes"])
        self.assertEqual(a["smoking"]["summary"]["frames_sampled"], 6)      # 3 s at 0.5 s steps
        self.assertEqual(a["restricted_zone"]["status"], "unavailable")
        self.assertEqual(r["status"], "partial")
        self.assertEqual(r["processing"]["device"], "cpu")
        json.dumps(r)                                                       # JSON-serialisable

    def test_module_failure_is_isolated(self):
        models = make_models(self.tmp)
        m = ModelManager(self.settings(fight_checkpoint=models["fight"], smoking_cnn_checkpoint=models["cnn"]))
        m.get("fighting").predict_array = lambda clip: (_ for _ in ()).throw(RuntimeError("boom"))
        r = analyze_video(m, write_video(self.tmp / "v.avi", 60), "abc", "cos", "COS")
        self.assertEqual(r["analysis"]["fighting"]["status"], "failed")
        self.assertIn("boom", r["analysis"]["fighting"]["error"])
        self.assertEqual(r["analysis"]["smoking"]["status"], "completed")   # other module still ran

    def test_unreadable_video(self):
        bad = self.tmp / "bad.mp4"
        bad.write_bytes(b"not a video" * 100)
        with self.assertRaises(VideoReadError):
            probe(bad)

    @unittest.skipUnless(COCO, "no COCO YOLO weights on disk")
    def test_restricted_zone_with_person_detector(self):
        zones = self.tmp / "zones"
        zones.mkdir()
        shutil.copy(AI_SERVICE_ROOT / "training/restricted_zone/configs/cameras/cos.yaml", zones / "cos.yaml")
        m = ModelManager(self.settings(person_weights=COCO))
        r = analyze_video(m, write_video(self.tmp / "v.avi", 60), "abc", "cos", "COS", modules=("restricted_zone",))
        self.assertEqual(r["analysis"]["restricted_zone"]["status"], "completed")
        r2 = analyze_video(m, write_video(self.tmp / "w.avi", 30), "abc", "library", "Outside Library",
                           modules=("restricted_zone",))
        self.assertEqual(r2["analysis"]["restricted_zone"]["status"], "not_configured")


# --------------------------------------------------------------------------- API
class TestAnalysisApi(TempCase):
    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import inference.model_manager as mm
        from api.routers import analysis

        models = make_models(self.tmp)
        self._old = mm._MANAGER
        mm._MANAGER = ModelManager(self.settings(fight_checkpoint=models["fight"], smoking_cnn_checkpoint=models["cnn"]))
        (self.tmp / "uploads").mkdir()
        write_video(self.tmp / "uploads" / "vid001.avi", 60)
        app = FastAPI()
        app.include_router(analysis.router)
        self.client = TestClient(app)

    def tearDown(self):
        import inference.model_manager as mm
        mm._MANAGER = self._old
        super().tearDown()

    def test_analyze_cache_and_errors(self):
        r = self.client.post("/videos/vid001/analyze", json={"camera_id": "cos"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual((body["upload_id"], body["camera_name"], body["cached"]), ("vid001", "COS", False))
        self.assertTrue(self.client.post("/videos/vid001/analyze", json={"camera_id": "cos"}).json()["cached"])
        self.assertEqual(self.client.post("/videos/nope/analyze", json={"camera_id": "cos"}).status_code, 404)
        self.assertEqual(self.client.post("/videos/a.b/analyze", json={"camera_id": "cos"}).status_code, 400)
        self.assertEqual(self.client.post("/videos/vid001/analyze", json={"camera_id": "mars"}).status_code, 422)
        self.assertEqual(self.client.post("/videos/vid001/analyze", json={}).status_code, 422)
        st = self.client.get("/models/status").json()
        self.assertEqual(st["models"]["fighting"]["status"], "loaded")
        self.assertEqual(st["models"]["person_yolo"]["status"], "unavailable")
        self.assertEqual(self.client.get("/evidence/vid001/..%2Fx.jpg").status_code, 404)


if __name__ == "__main__":
    unittest.main()
