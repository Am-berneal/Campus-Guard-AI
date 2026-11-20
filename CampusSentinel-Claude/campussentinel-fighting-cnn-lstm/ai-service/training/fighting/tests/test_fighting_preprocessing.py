"""Unit tests for fighting-video preprocessing, using small synthetic AVI files.

Run from the ai-service folder:
    python -m unittest discover -s training\\fighting\\tests -t . -v
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from training.fighting import prepare_data as pdm
from training.fighting.dataset import IMAGENET_MEAN, IMAGENET_STD, FightingVideoDataset


def write_avi(path: Path, n_frames: int = 150, size: tuple[int, int] = (320, 240), fps: float = 30.0,
              bgr: tuple[int, int, int] | None = None) -> None:
    """Synthetic video. Each frame's grey level equals its index (so sampling can be verified),
    unless a fixed BGR colour is given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert writer.isOpened(), "OpenCV could not create a test AVI"
    for i in range(n_frames):
        frame = np.full((h, w, 3), bgr if bgr else (i, i, i), np.uint8)
        writer.write(frame)
    writer.release()


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cs_fight_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# --------------------------------------------------------------------------- pure functions
class TestSampling(unittest.TestCase):
    def test_150_frames_16_samples(self):
        idx = pdm.uniform_indices(150, 16)
        self.assertEqual(len(idx), 16)
        self.assertEqual((idx[0], idx[-1]), (0, 149))            # beginning and end included
        self.assertEqual(idx, sorted(idx))
        gaps = np.diff(idx)
        self.assertTrue(set(gaps) <= {9, 10, 11})                 # evenly spaced (~9.93)

    def test_short_video_repeats(self):
        idx = pdm.uniform_indices(5, 16)
        self.assertEqual(len(idx), 16)
        self.assertEqual((min(idx), max(idx)), (0, 4))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            pdm.uniform_indices(0, 16)


class TestLetterbox(unittest.TestCase):
    def test_landscape_padding_and_aspect(self):
        frame = np.full((240, 320, 3), 200, np.uint8)            # 4:3
        out = pdm.letterbox(frame, 224)
        self.assertEqual(out.shape, (224, 224, 3))
        self.assertEqual(out.dtype, np.uint8)
        content_rows = np.where(out[:, 112, 0] > 0)[0]
        self.assertEqual(len(content_rows), 168)                 # 240 * 224/320 = 168 (not stretched)
        self.assertEqual(out[0, 112, 0], 0)                      # black padding top
        self.assertEqual(out[-1, 112, 0], 0)                     # and bottom
        self.assertEqual(out[112, 0, 0], 200)                    # full width used

    def test_wide_hd_and_portrait(self):
        wide = pdm.letterbox(np.full((720, 1280, 3), 100, np.uint8), 224)
        self.assertEqual(len(np.where(wide[:, 112, 0] > 0)[0]), 126)   # 720 * 224/1280
        tall = pdm.letterbox(np.full((320, 240, 3), 100, np.uint8), 224)
        self.assertEqual(len(np.where(tall[112, :, 0] > 0)[0]), 168)   # padding left/right

    def test_upscaling_small_frame(self):
        self.assertEqual(pdm.letterbox(np.full((60, 80, 3), 50, np.uint8), 224).shape, (224, 224, 3))

    def test_bgr_to_rgb(self):
        blue_bgr = np.zeros((240, 320, 3), np.uint8)
        blue_bgr[..., 0] = 255                                   # pure blue in OpenCV's BGR
        out = pdm.preprocess_frame(blue_bgr, 224)
        self.assertEqual(tuple(out[112, 112]), (0, 0, 255))      # blue is now the LAST (B) channel of RGB


# --------------------------------------------------------------------------- video processing
class TestProcessVideo(TempDirCase):
    def test_shape_dtype_and_sampled_frames(self):
        path = self.tmp / "v.avi"
        write_avi(path, 150, (320, 240))
        clip, info = pdm.process_video(path, 16, 224)
        self.assertEqual(clip.shape, (16, 224, 224, 3))
        self.assertEqual(clip.dtype, np.uint8)
        self.assertEqual(info.frame_count, 150)
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertEqual(info.indices, pdm.uniform_indices(150, 16))
        # grey level of each sampled frame ≈ its source index (MJPEG is lossy → tolerance)
        centre = clip[:, 112, 112, 0].astype(int)
        self.assertTrue(np.all(np.abs(centre - np.array(info.indices)) <= 4), (centre, info.indices))

    def test_colour_order_through_video(self):
        path = self.tmp / "red.avi"
        write_avi(path, 20, (320, 240), bgr=(0, 0, 255))         # red in BGR
        clip, _ = pdm.process_video(path, 16, 224)
        r, g, b = clip[0, 112, 112]
        self.assertGreater(r, 200)
        self.assertLess(b, 50)

    def test_truncated_video_resampled_with_warning(self):
        path = self.tmp / "cut.avi"
        write_avi(path, 150, (320, 240))
        data = path.read_bytes()
        path.write_bytes(data[: int(len(data) * 0.6)])          # header still says 150 frames
        clip, info = pdm.process_video(path, 16, 224)
        self.assertEqual(clip.shape, (16, 224, 224, 3))
        self.assertLess(info.frame_count, 150)
        self.assertEqual(info.indices, pdm.uniform_indices(info.frame_count, 16))
        self.assertIn("decoded", info.note)

    def test_corrupt_and_empty_files(self):
        bad = self.tmp / "bad.avi"
        bad.write_bytes(b"this is not a video" * 100)
        with self.assertRaises(pdm.VideoError):
            pdm.process_video(bad, 16, 224)
        empty = self.tmp / "empty.avi"
        empty.write_bytes(b"")
        with self.assertRaises(pdm.VideoError):
            pdm.process_video(empty, 16, 224)


# --------------------------------------------------------------------------- full pipeline
class TestPipeline(TempDirCase):
    def make_dataset(self):
        src = self.tmp / "fighting"
        for split, n in (("train", 3), ("val", 2)):
            for cls in ("Fight", "NonFight"):
                for i in range(n):
                    size = [(320, 240), (640, 360), (240, 320)][i % 3]
                    write_avi(src / split / cls / f"{cls.lower()}_{i}.avi", 40 + i * 10, size)
        (src / "train" / "Fight" / "broken.avi").write_bytes(b"\x00garbage" * 50)
        (src / "train" / "Fight" / "notes.txt").write_text("not a video")
        return src

    def run_pipeline(self, src, out, **kw):
        return pdm.run(src, out, 16, 224, **kw)

    def read_manifest(self, out):
        with (out / "manifest.csv").open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_end_to_end_manifest_report_resume_force(self):
        src, out = self.make_dataset(), self.tmp / "processed"
        before = {p: p.stat().st_mtime_ns for p in src.rglob("*") if p.is_file()}

        rep = self.run_pipeline(src, out)
        pc = rep["this_run"]["per_class"]
        self.assertEqual(pc["train/Fight"], {"processed": 3, "skipped": 0, "failed": 1})
        self.assertEqual(pc["val/NonFight"]["processed"], 2)
        self.assertEqual(len(rep["ignored_non_video_files"]), 1)

        rows = self.read_manifest(out)
        self.assertEqual(list(rows[0].keys()), pdm.MANIFEST_FIELDS)
        self.assertEqual(len(rows), 11)                           # 10 good + 1 broken
        failed = [r for r in rows if r["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("broken.avi", failed[0]["source_path"])
        self.assertTrue(failed[0]["error"])
        for r in rows:
            self.assertEqual(int(r["label"]), {"Fight": 1, "NonFight": 0}[r["class_name"]])
            if r["status"] == "ok":
                self.assertEqual(r["output_shape"], "16x224x224x3")
                self.assertEqual(len(r["frame_indices"].split()), 16)
                self.assertTrue((out / r["npy_path"]).is_file())
        self.assertTrue((out / "train/Fight/fight_0.npy").is_file())   # deterministic name
        self.assertTrue(json.loads((out / "preprocessing_report.json").read_text())["failures_this_run"])

        # Resume: nothing valid is reprocessed, files untouched.
        npy = out / "train/Fight/fight_0.npy"
        mtime = npy.stat().st_mtime_ns
        rep2 = self.run_pipeline(src, out)
        self.assertEqual(rep2["this_run"]["per_class"]["train/Fight"], {"processed": 0, "skipped": 3, "failed": 1})
        self.assertEqual(npy.stat().st_mtime_ns, mtime)

        # A damaged output file is detected and regenerated on resume.
        npy.write_bytes(b"truncated")
        rep3 = self.run_pipeline(src, out)
        self.assertEqual(rep3["this_run"]["per_class"]["train/Fight"]["processed"], 1)
        self.assertEqual(np.load(npy).shape, (16, 224, 224, 3))

        # Lost manifest is rebuilt from valid files without reprocessing.
        (out / "manifest.csv").unlink()
        rep4 = self.run_pipeline(src, out)
        self.assertEqual(rep4["this_run"]["per_class"]["val/Fight"], {"processed": 0, "skipped": 2, "failed": 0})
        self.assertEqual(len(self.read_manifest(out)), 11)

        # --force regenerates.
        rep5 = self.run_pipeline(src, out, force=True)
        self.assertEqual(rep5["this_run"]["per_class"]["train/NonFight"]["processed"], 3)

        # Source dataset untouched (no files changed, none added).
        after = {p: p.stat().st_mtime_ns for p in src.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_limit_per_class(self):
        src, out = self.make_dataset(), self.tmp / "processed"
        rep = self.run_pipeline(src, out, limit=2)
        for key, c in rep["this_run"]["per_class"].items():
            self.assertLessEqual(c["processed"] + c["failed"], 2, key)

    def test_missing_directory_reported(self):
        src, out = self.make_dataset(), self.tmp / "processed"
        shutil.rmtree(src / "val" / "NonFight")
        rep = self.run_pipeline(src, out)
        self.assertEqual(len(rep["missing_directories"]), 1)
        self.assertEqual(rep["this_run"]["per_class"]["train/NonFight"]["processed"], 3)

    def test_output_inside_source_rejected(self):
        src = self.make_dataset()
        with self.assertRaises(ValueError):
            self.run_pipeline(src, src / "processed")

    def test_dataset_loader(self):
        src, out = self.make_dataset(), self.tmp / "processed"
        self.run_pipeline(src, out)
        ds = FightingVideoDataset(out / "manifest.csv", split="train")
        self.assertEqual(len(ds), 6)                              # broken video excluded
        self.assertEqual(ds.class_counts(), {"NonFight": 3, "Fight": 3})
        video, label = ds[0]
        self.assertEqual(tuple(video.shape), (16, 3, 224, 224))
        self.assertEqual(video.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.long)
        self.assertIn(int(label), (0, 1))
        # normalisation: a black padding pixel (0) maps to -mean/std
        raw = ds.load_uint8(0)
        self.assertEqual(raw[0, 0, 0, 0], 0)
        self.assertAlmostEqual(float(video[0, 0, 0, 0]), -IMAGENET_MEAN[0] / IMAGENET_STD[0], places=4)
        raw_ds = FightingVideoDataset(out / "manifest.csv", split="val", normalize=False)
        v, _ = raw_ds[0]
        self.assertGreaterEqual(float(v.min()), 0.0)
        self.assertLessEqual(float(v.max()), 1.0)
        self.assertEqual(len(raw_ds), 4)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
        xb, yb = next(iter(loader))
        self.assertEqual(tuple(xb.shape), (2, 16, 3, 224, 224))
        self.assertEqual(tuple(yb.shape), (2,))

    def test_dataset_errors(self):
        src, out = self.make_dataset(), self.tmp / "processed"
        self.run_pipeline(src, out)
        ds = FightingVideoDataset(out / "manifest.csv", split="train")
        ds.items[0][0].unlink()
        with self.assertRaises(FileNotFoundError):
            ds[0]
        np.save(ds.items[1][0], np.zeros((8, 224, 224, 3), np.uint8))
        with self.assertRaises(ValueError):
            ds[1]
        with self.assertRaises(ValueError):
            FightingVideoDataset(out / "manifest.csv", split="test")
        with self.assertRaises(FileNotFoundError):
            FightingVideoDataset(self.tmp / "nope.csv", split="train")


if __name__ == "__main__":
    unittest.main()
