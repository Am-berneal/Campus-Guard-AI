"""Unit tests for the CNN-LSTM fight classifier (synthetic data, CPU; optional CUDA test).

ResNet18 is built with pretrained=False here so tests run offline. Loading the
ImageNet weights is exercised only by real training.

Run from the ai-service folder:
    python -m unittest discover -s training\\fighting\\tests -t . -v
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.fighting import evaluate as eval_mod
from training.fighting import prepare_data as pdm
from training.fighting import train as train_mod
from training.fighting import verify_checkpoint as verify_mod
from training.fighting.checkpoint import load_checkpoint, save_checkpoint
from training.fighting.config import ModelConfig
from training.fighting.dataset import FightingVideoDataset
from training.fighting.metrics import binary_metrics, classification_report, roc_auc
from training.fighting.model import CNNLSTM, count_parameters
from training.fighting.predict import FightPredictor
from training.fighting.tests.test_fighting_preprocessing import write_avi


def small_model(**kw) -> CNNLSTM:
    torch.manual_seed(0)
    return CNNLSTM(ModelConfig(pretrained=False, hidden_size=32, **kw))


def make_processed(tmp: Path, n_per_class: int = 2) -> Path:
    """Synthetic AVIs → real prepare_data pipeline → processed folder with manifest.csv."""
    src = tmp / "raw"
    for split in ("train", "val"):
        for cls in ("Fight", "NonFight"):
            for i in range(n_per_class):
                write_avi(src / split / cls / f"{cls}_{i}.avi", 30, (160, 120),
                          bgr=(40 * i, 200 if cls == "Fight" else 20, 90))
    out = tmp / "processed"
    pdm.run(src, out, 16, 224)
    return out


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cs_cnnlstm_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# --------------------------------------------------------------------------- model
class TestModel(unittest.TestCase):
    def test_initialisation_and_frozen_backbone(self):
        m = small_model()
        total, trainable = count_parameters(m)
        temporal = sum(p.numel() for p in m.temporal_parameters())
        self.assertEqual(trainable, temporal)                         # only LSTM + head train
        self.assertGreater(total, 11_000_000)                         # ResNet18 ≈ 11.2 M inside
        self.assertFalse(any(isinstance(x, nn.Sigmoid) for x in m.modules()))

    def test_layer4_option(self):
        m = small_model(train_layer4=True)
        self.assertTrue(all(p.requires_grad for p in m.layer4.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.stem.parameters()))

    def test_forward_shape_and_finite(self):
        m = small_model().eval()
        with torch.no_grad():
            out = m(torch.randn(2, 16, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (2, 1))
        self.assertTrue(torch.isfinite(out).all())

    def test_batched_reshape_matches_per_clip(self):
        """(B,T,C,H,W)→(B·T,…)→(B,T,512) must keep clip/frame order: batch result == one clip at a time."""
        m = small_model().eval()
        x = torch.randn(3, 4, 3, 64, 64)
        with torch.no_grad():
            batched = m(x)
            single = torch.cat([m(x[i:i + 1]) for i in range(3)])
        self.assertTrue(torch.allclose(batched, single, atol=1e-5))

    def test_frozen_backbone_not_updated(self):
        m = small_model()
        m.train()
        self.assertFalse(m.stem.training)                             # frozen parts stay in eval mode
        bn = m.stem[1]
        before = bn.running_mean.clone()
        out = m(torch.randn(2, 4, 3, 64, 64))
        nn.BCEWithLogitsLoss()(out, torch.ones(2, 1)).backward()
        self.assertTrue(torch.equal(before, bn.running_mean))         # BatchNorm statistics unchanged
        self.assertTrue(all(p.grad is None for p in m.stem.parameters()))
        self.assertTrue(all(p.grad is not None for p in m.lstm.parameters()))

    def test_bad_input_shape(self):
        with self.assertRaises(ValueError):
            small_model()(torch.randn(16, 3, 64, 64))

    def test_multilayer_lstm(self):
        m = CNNLSTM(ModelConfig(pretrained=False, hidden_size=16, num_layers=2)).eval()
        self.assertAlmostEqual(m.lstm.dropout, 0.3)
        with torch.no_grad():
            self.assertEqual(tuple(m(torch.randn(1, 4, 3, 64, 64)).shape), (1, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gpu_forward(self):
        m = small_model().cuda().eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = m(torch.randn(2, 16, 3, 224, 224, device="cuda"))
        self.assertEqual(tuple(out.shape), (2, 1))
        self.assertTrue(torch.isfinite(out.float()).all())


# --------------------------------------------------------------------------- metrics
class TestMetrics(unittest.TestCase):
    def test_auc_and_threshold(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        self.assertEqual(roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        self.assertIsNone(roc_auc(np.array([1, 1]), np.array([0.2, 0.3])))
        m = binary_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]), threshold=0.5)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (1, 1, 1, 1))
        m2 = binary_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]), threshold=0.3)
        self.assertEqual(m2["tp"], 2)
        self.assertIn("Fight", classification_report(m))


# --------------------------------------------------------------------------- data + pipelines
class TestPipelines(TempDirCase):
    def test_dataset_loading_and_labels(self):
        root = make_processed(self.tmp)
        ds = FightingVideoDataset(root / "manifest.csv", "train")
        self.assertEqual(ds.class_counts(), {"NonFight": 2, "Fight": 2})
        for path, label in ds.items:
            self.assertEqual(label, 1 if path.parent.name == "Fight" else 0)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (16, 3, 224, 224))
        self.assertEqual(y.dtype, torch.long)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)))
        self.assertEqual(tuple(batch[0].shape), (2, 16, 3, 224, 224))
        self.assertEqual(tuple(batch[1].shape), (2,))

    def test_checkpoint_roundtrip(self):
        m = small_model().eval()
        opt = torch.optim.AdamW(m.temporal_parameters(), lr=1e-3)
        path = self.tmp / "ck" / "best.pt"
        save_checkpoint(path, m, opt, epoch=3, best_val_loss=0.42, train_config={"threshold": 0.5})
        m2, meta = load_checkpoint(path, "cpu")                        # uses weights_only=True
        self.assertEqual((meta["epoch"], meta["best_val_loss"]), (3, 0.42))
        self.assertEqual(meta["class_mapping"], {"0": "NonFight", "1": "Fight"})
        self.assertEqual(ModelConfig.from_dict(meta["model_config"]).hidden_size, 32)
        x = torch.randn(1, 4, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(x), m2(x), atol=1e-6))
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.tmp / "missing.pt")

    def test_predictor(self):
        root = make_processed(self.tmp)
        path = self.tmp / "ck" / "best.pt"
        save_checkpoint(path, small_model(), None, 1, 0.7, {"threshold": 0.5})
        pred = FightPredictor(path, device="cpu")
        clip = next((root / "val" / "Fight").glob("*.npy"))
        r = pred.predict_npy(clip)
        self.assertEqual(set(r), {"predicted_class", "fight_probability", "nonfight_probability", "threshold"})
        self.assertAlmostEqual(r["fight_probability"] + r["nonfight_probability"], 1.0, places=3)
        self.assertEqual(r["predicted_class"], "Fight" if r["fight_probability"] >= 0.5 else "NonFight")
        with self.assertRaises(ValueError):
            pred.predict_array(np.zeros((8, 224, 224, 3), np.uint8))
        with self.assertRaises(ValueError):
            pred.predict_array(np.zeros((16, 224, 224, 3), np.float32))
        # same tensor as the Dataset produces
        ds = FightingVideoDataset(root / "manifest.csv", "val")
        i = [p for p, _ in ds.items].index(clip)
        self.assertTrue(torch.allclose(pred._to_tensor(np.load(clip))[0], ds[i][0], atol=1e-6))
        rv = pred.predict_video(self.tmp / "raw" / "val" / "Fight" / "Fight_0.avi")
        self.assertEqual(len(rv["frame_indices"]), 16)
        always_fight = FightPredictor(path, device="cpu", threshold=1e-6).predict_npy(clip)
        self.assertEqual(always_fight["predicted_class"], "Fight")

    def test_train_resume_evaluate_verify(self):
        root = make_processed(self.tmp)
        ck, run = self.tmp / "ck", self.tmp / "run"
        base = ["--data-root", str(root), "--no-pretrained", "--hidden-size", "16", "--batch-size", "2",
                "--checkpoint-dir", str(ck), "--run-dir", str(run), "--device", "cpu"]
        self.assertEqual(train_mod.main(base + ["--epochs", "2"]), 0)
        for f in ("best.pt", "last.pt", "best.json"):
            self.assertTrue((ck / f).is_file(), f)
        for f in ("history.csv", "loss.png", "accuracy.png", "train.log"):
            self.assertTrue((run / f).is_file(), f)
        self.assertIn("VALIDATION", json.loads((ck / "best.json").read_text())["note"])
        # resume continues from epoch 3
        self.assertEqual(train_mod.main(base + ["--epochs", "3", "--resume"]), 0)
        last = torch.load(ck / "last.pt", weights_only=True)
        self.assertEqual(last["epoch"], 3)
        self.assertEqual(len(last["history"]), 3)
        # evaluation on validation
        out = self.tmp / "eval"
        self.assertEqual(eval_mod.main(["--checkpoint", str(ck / "best.pt"), "--data-root", str(root),
                                        "--device", "cpu", "--out-dir", str(out)]), 0)
        res = json.loads((out / "val_metrics.json").read_text())
        self.assertEqual(res["split"], "validation")
        self.assertEqual(res["clips"], 4)
        for f in ("val_confusion_matrix.png", "val_classification_report.txt", "val_predictions.csv"):
            self.assertTrue((out / f).is_file(), f)
        # checkpoint verification utility
        self.assertEqual(verify_mod.main(["--checkpoint", str(ck / "best.pt"), "--device", "cpu"]), 0)

    def test_missing_data_root(self):
        self.assertEqual(train_mod.main(["--data-root", str(self.tmp / "nope"), "--run-dir", str(self.tmp / "r")]), 1)


if __name__ == "__main__":
    unittest.main()
