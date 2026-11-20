"""Smoking analysis: SmokingCNN (frame classification) + cigarette YOLO (object detection).

Both signals are kept separate in the output:
  - CNN:  "does this frame look like smoking?"  → probability per sampled frame
  - YOLO: "is a cigarette-like OBJECT visible, and where?" → boxes per sampled frame
A cigarette detection alone is NOT proof of smoking. A POTENTIAL smoking event is created
when the configured rule holds on at least `smoking_min_frames` sampled frames that are no
more than `event_gap_s` apart:
    both   → CNN ≥ cnn_threshold AND a cigarette ≥ cigarette_confidence in the same frame (default)
    either → one of the two;   cnn → CNN only;   yolo → cigarette only
This fusion rule has NOT been validated on labelled data yet; it is a transparent heuristic.

Frames are sampled every `smoking_sample_interval_s` (default 0.5 s), never every frame.
The CNN runs batched on all sampled frames (it is tiny: 455k parameters); YOLO runs per sampled frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from inference.settings import Settings
from inference.video_io import VideoInfo, group_times, iter_frames, label, save_jpeg

LOG = logging.getLogger("inference.smoking")
CNN_BATCH = 32


def sample_indices(info: VideoInfo, interval_s: float, max_frames: int) -> list[int]:
    if info.frame_count <= 0:
        return []
    step = max(1, round(interval_s * info.fps))
    idx = list(range(0, info.frame_count, step))
    if len(idx) > max_frames:   # very long video: spread max_frames evenly instead
        idx = sorted({int(round(x)) for x in np.linspace(0, info.frame_count - 1, max_frames)})
    return idx


@torch.inference_mode()
def _cnn_probs(cnn: dict, rgb_frames: list[np.ndarray], device: str) -> list[float]:
    """Exactly the SmokingCNN evaluation preprocessing (dataset.build_transforms(train=False))."""
    model, transform = cnn["model"], cnn["transform"]
    out: list[float] = []
    for i in range(0, len(rgb_frames), CNN_BATCH):
        batch = torch.stack([transform(Image.fromarray(f)) for f in rgb_frames[i:i + CNN_BATCH]]).to(device)
        out += torch.sigmoid(model(batch).float()).squeeze(1).cpu().tolist()
    return out


def _frame_score(rule: str, cnn_p: float | None, cig: float, s: Settings) -> float | None:
    """Score of a positive frame under the rule, or None if the frame is not positive."""
    cnn_ok = cnn_p is not None and cnn_p >= s.smoking_cnn_threshold
    cig_ok = cig >= s.cigarette_confidence
    if rule == "both":
        return min(cnn_p, cig) if cnn_ok and cig_ok else None
    if rule == "either":
        return max(cnn_p or 0.0, cig) if cnn_ok or cig_ok else None
    if rule == "cnn":
        return cnn_p if cnn_ok else None
    return cig if cig_ok else None


def analyze(video: Path, info: VideoInfo, cnn: dict | None, yolo, device: str, settings: Settings,
            evidence_dir: Path) -> dict:
    rule = settings.smoking_rule
    if rule in ("both", "cnn") and cnn is None or rule in ("both", "yolo") and yolo is None:
        missing = [n for n, m in (("smoking CNN", cnn), ("cigarette YOLO", yolo)) if m is None]
        if rule == "both":
            rule = "cnn" if cnn is not None else "yolo"
        note = f"{' and '.join(missing)} unavailable; rule changed to '{rule}'"
    else:
        note = None
    if yolo is not None:
        yolo.conf = settings.cigarette_confidence   # shared detector: set the per-request threshold

    indices = sample_indices(info, settings.smoking_sample_interval_s, settings.smoking_max_frames)
    decoded, times, rgb, cig_boxes, frames_kept = [], [], [], [], {}
    for i, frame in iter_frames(video, set(indices)):
        decoded.append(i)
        times.append(i / info.fps)
        rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes = yolo.detect(frame) if yolo is not None else []
        cig_boxes.append(boxes)
        frames_kept[i] = frame if boxes else None  # keep raw frames only where something was seen
    cnn_p = _cnn_probs(cnn, rgb, device) if cnn is not None else [None] * len(rgb)
    del rgb

    per_frame = []
    for k, frame_idx in enumerate(decoded):
        cig = max((b["confidence"] for b in cig_boxes[k]), default=0.0)
        per_frame.append({"frame_index": frame_idx, "time_s": round(times[k], 2),
                          "cnn_smoking_probability": None if cnn_p[k] is None else round(cnn_p[k], 4),
                          "max_cigarette_confidence": round(cig, 4), "cigarette_boxes": cig_boxes[k],
                          "score": _frame_score(rule, cnn_p[k], cig, settings)})

    positives = [f for f in per_frame if f["score"] is not None]
    events = []
    for g in group_times([f["time_s"] for f in positives], settings.event_gap_s):
        fr = [positives[i] for i in g]
        if len(fr) < settings.smoking_min_frames:
            continue
        top = sorted(fr, key=lambda f: f["score"], reverse=True)[: settings.max_evidence_per_event]
        evidence = []
        for n, f in enumerate(top):
            image = None
            raw = frames_kept.get(f["frame_index"])
            if raw is None and n == 0:
                # positive by CNN only: re-read this single frame for the evidence picture
                raw = next((fr_ for _, fr_ in iter_frames(video, {f["frame_index"]})), None)
            if raw is not None:
                draw = raw.copy()
                for b in f["cigarette_boxes"]:
                    x1, y1, x2, y2 = map(int, b["bbox_xyxy"])
                    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(draw, f"cigarette {b['confidence']:.2f}", (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cnn_txt = "n/a" if f["cnn_smoking_probability"] is None else f"{f['cnn_smoking_probability']:.2f}"
                name = f"smoking_{len(events) + 1}_{f['frame_index']}.jpg"
                save_jpeg(label(draw, f"POTENTIAL SMOKING  CNN={cnn_txt}  cig={f['max_cigarette_confidence']:.2f}"
                                      f"  t={f['time_s']:.1f}s"), evidence_dir / name)
                image = name
            evidence.append({"time_s": f["time_s"], "frame_index": f["frame_index"], "confidence": round(f["score"], 4),
                             "image": image,
                             "details": {"cnn_smoking_probability": f["cnn_smoking_probability"],
                                         "max_cigarette_confidence": f["max_cigarette_confidence"],
                                         "cigarette_boxes": f["cigarette_boxes"][:5]}})
        cnn_vals = [f["cnn_smoking_probability"] for f in fr if f["cnn_smoking_probability"] is not None]
        events.append({
            "event_id": f"smoking-{fr[0]['frame_index']}",
            "type": "smoking",
            "start_s": fr[0]["time_s"], "end_s": fr[-1]["time_s"],
            "confidence": round(max(f["score"] for f in fr), 4),
            "description": (f"Possible smoking between {fr[0]['time_s']:.1f}s and {fr[-1]['time_s']:.1f}s: "
                            f"{len(fr)} sampled frames met rule '{rule}'"
                            + (f"; max CNN {max(cnn_vals):.2f}" if cnn_vals else "")
                            + f"; max cigarette {max(f['max_cigarette_confidence'] for f in fr):.2f}. "
                            "A visible cigarette alone does not prove smoking."),
            "evidence": evidence,
        })

    cnn_list = [f["cnn_smoking_probability"] for f in per_frame if f["cnn_smoking_probability"] is not None]
    return {
        "potential_incident": bool(events),
        "events": events,
        "rule": rule,
        "thresholds": {"cnn": settings.smoking_cnn_threshold, "cigarette": settings.cigarette_confidence,
                       "min_frames": settings.smoking_min_frames},
        "summary": {
            "frames_sampled": len(per_frame),
            "sample_interval_s": settings.smoking_sample_interval_s,
            "frames_cnn_positive": sum(1 for p in cnn_list if p >= settings.smoking_cnn_threshold),
            "frames_with_cigarette": sum(1 for f in per_frame if f["cigarette_boxes"]),
            "max_cnn_smoking_probability": max(cnn_list) if cnn_list else None,
            "mean_cnn_smoking_probability": round(float(np.mean(cnn_list)), 4) if cnn_list else None,
            "max_cigarette_confidence": max((f["max_cigarette_confidence"] for f in per_frame), default=0.0),
        },
        # Frames with any cigarette box (capped), so cigarette detections are visible separately.
        "detections": [{k: f[k] for k in ("time_s", "frame_index", "max_cigarette_confidence",
                                          "cnn_smoking_probability", "cigarette_boxes")}
                       for f in per_frame if f["cigarette_boxes"]][:50],
        "notes": [n for n in [note] if n],
    }
