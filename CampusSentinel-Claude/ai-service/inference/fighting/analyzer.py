"""Fighting analysis with the existing CNN-LSTM predictor.

The model was trained on 5-second RWF-2000 clips, each reduced to 16 uniformly sampled
frames. To keep that input distribution for longer uploads, the video is split into
consecutive windows of `fight_window_seconds` (default 5 s) and EACH window is
preprocessed exactly like training: 16 uniformly spaced frames → letterbox 224 → RGB uint8
(training.fighting.prepare_data.uniform_indices / preprocess_frame). Very long videos get at
most `fight_max_windows` windows spread evenly across the video.

Output is a POTENTIAL fight for review, never confirmed misconduct.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from inference.settings import Settings
from inference.video_io import VideoInfo, group_times, iter_frames, label, save_jpeg
from training.fighting.prepare_data import preprocess_frame, process_video, uniform_indices

LOG = logging.getLogger("inference.fighting")


def plan_windows(frame_count: int, fps: float, window_s: float, max_windows: int) -> list[tuple[int, int]]:
    """[(start, end_exclusive), ...] covering the video with ~window_s-second windows."""
    if frame_count <= 0:
        return []
    win = max(16, round(window_s * fps))
    if frame_count <= int(win * 1.5):
        return [(0, frame_count)]            # short video: one window = whole video
    starts = list(range(0, frame_count - win + 1, win))
    if frame_count - (starts[-1] + win) > win // 2:
        starts.append(frame_count - win)     # cover a long tail with a final (overlapping) window
    if len(starts) > max_windows:
        starts = sorted({int(round(x)) for x in np.linspace(0, frame_count - win, max_windows)})
    return [(s, s + win) for s in starts]


def analyze(video: Path, info: VideoInfo, predictor, settings: Settings, evidence_dir: Path) -> dict:
    threshold = settings.fight_threshold if settings.fight_threshold is not None else predictor.threshold
    t_frames, size = predictor.clip_shape[0], predictor.clip_shape[1]
    windows = plan_windows(info.frame_count, info.fps, settings.fight_window_seconds, settings.fight_max_windows)
    notes = []

    # One sequential pass collects the preprocessed frames for every window.
    needed = {}
    for w_idx, (s, e) in enumerate(windows):
        needed[w_idx] = [s + i for i in uniform_indices(e - s, t_frames)]
    wanted = {i for idx in needed.values() for i in idx}
    frames: dict[int, np.ndarray] = {}
    raw_mid: dict[int, np.ndarray] = {}
    mid_frames = {min(needed[w], key=lambda i: abs(i - (s + e) // 2)) for w, (s, e) in enumerate(windows)}
    for i, frame in iter_frames(video, wanted):
        frames[i] = preprocess_frame(frame, size)
        if i in mid_frames:
            raw_mid[i] = frame

    results = []
    for w_idx, (s, e) in enumerate(windows):
        idx = needed[w_idx]
        if not all(i in frames for i in idx):
            notes.append(f"window {s}-{e} skipped: frames could not be decoded")
            continue
        clip = np.stack([frames[i] for i in idx]).astype(np.uint8, copy=False)
        pred = predictor.predict_array(clip)
        results.append({"window": w_idx, "start_frame": s, "end_frame": e,
                        "start_s": round(s / info.fps, 2), "end_s": round(e / info.fps, 2),
                        "fight_probability": pred["fight_probability"],
                        "sampled_frames": idx})

    if not results:
        # Header frame count wrong or no window decodable: fall back to the original whole-video
        # preprocessing, which re-measures the real length and resamples.
        clip, vinfo = process_video(video, t_frames, size)
        pred = predictor.predict_array(clip)
        notes.append("window decoding failed; used whole-video sampling (prepare_data.process_video)")
        results = [{"window": 0, "start_frame": 0, "end_frame": vinfo.frame_count, "start_s": 0.0,
                    "end_s": round(vinfo.frame_count / info.fps, 2), "fight_probability": pred["fight_probability"],
                    "sampled_frames": vinfo.indices}]

    if info.frame_count and info.frame_count < t_frames:
        notes.append(f"very short video ({info.frame_count} frames): frames are repeated; result is unreliable")

    best = max(results, key=lambda r: r["fight_probability"])
    positive = [r for r in results if r["fight_probability"] >= threshold]

    # Neighbouring positive windows form one event.
    events = []
    for g in group_times([r["start_s"] for r in positive], settings.fight_window_seconds + 0.01):
        rs = [positive[i] for i in g]
        top = max(rs, key=lambda r: r["fight_probability"])
        evidence = []
        mid = min(top["sampled_frames"], key=lambda i: abs(i - (top["start_frame"] + top["end_frame"]) // 2))
        image = None
        if mid in raw_mid:
            name = f"fighting_{len(events) + 1}_{rs[0]['start_frame']}.jpg"
            save_jpeg(label(raw_mid[mid], f"POTENTIAL FIGHT  p={top['fight_probability']:.2f}  t={mid / info.fps:.1f}s"),
                      evidence_dir / name)
            image = name
        evidence.append({"time_s": round(mid / info.fps, 2), "frame_index": mid,
                         "confidence": top["fight_probability"], "image": image,
                         "details": {"window_start_s": top["start_s"], "window_end_s": top["end_s"]}})
        events.append({
            "event_id": f"fighting-{rs[0]['start_frame']}",
            "type": "fighting",
            "start_s": rs[0]["start_s"], "end_s": rs[-1]["end_s"],
            "confidence": top["fight_probability"],
            "description": (f"Possible fight between {rs[0]['start_s']:.1f}s and {rs[-1]['end_s']:.1f}s "
                            f"(max fight probability {top['fight_probability']:.2f} ≥ threshold {threshold})."),
            "evidence": evidence,
        })

    max_p = best["fight_probability"]
    return {
        "predicted_class": "Fight" if positive else "NonFight",
        "fight_probability": max_p,                     # highest window probability
        "nonfight_probability": round(1.0 - max_p, 4),
        "threshold": threshold,
        "potential_incident": bool(events),
        "events": events,
        "windows": [{k: v for k, v in r.items() if k != "sampled_frames"} for r in results],
        "sampling": {"window_seconds": settings.fight_window_seconds, "frames_per_window": t_frames,
                     "windows_analysed": len(results),
                     "frame_indices_first_window": results[0]["sampled_frames"]},
        "notes": notes,
    }
