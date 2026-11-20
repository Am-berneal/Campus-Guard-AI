# Restricted-zone / restricted-time person detection

Flags **potential** violations when a tracked person stands inside a configured zone while that zone's schedule marks it restricted. Every incident has `status: pending_review`. The module takes no action by itself and does not identify anyone.

**No training.** The person detector is a **pretrained** Ultralytics model (COCO class 0 = person). Nothing in this module is trained by us.

## Data flow

```
video frame
 → PersonDetector.track()     YOLO26n person detection + ByteTrack → boxes, confidence, track IDs
 → bottom_center()            ground point = middle of the box's bottom edge (≈ feet), normalised 0..1
 → Zone.contains()            OpenCV point-in-polygon (edge counts as inside)
 → Zone.is_restricted_at()    schedule evaluated in the configured campus time zone
 → RuleEngine.update()        dwell time + cooldown → potential incident dicts
```

| File | Role |
|---|---|
| `config.py` | Loads and validates the YAML config (`ConfigError` names the zone and field) |
| `schedule.py` | Time windows, weekdays, overnight windows, precedence |
| `zones.py` | `Zone` (polygon + schedule), `bottom_center()` |
| `detector.py` | `PersonDetector` (Ultralytics wrapper) and `parse_boxes()` |
| `rules.py` | `RuleEngine`: entry / dwell / cooldown logic, incident records |
| `demo_video.py` | Video or webcam demo: annotated MP4, `incidents.json`, evidence frames |
| `pick_zone.py` | Click a polygon on a frame and print normalised coordinates |
| `configs/zones.sample.yaml` | Sample config with **two fictional zones** |
| `trackers/bytetrack_campus.yaml` | ByteTrack settings (Ultralytics defaults, `track_buffer` 30 → 60) |
| `tests/test_restricted_zone.py` | 25 synthetic tests (no model, video or GPU needed) |

For the later FastAPI service, reuse `load_config` → `PersonDetector` → `RuleEngine` exactly as `demo_video.py` does. Use one detector instance per camera stream, because tracker state is per stream.

## Model and tracker

| Item | Choice | Reason |
|---|---|---|
| Ultralytics | 8.4.212 (pinned in `ai-service/requirements.txt`) | Latest release when this module was written |
| Weights | `yolo26n.pt`, downloaded automatically to `ai-service/models/person/` on first run (~5.3 MB) | Nano model of the newest detection family in Ultralytics 8.4.x; small VRAM use |
| Precision | float16 on CUDA (`quantize=16`), float32 on CPU | Halves activation memory on the GPU |
| Tracker | ByteTrack (`lap` package required) | Built into Ultralytics. Fast, and needs no extra appearance model, so no extra VRAM |
| Lost-track buffer | 60 frames (≈2 s at 30 fps) | Keeps an ID through short occlusions. Higher values increase the risk of ID swaps |

Licence: Ultralytics code and weights are AGPL-3.0. That's fine for this academic project; any commercial deployment would need a licence review.

## Configuration format (`configs/zones.sample.yaml`)

```yaml
campus_timezone: Asia/Kolkata        # required; the computer's own time zone is never used
camera_id: demo-camera-01
detector: {weights: models/person/yolo26n.pt, confidence: 0.40, iou: 0.50, image_size: 640, device: auto, half: true}
tracker:  {config: training/restricted_zone/trackers/bytetrack_campus.yaml}
incidents: {min_dwell_seconds: 1.0, cooldown_seconds: 300, save_evidence_frames: true}
zones:
  - id: zone-a-library-hall            # unique
    name: Library Hall A (fictional)
    status: restricted                 # restricted → schedule decides | allowed → never an incident
    polygon: [[0.05, 0.55], [0.45, 0.50], [0.50, 0.98], [0.02, 0.98]]   # normalised [x, y]
    schedule:
      default: restricted              # status outside all windows
      allowed:    [{days: [mon, tue, wed, thu, fri], start: "06:00", end: "20:00"}]
      restricted: []                   # optional
```

Schedule rules:
- **Precedence:** a restricted window beats an allowed window, which beats the default.
- **Windows** include the start time and exclude the end time. Use `end: "24:00"` for "until midnight".
- **Overnight windows:** if `end <= start`, the window crosses midnight, and `days` is the day it **starts**. So `fri 22:00–06:00` covers Saturday 00:00–06:00 too.
- **Polygons** are normalised. If the camera moves or zooms, redraw the zone with `pick_zone.py`.

## Incident record (one entry in `incidents.json`)

`incident_id`, `type: restricted_zone_entry`, `status: pending_review`, `camera_id`, `zone_id`, `zone_name`, `track_id`, `timestamp_utc`, `timestamp_local`, `campus_timezone`, `confidence`, `bbox_xyxy_px`, `bbox_xyxy_norm`, `ground_point_norm`, `dwell_seconds`, `frame_size`, `evidence_frame`, `frame_index`, `detector`.

**Duplicate suppression:**
- A person must stay inside for `min_dwell_seconds` before an incident is raised. This filters out one-frame flickers.
- After that, the same track and zone pair stays silent for `cooldown_seconds`, even if the person steps out and back in.
- Detections without a track ID are ignored.

## Commands (CMD, from `ai-service`)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\ai-service
.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m unittest discover -s training\restricted_zone\tests -t . -v
python -m training.restricted_zone.demo_video --video path\to\clip.mp4 --start-time "2026-11-16 21:30"
```

Useful demo options:

| Option | Effect |
|---|---|
| `--show` | Live window; press `q` to stop |
| `--max-frames 300` | Process only the first 300 frames |
| `--device cpu` | Force CPU |
| `--video 0` | Webcam, using the real current time |
| `--config path\to\my_zones.yaml` | Use your own zones |

To draw zones for your own clip: `python -m training.restricted_zone.pick_zone --source path\to\clip.mp4`

## Limitations

- **Recorded video has no trustworthy clock.** Frame time is `--start-time` plus the frame position. Live CCTV integration with real timestamps is **not** implemented.
- **ID switches.** After a long occlusion, or when people cross paths, a person can get a new track ID. That can produce a second incident inside the cooldown, or move an incident to the wrong ID.
- **Ground point.** If the feet are hidden (a desk, the frame edge), the bottom of the box isn't the floor position, so membership near zone edges is approximate. Polygons are drawn in image space with no camera calibration.
- **Detector limits.** Small, distant or heavily occluded people, low light and unusual camera angles reduce recall. Confidence scores are not calibrated probabilities.
- **No identity information.** The module doesn't know whether a person is authorised (staff, security). Every incident needs human review.
