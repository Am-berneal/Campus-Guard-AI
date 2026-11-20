# CNN + YOLO fusion — evaluation plan (Stage 6)

**Status:** plan only. No experiment run. Whether fusion helps is unknown until measured.

## The data problem to solve first

The two models are trained on **different datasets**:

- CNN: `dataset_smoking` — image-level labels (smoking / not_smoking), no boxes.
- YOLO: a cigarette dataset (Stage 3) — boxes, but no "is this person smoking" label.

Fusion can only be evaluated on images that carry the label we care about: **smoking / not smoking**. Plan:

| Purpose | Data |
|---|---|
| YOLO detection quality (P, R, mAP) | YOLO dataset's own val/test split |
| CNN quality | Validation split carved from `dataset_smoking/smoking` + `not_smoking` (Stage 2) |
| **Tuning fusion thresholds / weights** | The **same CNN validation split** — run both models on it |
| **Final comparison** (CNN vs YOLO vs fusion) | `dataset_smoking/testing` — used **once**, at the end |

"YOLO alone" as a smoking classifier means: *image = smoking if at least one cigarette box ≥ threshold t*.

## Candidate strategies (to be compared, not assumed)

| Strategy | Rule | Expected behaviour |
|---|---|---|
| CNN only | `p_cnn ≥ t₁` | Baseline |
| YOLO only | `max_box_conf ≥ t₂` | Misses smoking without a visible cigarette (e.g. hidden, vape); flags held/unlit cigarettes |
| AND | `p_cnn ≥ t₁` **and** `max_box_conf ≥ t₂` | Fewer false positives, more false negatives |
| OR | either condition | More recall, more false positives |
| Weighted score | `w·p_cnn + (1−w)·max_box_conf ≥ t` | Tunable trade-off; scores from two models are not automatically comparable |
| CNN with YOLO boost | `p_cnn ≥ t₁`, or `p_cnn ≥ t₁' (lower)` if a cigarette is detected | Uses YOLO as supporting evidence |

All thresholds and weights are chosen on **validation data only**, using a pre-declared criterion (e.g. best F1, or best recall at precision ≥ X). Then frozen before touching the test set.

## Reporting
For each strategy on the test set: precision, recall, F1, confusion matrix, and examples of false positives / false negatives.
A fusion result is reported as an improvement only if the test-set numbers show it; otherwise we report that it did not help.

## Known limitations (stated in advance)
- Whole-image labels: an image labelled `smoking` may contain a cigarette the YOLO model cannot see at low resolution.
- Domain gap: both datasets are likely web photos, not campus CCTV. Test-set numbers will overestimate real-world performance.
- Score calibration: a CNN probability of 0.7 and a YOLO confidence of 0.7 do not mean the same thing.
