# Model card — SmokingCNN

- **Origin:** trained from scratch (no pretrained weights)
- **Status:** architecture implemented; **not trained**. No metrics exist yet.
- **Code:** `ai-service/training/smoking/cnn/model.py`
- **Weights file:** — (after Stage 2: `ai-service/models/smoking/cnn/`)

## Architecture
4 × [Conv3×3 → BatchNorm → ReLU → MaxPool2] with 32/64/128/256 filters, then AdaptiveAvgPool → Linear(256) → ReLU → Dropout(0.5) → Linear(1). 454,945 parameters. Receptive field per final cell: 46×46 px. Details: `ai-service/training/smoking/cnn/README.md`.

## Data
`dataset_smoking` (developer-provided). Source and license: **to be recorded in Stage 2**.

| Split | not_smoking | smoking |
|---|---:|---:|
| Train + validation pool (developer-reported counts) | 2,751 | 2,936 |
| Test (`testing/`, held out, developer-reported) | 200 | 200 |

Train/validation split: to be created in Stage 2 (stratified, fixed seed, saved to file).

## Input / output
Input `(N, 3, 224, 224)` RGB float. Output `(N, 1)` logit; `sigmoid` → probability of `smoking`.

## Results
_None. To be filled with measured values after training and a single final test evaluation._

## Limitations (expected, to be confirmed)
- Small receptive field → depends on local cues; poor on small/distant people unless given person crops.
- Dataset likely differs from campus CCTV (viewpoint, resolution, lighting).

## Intended use
Flags possible smoking for administrator review only. Never used for automatic action.
