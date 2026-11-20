# Smoking inference (Stage 7)

Not implemented yet. Will contain local scripts that load trained weights from `ai-service/models/smoking/` and run:

- CNN only on images,
- YOLO only on images and video,
- the fusion rule selected in Stage 6.

The CNN architecture is imported from `training/smoking/cnn/model.py` so inference always uses exactly the trained architecture.
