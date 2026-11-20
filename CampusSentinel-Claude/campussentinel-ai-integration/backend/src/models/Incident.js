// A POTENTIAL incident produced by AI analysis of an uploaded (simulated-camera) video.
// It is never a confirmed violation: an administrator reviews it (reviewed / dismissed)
// and, if needed, contacts security by walkie-talkie. The system takes no action itself.

import mongoose from 'mongoose';

export const INCIDENT_TYPES = ['fighting', 'smoking', 'restricted_zone'];
export const INCIDENT_STATUSES = ['pending_review', 'reviewed', 'dismissed'];
export const CAMERA_IDS = ['library', 'b_block', 'cos', 'kravings'];

const evidenceSchema = new mongoose.Schema(
  {
    timeSeconds: { type: Number, min: 0 },          // position in the uploaded video
    frameIndex: { type: Number, min: 0 },
    confidence: { type: Number, min: 0, max: 1 },
    imageFile: { type: String, maxlength: 200 },     // file name served by the AI service (proxied by /api/incidents/:id/evidence/:n)
    details: { type: mongoose.Schema.Types.Mixed },  // model-specific values (boxes, CNN probability, zone, ...)
  },
  { _id: false }
);

const incidentSchema = new mongoose.Schema(
  {
    cameraId: { type: String, enum: CAMERA_IDS, required: true },
    cameraName: { type: String, required: true, maxlength: 100 },
    simulatedSource: { type: Boolean, default: true },        // uploaded recording, not live CCTV
    incidentType: { type: String, enum: INCIDENT_TYPES, required: true },
    confidence: { type: Number, min: 0, max: 1, required: true },
    timestamp: { type: Date, required: true },                // when the AI analysis produced it
    videoId: { type: String, required: true, maxlength: 64 }, // FastAPI upload_id
    originalFilename: { type: String, maxlength: 255 },
    eventKey: { type: String, required: true, maxlength: 200 }, // stable per event → no duplicates
    startSeconds: { type: Number, min: 0 },
    endSeconds: { type: Number, min: 0 },
    evidence: { type: [evidenceSchema], default: [] },
    status: { type: String, enum: INCIDENT_STATUSES, default: 'pending_review' },
    modelName: { type: String, required: true, maxlength: 100 },
    description: { type: String, maxlength: 1000 },
    zone: { id: String, name: String },                       // restricted_zone only
    review: {
      reviewedAt: Date,
      note: { type: String, maxlength: 1000 },
    },
  },
  { timestamps: true }
);

// One document per detected event of one video: re-analysing the same upload updates it instead of duplicating.
incidentSchema.index({ videoId: 1, incidentType: 1, eventKey: 1 }, { unique: true });
incidentSchema.index({ status: 1, createdAt: -1 });

export const Incident = mongoose.models.Incident || mongoose.model('Incident', incidentSchema);
