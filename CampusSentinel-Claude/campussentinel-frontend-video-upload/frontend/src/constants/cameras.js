// SIMULATED camera sources for the demo. No physical CCTV cameras are connected:
// an administrator uploads a recorded video "as if" it came from one of these locations.
// IDs must match the backend's allowed camera_id values exactly.

export const SIMULATED_CAMERAS = Object.freeze([
  { id: 'library', name: 'Outside Library' },
  { id: 'b_block', name: 'B Block' },
  { id: 'cos', name: 'COS' },
  { id: 'kravings', name: 'Kravings' },
]);

export function cameraName(id) {
  return SIMULATED_CAMERAS.find((c) => c.id === id)?.name ?? id;
}

// Same rules the backend enforces; checking here gives instant feedback before uploading.
export const VIDEO_EXTENSIONS = Object.freeze(['.mp4', '.avi', '.mov', '.mkv']);
export const MAX_VIDEO_BYTES = 500 * 1024 * 1024; // 500 MB, identical to the backend limit
