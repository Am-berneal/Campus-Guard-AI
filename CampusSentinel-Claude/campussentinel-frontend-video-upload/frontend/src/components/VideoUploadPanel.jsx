// Upload a recorded video "from" one of the simulated campus cameras.
// Flow: pick camera → pick file → Upload → progress → result returned by Express (from FastAPI).
// AI analysis is NOT connected yet; the panel says so and shows no detection results.

import { useRef, useState } from 'react';
import { cameraName, MAX_VIDEO_BYTES, SIMULATED_CAMERAS, VIDEO_EXTENSIONS } from '../constants/cameras.js';
import { uploadVideo, validateVideoFile } from '../services/videoService.js';
import { useHealth } from '../hooks/BackendHealthContext.jsx';
import '../styles/upload.css';

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '—';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB'];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 ? 2 : 1)} ${units[i]}`;
}

// Extra guidance for errors an administrator can fix during a demo.
function hintFor(error) {
  if (error.code === 'AI_SERVICE_UNAVAILABLE') return 'Start the Python AI service (FastAPI on port 8000) and try again.';
  if (error.code === 'AI_SERVICE_TIMEOUT') return 'The AI service took too long. Check its console for errors.';
  if (error.kind === 'network' || (error.kind === 'http' && !error.data?.error)) {
    return 'The backend did not answer. Check that it is running on port 5000.';
  }
  return null;
}

// Stages: idle → uploading (bytes going to Express) → forwarding (Express ⇄ FastAPI) → success | error
export default function VideoUploadPanel() {
  const { state: backendState } = useHealth();
  const [cameraId, setCameraId] = useState('');
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState(null);
  const [stage, setStage] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  const busy = stage === 'uploading' || stage === 'forwarding';
  const canUpload = Boolean(cameraId && file && !fileError && !busy);

  function chooseFile(selected) {
    setFile(selected ?? null);
    setFileError(selected ? validateVideoFile(selected) : null);
    setError(null);
    if (stage === 'success' || stage === 'error') setStage('idle');
  }

  async function handleUpload() {
    if (!canUpload) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setStage('uploading');
    setProgress(0);
    setError(null);
    setResult(null);
    try {
      const data = await uploadVideo({
        cameraId,
        file,
        signal: controller.signal,
        onProgress: (percent) => {
          setProgress(percent);
          // All bytes sent: Express is now forwarding the file to FastAPI.
          if (percent >= 100) setStage('forwarding');
        },
      });
      setResult(data);
      setStage('success');
    } catch (apiError) {
      setError(apiError);
      setStage('error');
    } finally {
      abortRef.current = null;
    }
  }

  function resetForAnotherUpload() {
    setFile(null);
    setFileError(null);
    setStage('idle');
    setProgress(0);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = ''; // allows choosing the same file again
  }

  function onDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    if (!busy && event.dataTransfer.files?.length) chooseFile(event.dataTransfer.files[0]);
  }

  const ai = result?.aiService ?? {};

  return (
    <section className="panel upload-panel" aria-labelledby="upload-title">
      <div className="panel-header">
        <h2 id="upload-title">Upload CCTV video</h2>
        <span className="badge badge-muted">Simulated camera sources</span>
      </div>
      <p className="upload-intro">
        No physical cameras are connected. Choose the campus location the recording represents, then upload
        a video file. It is sent to the backend, which passes it to the AI service for storage.
      </p>

      {backendState === 'offline' && (
        <div className="upload-alert upload-alert-error" role="alert">
          The backend is offline, so uploads will fail until it is running.
        </div>
      )}

      {/* Step 1: camera */}
      <fieldset className="upload-step" disabled={busy}>
        <legend>1. Simulated camera</legend>
        <div className="camera-grid">
          {SIMULATED_CAMERAS.map((cam) => (
            <button
              key={cam.id}
              type="button"
              className={`camera-option ${cameraId === cam.id ? 'selected' : ''}`}
              aria-pressed={cameraId === cam.id}
              onClick={() => setCameraId(cam.id)}
            >
              <span className="camera-option-name">{cam.name}</span>
              <span className="camera-option-id">{cam.id} · demo source</span>
            </button>
          ))}
        </div>
      </fieldset>

      {/* Step 2: file */}
      <fieldset className="upload-step" disabled={busy}>
        <legend>2. Video file</legend>
        <label
          className={`dropzone ${isDragging ? 'dragging' : ''} ${fileError ? 'invalid' : ''}`}
          onDragOver={(e) => { e.preventDefault(); if (!busy) setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
        >
          <input
            ref={inputRef}
            type="file"
            accept={VIDEO_EXTENSIONS.join(',')}
            onChange={(e) => chooseFile(e.target.files?.[0])}
          />
          {file ? (
            <span className="dropzone-file">
              <strong>{file.name}</strong>
              <span>{formatBytes(file.size)}</span>
            </span>
          ) : (
            <span className="dropzone-text">
              <strong>Choose a video</strong> or drag it here
            </span>
          )}
          <span className="dropzone-rules">
            {VIDEO_EXTENSIONS.map((e) => e.slice(1).toUpperCase()).join(', ')} · max {MAX_VIDEO_BYTES / 1024 / 1024} MB
          </span>
        </label>
        {fileError && <p className="field-error" role="alert">{fileError}</p>}
      </fieldset>

      {/* Step 3: upload */}
      {stage !== 'success' && (
      <div className="upload-actions">
        <button type="button" className="button button-primary" onClick={handleUpload} disabled={!canUpload}>
          {busy ? 'Uploading…' : 'Upload video'}
        </button>
        {busy && (
          <button type="button" className="button" onClick={() => abortRef.current?.abort()}>
            Cancel
          </button>
        )}
        {!busy && !canUpload && stage !== 'success' && (
          <span className="upload-hint">
            {!cameraId && !file ? 'Select a camera and a video.' : !cameraId ? 'Select a camera.' : !file ? 'Select a video.' : ''}
          </span>
        )}
      </div>
      )}

      {busy && (
        <div className="upload-progress" aria-live="polite">
          <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}>
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <span>
            {stage === 'uploading'
              ? `Sending to backend… ${progress}%`
              : 'Upload received. Backend is forwarding the video to the AI service…'}
          </span>
        </div>
      )}

      {stage === 'error' && error && (
        <div className="upload-alert upload-alert-error" role="alert">
          <strong>{error.kind === 'cancelled' ? 'Upload cancelled.' : 'Upload failed.'}</strong>{' '}
          {error.kind !== 'cancelled' && error.message}
          {error.code && error.kind !== 'cancelled' && <code className="error-code">{error.code}</code>}
          {Array.isArray(error.data?.error?.details) && (
            <ul>
              {error.data.error.details.map((d, i) => (
                <li key={i}>{[d.field, d.message].filter(Boolean).join(': ')}</li>
              ))}
            </ul>
          )}
          {hintFor(error) && <div className="upload-alert-hint">{hintFor(error)}</div>}
        </div>
      )}

      {stage === 'success' && result && (
        <div className="upload-result" aria-live="polite">
          <div className="upload-alert upload-alert-success">
            <strong>Video uploaded and stored.</strong> AI analysis is not connected yet, so no detection has been
            run on this video.
          </div>
          <dl className="detail-list">
            <div className="detail-row"><dt>Camera</dt><dd>{ai.camera_name ?? cameraName(result.cameraId)}</dd></div>
            <div className="detail-row"><dt>Original filename</dt><dd>{ai.original_filename ?? result.originalName}</dd></div>
            <div className="detail-row"><dt>File size</dt><dd>{formatBytes(ai.size_bytes ?? result.sizeBytes)}</dd></div>
            <div className="detail-row"><dt>Upload status</dt><dd>{ai.status ?? '—'}</dd></div>
            <div className="detail-row"><dt>Upload ID</dt><dd><code>{ai.upload_id ?? '—'}</code></dd></div>
            {ai.saved_filename && (
              <div className="detail-row"><dt>Stored as</dt><dd><code>{ai.saved_filename}</code></dd></div>
            )}
          </dl>
          {ai.message && <p className="panel-note">AI service: “{ai.message}”</p>}
          <button type="button" className="button" onClick={resetForAnotherUpload}>
            Upload another video
          </button>
        </div>
      )}
    </section>
  );
}
