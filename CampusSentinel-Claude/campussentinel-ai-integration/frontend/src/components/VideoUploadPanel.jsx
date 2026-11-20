// Upload a recorded video "from" one of the simulated campus cameras.
// Flow: pick camera → pick file → Upload → progress → result returned by Express (from FastAPI).
// After a successful upload the video can be analysed by the AI service (all available models);
// potential incidents are stored by the backend for administrator review.

import { useEffect, useRef, useState } from 'react';
import { cameraName, MAX_VIDEO_BYTES, SIMULATED_CAMERAS, VIDEO_EXTENSIONS } from '../constants/cameras.js';
import { analyzeVideo, uploadVideo, validateVideoFile } from '../services/videoService.js';
import AnalysisResults from './AnalysisResults.jsx';
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
  if (error.code === 'DATABASE_UNAVAILABLE') return 'Start MongoDB, then try again (the analysis result is cached by the AI service).';
  if (error.code === 'AI_MODELS_UNAVAILABLE') return 'No AI model could be loaded. Check the Model status panel on the dashboard.';
  if (error.code === 'ANALYSIS_IN_PROGRESS') return 'This video is already being analysed. Wait for it to finish.';
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
  // AI analysis state (separate from the upload state above)
  const [autoAnalyze, setAutoAnalyze] = useState(true);
  const [recordedAt, setRecordedAt] = useState('');
  const [analysisStage, setAnalysisStage] = useState('idle'); // idle | running | done | error
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  // Elapsed-time counter while the analysis runs (analysis can take minutes on CPU).
  useEffect(() => {
    if (analysisStage !== 'running') return undefined;
    const started = Date.now();
    const id = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(id);
  }, [analysisStage]);

  const uploading = stage === 'uploading' || stage === 'forwarding';
  const busy = uploading || analysisStage === 'running'; // disables the form during upload AND analysis
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
      if (autoAnalyze) runAnalysis(data);
    } catch (apiError) {
      setError(apiError);
      setStage('error');
    } finally {
      abortRef.current = null;
    }
  }

  async function runAnalysis(uploaded = result, force = false) {
    const uploadId = uploaded?.aiService?.upload_id;
    if (!uploadId) return;
    setAnalysisStage('running');
    setAnalysisError(null);
    setElapsed(0);
    try {
      const data = await analyzeVideo({
        uploadId,
        cameraId: uploaded.cameraId,
        originalName: uploaded.originalName,
        recordedAt: recordedAt || undefined,
        force,
      });
      setAnalysisResult(data);
      setAnalysisStage('done');
    } catch (apiError) {
      setAnalysisError(apiError);
      setAnalysisStage('error');
    }
  }

  function resetForAnotherUpload() {
    setAnalysisStage('idle');
    setAnalysisResult(null);
    setAnalysisError(null);
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
        No physical cameras or live streams are connected. Choose the campus location the recording represents,
        upload a video file, and let the AI service analyse it. Results are potential incidents for review.
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

      {/* Step 3: analysis options */}
      <fieldset className="upload-step" disabled={busy}>
        <legend>3. AI analysis</legend>
        <label className="check-row">
          <input type="checkbox" checked={autoAnalyze} onChange={(e) => setAutoAnalyze(e.target.checked)} />
          Analyse automatically after upload (fighting, smoking, restricted zone)
        </label>
        <label className="time-row">
          <span>Recording time (campus time, optional; used for restricted-hour rules)</span>
          <input type="datetime-local" value={recordedAt} onChange={(e) => setRecordedAt(e.target.value)} />
        </label>
        <p className="field-help">Leave empty to use the current time.</p>
      </fieldset>

      {/* Step 4: upload */}
      {stage !== 'success' && (
      <div className="upload-actions">
        <button type="button" className="button button-primary" onClick={handleUpload} disabled={!canUpload}>
          {uploading ? 'Uploading…' : 'Upload video'}
        </button>
        {uploading && (
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

      {uploading && (
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
            <strong>Video uploaded and stored by the AI service.</strong>
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

          {/* ---- AI analysis of this upload ---- */}
          {analysisStage === 'idle' && (
            <button type="button" className="button button-primary analysis-start" onClick={() => runAnalysis()}>
              Analyse with AI
            </button>
          )}
          {analysisStage === 'running' && (
            <div className="upload-progress" aria-live="polite">
              <div className="progress-track progress-indeterminate"><div className="progress-fill" /></div>
              <span>
                Analysing with all available AI models… {elapsed}s elapsed. This can take a few minutes, longer
                without a GPU.
              </span>
            </div>
          )}
          {analysisStage === 'error' && analysisError && (
            <div className="upload-alert upload-alert-error" role="alert">
              <strong>AI analysis failed.</strong> {analysisError.message}
              {analysisError.code && <code className="error-code">{analysisError.code}</code>}
              {hintFor(analysisError) && <div className="upload-alert-hint">{hintFor(analysisError)}</div>}
              <div><button type="button" className="button" onClick={() => runAnalysis()}>Try again</button></div>
            </div>
          )}
          {analysisStage === 'done' && analysisResult && <AnalysisResults result={analysisResult} />}

          <div className="upload-actions result-actions">
            <button type="button" className="button" onClick={resetForAnotherUpload} disabled={analysisStage === 'running'}>
              Upload another video
            </button>
            {analysisStage === 'done' && (
              <button type="button" className="button" onClick={() => runAnalysis(result, true)}>
                Re-run analysis
              </button>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
