// Live AI model availability, read from the AI service through the backend (/api/ai/models/status).

import { useCallback, useEffect, useState } from 'react';
import { getModelStatus } from '../services/incidentService.js';

const MODULES = [
  { name: 'Fighting detection', models: ['fighting'] },
  { name: 'Smoking detection', models: ['smoking_cnn', 'cigarette_yolo'] },
  { name: 'Restricted zone / time', models: ['person_yolo'] },
];

const STATUS = {
  loaded: ['Loaded', 'badge-ok'],
  unavailable: ['Unavailable', 'badge-warn'],
  failed: ['Failed to load', 'badge-danger'],
  not_loaded: ['Not loaded yet', 'badge-muted'],
};

export default function ModelStatusPanel() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await getModelStatus());
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>AI models</h2>
        <button className="button" onClick={refresh} disabled={loading}>
          {loading ? 'Checking…' : 'Check now'}
        </button>
      </div>
      {error && (
        <div className="status-banner status-offline">
          <span className="status-dot" aria-hidden="true" />
          {error.message}
        </div>
      )}
      {!data && !error && <p className="panel-note">Loading model status… (first check loads the models)</p>}
      {data && (
        <>
          <ul className="module-list">
            {MODULES.map((mod) => (
              <li key={mod.name} className="model-row">
                <span className="model-row-name">{mod.name}</span>
                <span className="model-row-badges">
                  {mod.models.map((key) => {
                    const m = data.models?.[key];
                    const [text, cls] = STATUS[m?.status] ?? ['Unknown', 'badge-muted'];
                    return (
                      <span key={key} className={`badge ${cls}`} title={m?.reason || m?.description || ''}>
                        {m?.description?.split(' (')[0] ?? key}: {text}
                      </span>
                    );
                  })}
                </span>
              </li>
            ))}
          </ul>
          {Object.values(data.models ?? {}).filter((m) => m.reason).map((m) => (
            <p key={m.name} className="panel-note">{m.name}: {m.reason}</p>
          ))}
          <p className="panel-note">
            Inference device: <strong>{data.device?.toUpperCase()}</strong>
            {data.gpu && ` (${data.gpu.name})`}. Models analyse uploaded recordings only, not live video.
          </p>
        </>
      )}
    </section>
  );
}
