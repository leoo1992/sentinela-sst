'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { calculatePoseMetrics, inferLiftingPhase, isPoseNearRiskZone, namedPoint, primaryPose } from '@/lib/geometry';
import { evaluateLocally } from '@/lib/localRules';
import { inspectPpe } from '@/lib/ppe';
import type { EvaluationPayload, EvaluationResponse, Finding, ModuleId, PoseLike, PoseMetrics, PpeAssessment } from '@/lib/types';

type FacingMode = 'user' | 'environment';
type PoseDetectorLike = {
  estimatePoses: (input: HTMLVideoElement, config?: { maxPoses?: number; flipHorizontal?: boolean }) => Promise<PoseLike[]>;
  dispose?: () => void;
};

const SKELETON: Array<[string, string]> = [
  ['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'], ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'], ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'], ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
  ['nose', 'left_eye'], ['nose', 'right_eye'], ['left_eye', 'left_ear'], ['right_eye', 'right_ear'],
];

const moduleTitles: Record<ModuleId, string> = {
  epi: 'Inspeção de EPI',
  altura: 'Segurança em Altura',
  ergonomia: 'Análise Ergonômica',
  cargas: 'Levantamento de Cargas',
};

function metricText(value: number | null, suffix = '°') { return value === null ? '—' : String(value) + suffix; }
function severityLabel(severity: Finding['severity']) {
  if (severity === 'alert') return 'ALERTA';
  if (severity === 'attention') return 'ATENÇÃO';
  if (severity === 'ok') return 'OK';
  return 'INFO';
}
function ppeLabel(ppe: PpeAssessment | null) {
  if (!ppe) return 'aguardando';
  const values = Object.values(ppe);
  return values.filter((item) => item.status === 'detectado').length + '/' + values.length;
}

export default function VisionCamera({ moduleId }: { moduleId: ModuleId }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<PoseDetectorLike | null>(null);
  const animationRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const lastInferenceRef = useRef(0);
  const lastPpeRef = useRef(0);
  const lastBackendRef = useRef(0);
  const currentPpeRef = useRef<PpeAssessment | null>(null);
  const apiBusyRef = useRef(false);
  const frameCounterRef = useRef({ count: 0, startedAt: 0 });

  const [running, setRunning] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [facingMode, setFacingMode] = useState<FacingMode>('environment');
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<PoseMetrics | null>(null);
  const [ppe, setPpe] = useState<PpeAssessment | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [summary, setSummary] = useState('Ative a câmera para iniciar a análise.');
  const [, setFps] = useState(0);
  const [peopleCount, setPeopleCount] = useState(0);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [riskSide, setRiskSide] = useState<'left' | 'right'>('right');
  const [riskWidth, setRiskWidth] = useState(24);
  const [zoneRisk, setZoneRisk] = useState(false);
  const [liftingPhase, setLiftingPhase] = useState<string | null>(null);

  const statusCards = useMemo(() => {
    const alertCount = findings.filter((item) => item.severity === 'alert').length;
    const attentionCount = findings.filter((item) => item.severity === 'attention').length;

    return [
      { label: 'Pessoas', value: String(peopleCount) },
      {
        label: 'Situação',
        value:
          alertCount > 0
            ? 'Alerta'
            : attentionCount > 0
              ? 'Atenção'
              : running
                ? 'Normal'
                : 'Em espera',
      },
      { label: 'Alertas', value: String(alertCount + attentionCount) },
      {
        label: 'EPIs',
        value: moduleId === 'epi' || moduleId === 'altura' ? ppeLabel(ppe) : '—',
      },
    ];
  }, [findings, moduleId, peopleCount, ppe, running]);

  const clearOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const stopCamera = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    animationRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setPeopleCount(0); setMetrics(null); setPpe(null); currentPpeRef.current = null;
    setZoneRisk(false); setLiftingPhase(null); setFps(0); clearOverlay();
  }, [clearOverlay]);

  const ensureDetector = useCallback(async () => {
    if (detectorRef.current) return detectorRef.current;
    setLoadingModel(true);
    try {
      const tf = await import('@tensorflow/tfjs-core');
      await import('@tensorflow/tfjs-backend-webgl');
      const poseDetection = await import('@tensorflow-models/pose-detection');
      await tf.setBackend('webgl');
      await tf.ready();
      const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
        modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING,
        enableTracking: true,
      });
      detectorRef.current = detector as unknown as PoseDetectorLike;
      return detectorRef.current;
    } finally {
      setLoadingModel(false);
    }
  }, []);

  const drawOverlay = useCallback((poses: PoseLike[], currentMetrics: PoseMetrics | null, currentZoneRisk: boolean, videoWidth: number, videoHeight: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (canvas.width !== videoWidth) canvas.width = videoWidth;
    if (canvas.height !== videoHeight) canvas.height = videoHeight;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, videoWidth, videoHeight);

    if (moduleId === 'altura') {
      const width = videoWidth * (riskWidth / 100);
      const x = riskSide === 'left' ? 0 : videoWidth - width;
      context.fillStyle = currentZoneRisk ? 'rgba(255, 92, 92, .16)' : 'rgba(231, 255, 88, .09)';
      context.fillRect(x, 0, width, videoHeight);
      context.strokeStyle = currentZoneRisk ? '#ff6b6b' : '#e7ff58';
      context.lineWidth = Math.max(2, videoWidth / 360);
      context.setLineDash([12, 10]);
      context.beginPath();
      context.moveTo(riskSide === 'left' ? width : videoWidth - width, 0);
      context.lineTo(riskSide === 'left' ? width : videoWidth - width, videoHeight);
      context.stroke();
      context.setLineDash([]);
    }

    poses.forEach((pose, poseIndex) => {
      context.lineWidth = Math.max(2, videoWidth / 420);
      context.strokeStyle = poseIndex === 0 ? '#4de8c2' : 'rgba(77, 232, 194, .52)';
      context.fillStyle = poseIndex === 0 ? '#e7ff58' : '#4de8c2';
      for (const [fromName, toName] of SKELETON) {
        const from = namedPoint(pose, fromName), to = namedPoint(pose, toName);
        if (!from || !to) continue;
        context.beginPath(); context.moveTo(from.x, from.y); context.lineTo(to.x, to.y); context.stroke();
      }
      pose.keypoints.forEach((point) => {
        if ((point.score ?? 1) < 0.28) return;
        context.beginPath(); context.arc(point.x, point.y, Math.max(3, videoWidth / 180), 0, Math.PI * 2); context.fill();
      });
    });

    if (currentMetrics?.bodyBox) {
      const box = currentMetrics.bodyBox;
      context.strokeStyle = '#ffffff';
      context.lineWidth = Math.max(1.5, videoWidth / 520);
      context.strokeRect(box.x, box.y, box.width, box.height);
    }
  }, [moduleId, riskSide, riskWidth]);

  const sendToBackend = useCallback(async (payload: EvaluationPayload) => {
    if (apiBusyRef.current) return;
    apiBusyRef.current = true;
    try {
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          module: payload.module,
          metrics: payload.metrics ? {
            trunk_inclination: payload.metrics.trunkInclination,
            neck_inclination: payload.metrics.neckInclination,
            left_knee_angle: payload.metrics.leftKneeAngle,
            right_knee_angle: payload.metrics.rightKneeAngle,
            left_hip_angle: payload.metrics.leftHipAngle,
            right_hip_angle: payload.metrics.rightHipAngle,
            shoulder_tilt: payload.metrics.shoulderTilt,
            hip_tilt: payload.metrics.hipTilt,
            knee_asymmetry: payload.metrics.kneeAsymmetry,
            hip_asymmetry: payload.metrics.hipAsymmetry,
            visibility: payload.metrics.visibility,
          } : null,
          ppe: payload.ppe ? Object.fromEntries(Object.entries(payload.ppe).map(([key, value]) => [key, {
            label: value.label, status: value.status, confidence: value.confidence, note: value.note ?? null,
          }])) : null,
          zone_risk: payload.zoneRisk,
          lifting_phase: payload.liftingPhase,
        }),
      });
      if (!response.ok) throw new Error('Backend indisponível');
      const data = (await response.json()) as EvaluationResponse;
      setBackendOnline(true);
      setFindings(data.findings);
      setSummary(data.summary);
    } catch {
      setBackendOnline(false);
      const fallback = evaluateLocally(payload);
      setFindings(fallback.findings);
      setSummary(fallback.summary);
    } finally {
      apiBusyRef.current = false;
    }
  }, []);

  const runFrame = useCallback(async (timestamp: number) => {
    if (!runningRef.current) return;
    const video = videoRef.current, detector = detectorRef.current;
    if (!video || !detector || video.readyState < 2) {
      animationRef.current = requestAnimationFrame(runFrame);
      return;
    }
    if (timestamp - lastInferenceRef.current < 105) {
      animationRef.current = requestAnimationFrame(runFrame);
      return;
    }
    lastInferenceRef.current = timestamp;

    try {
      const poses = await detector.estimatePoses(video, { maxPoses: 6, flipHorizontal: false });
      setPeopleCount(poses.length);
      const primary = primaryPose(poses);
      const currentMetrics = primary ? calculatePoseMetrics(primary) : null;
      setMetrics(currentMetrics);

      let currentPpe = currentPpeRef.current;
      if (primary && (moduleId === 'epi' || moduleId === 'altura') && timestamp - lastPpeRef.current > 650) {
        lastPpeRef.current = timestamp;
        currentPpe = inspectPpe(video, primary);
        currentPpeRef.current = currentPpe;
        setPpe(currentPpe);
      }
      if (moduleId !== 'epi' && moduleId !== 'altura' && currentPpeRef.current) {
        currentPpeRef.current = null; currentPpe = null; setPpe(null);
      }

      const currentZoneRisk = moduleId === 'altura' ? isPoseNearRiskZone(currentMetrics, video.videoWidth, riskSide, riskWidth) : false;
      setZoneRisk(currentZoneRisk);
      const currentLiftingPhase = moduleId === 'cargas' ? inferLiftingPhase(currentMetrics) : null;
      setLiftingPhase(currentLiftingPhase);
      drawOverlay(poses, currentMetrics, currentZoneRisk, video.videoWidth, video.videoHeight);

      const payload: EvaluationPayload = {
        module: moduleId, metrics: currentMetrics, ppe: currentPpe, zoneRisk: currentZoneRisk, liftingPhase: currentLiftingPhase,
      };
      const local = evaluateLocally(payload);
      setFindings((previous) => backendOnline === true ? previous : local.findings);
      setSummary((previous) => backendOnline === true ? previous : local.summary);

      if (timestamp - lastBackendRef.current > 900) {
        lastBackendRef.current = timestamp;
        void sendToBackend(payload);
      }

      const counter = frameCounterRef.current;
      if (!counter.startedAt) counter.startedAt = timestamp;
      counter.count += 1;
      const elapsed = timestamp - counter.startedAt;
      if (elapsed >= 1000) {
        setFps(Math.round((counter.count * 1000) / elapsed));
        frameCounterRef.current = { count: 0, startedAt: timestamp };
      }
    } catch (frameError) {
      console.error('Falha no frame de visão computacional', frameError);
    }

    animationRef.current = requestAnimationFrame(runFrame);
  }, [backendOnline, drawOverlay, moduleId, riskSide, riskWidth, sendToBackend]);

  const startCamera = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Este navegador não oferece acesso compatível à câmera.');
      return;
    }
    stopCamera();
    try {
      const detector = await ensureDetector();
      if (!detector) throw new Error('Modelo não carregado');
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
        });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      runningRef.current = true;
      setRunning(true);
      setBackendOnline(null);
      setSummary('Análise iniciada. Mantenha o corpo no enquadramento.');
      frameCounterRef.current = { count: 0, startedAt: performance.now() };
      animationRef.current = requestAnimationFrame(runFrame);
    } catch (cameraError) {
      console.error(cameraError);
      setError('Não foi possível iniciar a câmera. Verifique a permissão do navegador e tente novamente.');
      stopCamera();
    }
  }, [ensureDetector, facingMode, runFrame, stopCamera]);

  const switchCamera = useCallback(() => {
    setFacingMode((current) => current === 'environment' ? 'user' : 'environment');
    if (running) stopCamera();
  }, [running, stopCamera]);

  useEffect(() => {
    setFindings([]);
    setSummary(running ? 'Módulo alterado. Recalculando análise…' : 'Ative a câmera para iniciar a análise.');
    currentPpeRef.current = null; setPpe(null); setMetrics(null); setZoneRisk(false); setLiftingPhase(null);
  }, [moduleId, running]);

  useEffect(() => {
    if (!runningRef.current) return;

    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
    }

    animationRef.current = requestAnimationFrame(runFrame);

    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [runFrame]);

  useEffect(() => () => {
    runningRef.current = false;
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    detectorRef.current?.dispose?.();
  }, []);

  return (
    <section className="visionGrid">
      <div className="cameraCard">
        <div className="cameraToolbar">
          <div><p className="sectionKicker">CÂMERA AO VIVO</p><h3>{moduleTitles[moduleId]}</h3></div>
          <div className="cameraActions">
            <button type="button" className="secondaryButton" onClick={switchCamera}>{facingMode === 'environment' ? 'Câmera traseira' : 'Câmera frontal'}</button>
            {running
              ? <button type="button" className="dangerButton" onClick={stopCamera}>Encerrar</button>
              : <button type="button" className="primaryButton" onClick={startCamera} disabled={loadingModel}>{loadingModel ? 'Carregando IA…' : 'Ativar câmera'}</button>}
          </div>
        </div>

        {moduleId === 'altura' && (
          <div className="heightControls">
            <div className="segmented">
              <button type="button" className={riskSide === 'left' ? 'selected' : ''} onClick={() => setRiskSide('left')}>Borda à esquerda</button>
              <button type="button" className={riskSide === 'right' ? 'selected' : ''} onClick={() => setRiskSide('right')}>Borda à direita</button>
            </div>
            <label>Zona de risco <strong>{riskWidth}%</strong>
              <input type="range" min="12" max="42" value={riskWidth} onChange={(event) => setRiskWidth(Number(event.target.value))} />
            </label>
          </div>
        )}

        <div className={running ? 'cameraStage live' : 'cameraStage'}>
          <video ref={videoRef} muted playsInline className="cameraVideo" />
          <canvas ref={canvasRef} className="cameraOverlay" />
          {!running && (
            <div className="cameraEmpty">
              <div className="scannerIcon" aria-hidden="true"><span /></div>
              <strong>Câmera desativada</strong>
              <p>O processamento da imagem acontece no seu dispositivo. Nenhum frame é enviado ou armazenado pelo backend.</p>
              <button type="button" className="primaryButton large" onClick={startCamera} disabled={loadingModel}>{loadingModel ? 'Preparando modelo…' : 'Abrir câmera'}</button>
            </div>
          )}
          {running && <>
            <div className="liveBadge"><span />AO VIVO</div>
            <div className="moduleBadge">{moduleTitles[moduleId]}</div>
            {moduleId === 'altura' && <div className={zoneRisk ? 'riskBadge alert' : 'riskBadge'}>{zoneRisk ? 'ZONA DE RISCO' : 'ZONA MONITORADA'}</div>}
          </>}
        </div>

        {error && <div className="errorBanner" role="alert">{error}</div>}
        <div className="statusStrip">
          {statusCards.map((status) => <div key={status.label}><span>{status.label}</span><strong>{status.value}</strong></div>)}
        </div>
      </div>

      <aside className="inspectorCard">
        <div className="inspectorHeader">
          <div><p className="sectionKicker">ANÁLISE EM TEMPO REAL</p><h3>Leitura atual</h3></div>
          <div className={running ? 'engineState online' : 'engineState'}><span />{running ? 'Análise ativa' : 'em espera'}</div>
        </div>

        <div className="summaryBox">
          <span>RESUMO</span><strong>{summary}</strong>
        </div>

        {(moduleId === 'ergonomia' || moduleId === 'cargas') && (
          <div className="metricsGrid">
            <div><span>Tronco</span><strong>{metricText(metrics?.trunkInclination ?? null)}</strong></div>
            <div><span>Pescoço</span><strong>{metricText(metrics?.neckInclination ?? null)}</strong></div>
            <div><span>Joelho E.</span><strong>{metricText(metrics?.leftKneeAngle ?? null)}</strong></div>
            <div><span>Joelho D.</span><strong>{metricText(metrics?.rightKneeAngle ?? null)}</strong></div>
            <div><span>Assimetria</span><strong>{metricText(metrics?.kneeAsymmetry ?? null)}</strong></div>
            <div><span>Visibilidade</span><strong>{metrics ? String(metrics.visibility) + '%' : '—'}</strong></div>
          </div>
        )}

        {moduleId === 'cargas' && <div className="phaseCard"><span>FASE ESTIMADA DO MOVIMENTO</span><strong>{liftingPhase ?? 'Aguardando movimento'}</strong></div>}

        {(moduleId === 'epi' || moduleId === 'altura') && ppe && (
          <div className="ppeList">
            {Object.values(ppe).map((entry) => (
              <div key={entry.label} className={'ppeRow ' + entry.status}>
                <span className="ppeState" aria-hidden="true" />
                <div><strong>{entry.label}</strong><small>{entry.note}</small></div>
                <b>{entry.status === 'detectado' ? 'Detectado' : entry.status === 'nao_detectado' ? 'Não detectado' : entry.status === 'incerto' ? 'Inconclusivo' : 'Não avaliável'}</b>
              </div>
            ))}
          </div>
        )}

        <div className="findingsList" aria-live="polite">
          {findings.length === 0
            ? <div className="emptyFinding"><span className="pulseRing" /><p>Aguardando dados da câmera.</p></div>
            : findings.map((entry) => <article key={entry.code} className={'finding ' + entry.severity}><span className="findingTag">{severityLabel(entry.severity)}</span><strong>{entry.title}</strong><p>{entry.detail}</p></article>)}
        </div>

        <div className="privacyCard">
          <strong>Privacidade</strong>
          <p>As imagens da câmera são usadas somente durante a análise e não ficam armazenadas. Ao encerrar a câmera, a sessão é finalizada.</p>
        </div>
      </aside>
    </section>
  );
}
