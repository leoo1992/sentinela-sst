'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { calculatePoseMetrics, inferLiftingPhase, isPoseNearRiskZone, namedPoint, primaryPose } from '@/lib/geometry';
import { evaluateLocally } from '@/lib/localRules';
import { inspectPpe } from '@/lib/ppe';
import type { EvaluationPayload, EvaluationResponse, Finding, ModuleId, PoseLike, PoseMetrics, PpeAssessment, PpeItem } from '@/lib/types';

type FacingMode = 'user' | 'environment';
type VisionSource = HTMLVideoElement | HTMLImageElement;
type PoseDetectorLike = {
  estimatePoses: (input: VisionSource, config?: { maxPoses?: number; flipHorizontal?: boolean }) => Promise<PoseLike[]>;
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
function ppeLabel(ppe: PpeAssessment | null, moduleId: ModuleId) {
  if (!ppe) return 'aguardando';
  const keys: Array<keyof PpeAssessment> = moduleId === 'altura'
    ? ['capacete', 'cinturao', 'talabarte', 'travaQuedas']
    : ['capacete', 'oculos', 'colete', 'luvas', 'calcado'];
  const values = keys.map((key) => ppe[key]);
  return values.filter((item) => item.status === 'detectado').length + '/' + values.length;
}

function ppeEntriesForModule(ppe: PpeAssessment, moduleId: ModuleId) {
  const keys: Array<keyof PpeAssessment> = moduleId === 'altura'
    ? ['capacete', 'cinturao', 'talabarte', 'travaQuedas']
    : ['capacete', 'oculos', 'colete', 'luvas', 'calcado'];
  return keys.map((key) => ppe[key]);
}

const PPE_LABELS: Record<keyof PpeAssessment, string> = {
  capacete: 'Capacete',
  oculos: 'Óculos de proteção',
  colete: 'Colete refletivo',
  luvas: 'Luvas',
  calcado: 'Calçado fechado',
  cinturao: 'Cinturão paraquedista',
  talabarte: 'Talabarte',
  travaQuedas: 'Trava-quedas',
};

function emptyPpeAssessment(): PpeAssessment {
  return Object.fromEntries(
    (Object.keys(PPE_LABELS) as Array<keyof PpeAssessment>).map((key) => [
      key,
      {
        label: PPE_LABELS[key],
        status: 'nao_avaliavel',
        confidence: 0,
        note: 'Não foi possível avaliar este item.',
      } satisfies PpeItem,
    ]),
  ) as unknown as PpeAssessment;
}

function mergePpeItem(local: PpeItem, gemini?: PpeItem): PpeItem {
  if (!gemini) return local;

  const aiItem: PpeItem = {
    ...gemini,
    note: gemini.note ? `Análise visual: ${gemini.note}` : undefined,
  };

  if (gemini.status === 'detectado' && gemini.confidence >= 0.52) return aiItem;

  if (
    local.status === 'detectado' &&
    local.confidence >= 0.62 &&
    gemini.status !== 'nao_detectado'
  ) {
    return local;
  }

  if (gemini.status === 'nao_detectado' && gemini.confidence >= 0.72) {
    if (local.status === 'detectado' && local.confidence >= 0.68) {
      return {
        ...gemini,
        status: 'incerto',
        confidence: Math.max(local.confidence, gemini.confidence),
        note: 'As análises visuais divergiram. Confirme o EPI presencialmente.',
      };
    }
    return aiItem;
  }

  if (gemini.status === 'incerto') {
    return local.status === 'detectado' && local.confidence >= 0.68 ? local : aiItem;
  }

  if (gemini.status === 'nao_avaliavel') {
    return local.status === 'detectado' ? local : aiItem;
  }

  return aiItem;
}

function mergePpeAssessments(
  local: PpeAssessment | null,
  gemini: Partial<PpeAssessment> | null,
): PpeAssessment {
  const base = local ?? emptyPpeAssessment();
  if (!gemini) return base;

  return Object.fromEntries(
    (Object.keys(PPE_LABELS) as Array<keyof PpeAssessment>).map((key) => [
      key,
      mergePpeItem(base[key], gemini[key]),
    ]),
  ) as unknown as PpeAssessment;
}

function imageForGemini(image: HTMLImageElement) {
  const maxSide = 1280;
  const largest = Math.max(image.naturalWidth, image.naturalHeight);
  const scale = largest > maxSide ? maxSide / largest : 1;
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Não foi possível preparar a imagem.');

  context.drawImage(image, 0, 0, width, height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.78);
  const imageBase64 = dataUrl.split(',', 2)[1];
  if (!imageBase64) throw new Error('Não foi possível preparar a imagem.');

  return { imageBase64, mimeType: 'image/jpeg' as const };
}

export default function VisionCamera({ moduleId }: { moduleId: ModuleId }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imageObjectUrlRef = useRef<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<PoseDetectorLike | null>(null);
  const animationRef = useRef<number | null>(null);
  const runFrameRef = useRef<((timestamp: number) => void) | null>(null);
  const runningRef = useRef(false);
  const lastInferenceRef = useRef(0);
  const lastPpeRef = useRef(0);
  const lastBackendRef = useRef(0);
  const currentPpeRef = useRef<PpeAssessment | null>(null);
  const apiBusyRef = useRef(false);
  const frameCounterRef = useRef({ count: 0, startedAt: 0 });

  const [running, setRunning] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [imageAnalyzing, setImageAnalyzing] = useState(false);
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
              : running || imageUrl
                ? 'Normal'
                : 'Em espera',
      },
      { label: 'Alertas', value: String(alertCount + attentionCount) },
      {
        label: moduleId === 'altura' ? 'Proteções' : 'EPIs',
        value: moduleId === 'epi' || moduleId === 'altura' ? ppeLabel(ppe, moduleId) : '—',
      },
    ];
  }, [findings, imageUrl, moduleId, peopleCount, ppe, running]);

  const clearOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (canvas && context) context.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const clearImage = useCallback(() => {
    if (imageObjectUrlRef.current) URL.revokeObjectURL(imageObjectUrlRef.current);
    imageObjectUrlRef.current = null;
    setImageUrl(null);
    setImageName(null);
    setImageAnalyzing(false);
    clearOverlay();
  }, [clearOverlay]);

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

  const scheduleNextFrame = useCallback(() => {
    animationRef.current = requestAnimationFrame((nextTimestamp) => {
      runFrameRef.current?.(nextTimestamp);
    });
  }, []);

  const analyzePpeWithGemini = useCallback(async (image: HTMLImageElement) => {
    if (moduleId !== 'epi' && moduleId !== 'altura') return null;

    try {
      const prepared = imageForGemini(image);
      const response = await fetch('/api/analyze-ppe-image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          module: moduleId,
          image_base64: prepared.imageBase64,
          mime_type: prepared.mimeType,
        }),
      });

      if (!response.ok) return null;
      const data = (await response.json()) as { items?: Partial<PpeAssessment> };
      return data.items ?? null;
    } catch (geminiError) {
      console.warn('Análise complementar de EPI indisponível', geminiError);
      return null;
    }
  }, [moduleId]);

  const runFrame = useCallback(async (timestamp: number) => {
    if (!runningRef.current) return;
    const video = videoRef.current, detector = detectorRef.current;
    if (!video || !detector || video.readyState < 2) {
      scheduleNextFrame();
      return;
    }
    if (timestamp - lastInferenceRef.current < 105) {
      scheduleNextFrame();
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

    scheduleNextFrame();
  }, [backendOnline, drawOverlay, moduleId, riskSide, riskWidth, scheduleNextFrame, sendToBackend]);

  const analyzeImage = useCallback(async (image: HTMLImageElement) => {
    if (!image.naturalWidth || !image.naturalHeight) return;

    setError(null);
    setImageAnalyzing(true);
    setSummary('Analisando a imagem selecionada…');

    try {
      const detector = await ensureDetector();
      if (!detector) throw new Error('Análise indisponível');

      const poses = await detector.estimatePoses(image, { maxPoses: 6, flipHorizontal: false });
      setPeopleCount(poses.length);

      const primary = primaryPose(poses);
      const currentMetrics = primary ? calculatePoseMetrics(primary) : null;
      setMetrics(currentMetrics);

      let currentPpe: PpeAssessment | null = null;
      if (moduleId === 'epi' || moduleId === 'altura') {
        const localPpe = primary ? inspectPpe(image, primary) : null;
        const geminiPpe = await analyzePpeWithGemini(image);
        currentPpe = mergePpeAssessments(localPpe, geminiPpe);
      }
      currentPpeRef.current = currentPpe;
      setPpe(currentPpe);

      const currentZoneRisk = moduleId === 'altura'
        ? isPoseNearRiskZone(currentMetrics, image.naturalWidth, riskSide, riskWidth)
        : false;
      setZoneRisk(currentZoneRisk);

      const currentLiftingPhase = moduleId === 'cargas' ? inferLiftingPhase(currentMetrics) : null;
      setLiftingPhase(currentLiftingPhase);

      drawOverlay(
        poses,
        currentMetrics,
        currentZoneRisk,
        image.naturalWidth,
        image.naturalHeight,
      );

      const payload: EvaluationPayload = {
        module: moduleId,
        metrics: currentMetrics,
        ppe: currentPpe,
        zoneRisk: currentZoneRisk,
        liftingPhase: currentLiftingPhase,
      };

      const local = evaluateLocally(payload);
      setFindings(local.findings);
      setSummary(local.summary);
      setBackendOnline(null);
      await sendToBackend(payload);
    } catch (imageError) {
      console.error('Falha ao analisar imagem', imageError);
      setFindings([]);
      setSummary('Não foi possível analisar esta imagem.');
      setError('Não foi possível analisar a imagem. Tente outra foto com a pessoa inteira e boa iluminação.');
      clearOverlay();
    } finally {
      setImageAnalyzing(false);
    }
  }, [analyzePpeWithGemini, clearOverlay, drawOverlay, ensureDetector, moduleId, riskSide, riskWidth, sendToBackend]);

  const handleImageSelection = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    event.target.value = '';
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Selecione um arquivo de imagem válido.');
      return;
    }

    if (file.size > 12 * 1024 * 1024) {
      setError('A imagem deve ter no máximo 12 MB.');
      return;
    }

    stopCamera();
    clearImage();
    setImageAnalyzing(true);
    setFindings([]);
    setMetrics(null);
    setPpe(null);
    setPeopleCount(0);
    setZoneRisk(false);
    setLiftingPhase(null);
    setBackendOnline(null);
    setError(null);

    const objectUrl = URL.createObjectURL(file);
    imageObjectUrlRef.current = objectUrl;
    setImageUrl(objectUrl);
    setImageName(file.name);
    setSummary('Imagem carregada. Preparando análise…');
  }, [clearImage, stopCamera]);

  const handleImageLoad = useCallback(() => {
    if (imageRef.current) void analyzeImage(imageRef.current);
  }, [analyzeImage]);

  const startCamera = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Este navegador não oferece acesso compatível à câmera.');
      return;
    }
    stopCamera();
    clearImage();
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
      scheduleNextFrame();
    } catch (cameraError) {
      console.error(cameraError);
      setError('Não foi possível iniciar a câmera. Verifique a permissão do navegador e tente novamente.');
      stopCamera();
    }
  }, [clearImage, ensureDetector, facingMode, scheduleNextFrame, stopCamera]);

  const switchCamera = useCallback(() => {
    setFacingMode((current) => current === 'environment' ? 'user' : 'environment');
    if (running) stopCamera();
  }, [running, stopCamera]);

  useEffect(() => {
    const resetFrame = requestAnimationFrame(() => {
      setFindings([]);
      currentPpeRef.current = null;
      setPpe(null);
      setMetrics(null);
      setZoneRisk(false);
      setLiftingPhase(null);

      if (imageUrl && imageRef.current?.complete) {
        setSummary('Módulo alterado. Reanalisando a imagem…');
        void analyzeImage(imageRef.current);
        return;
      }

      setSummary(
        running
          ? 'Módulo alterado. Recalculando análise…'
          : 'Ative a câmera ou selecione uma imagem para iniciar a análise.',
      );
    });

    return () => cancelAnimationFrame(resetFrame);
  }, [analyzeImage, imageUrl, moduleId, running]);

  useEffect(() => {
    runFrameRef.current = runFrame;
    if (!runningRef.current) return;

    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
    }

    scheduleNextFrame();

    return () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [runFrame, scheduleNextFrame]);

  useEffect(() => () => {
    runningRef.current = false;
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    detectorRef.current?.dispose?.();
    if (imageObjectUrlRef.current) URL.revokeObjectURL(imageObjectUrlRef.current);
  }, []);

  return (
    <section className="visionGrid">
      {imageAnalyzing && (
        <div className="analysisLoadingOverlay" role="status" aria-live="polite" aria-busy="true">
          <div className="analysisLoadingCard">
            <span className="analysisSpinner" aria-hidden="true" />
            <strong>Analisando imagem</strong>
            <p>Verificando pessoas, postura e equipamentos de proteção…</p>
          </div>
        </div>
      )}
      <div className="cameraCard">
        <div className="cameraToolbar">
          <div><p className="sectionKicker">ENTRADA PARA ANÁLISE</p><h3>{moduleTitles[moduleId]}</h3></div>
          <div className="cameraActions">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelection}
              style={{ display: 'none' }}
              aria-label="Selecionar imagem para análise"
            />
            <button type="button" className="secondaryButton" onClick={() => fileInputRef.current?.click()}>
              Selecionar imagem
            </button>
            {running ? (
              <>
                <button type="button" className="secondaryButton" onClick={switchCamera}>
                  {facingMode === 'environment' ? 'Câmera traseira' : 'Câmera frontal'}
                </button>
                <button type="button" className="dangerButton" onClick={stopCamera}>Encerrar</button>
              </>
            ) : (
              <button type="button" className="primaryButton" onClick={startCamera} disabled={loadingModel}>
                {loadingModel ? 'Preparando análise…' : imageUrl ? 'Usar câmera' : 'Ativar câmera'}
              </button>
            )}
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

        <div className={running || imageUrl ? 'cameraStage live' : 'cameraStage'}>
          <video ref={videoRef} muted playsInline className={imageUrl ? 'cameraVideo sourceHidden' : 'cameraVideo'} />
          {imageUrl && (
            <img
              ref={imageRef}
              src={imageUrl}
              alt={imageName ? `Imagem selecionada: ${imageName}` : 'Imagem selecionada para análise'}
              className="cameraImage"
              onLoad={handleImageLoad}
            />
          )}
          <canvas ref={canvasRef} className="cameraOverlay" />
          {!running && !imageUrl && (
            <div className="cameraEmpty">
              <div className="scannerIcon" aria-hidden="true"><span /></div>
              <strong>Escolha como deseja analisar</strong>
              <p>Use a câmera ao vivo ou selecione uma foto. A imagem é utilizada somente durante a análise e não fica armazenada.</p>
              <div className="emptyActions">
                <button type="button" className="primaryButton large" onClick={startCamera} disabled={loadingModel}>
                  {loadingModel ? 'Preparando análise…' : 'Abrir câmera'}
                </button>
                <button type="button" className="secondaryButton large" onClick={() => fileInputRef.current?.click()}>
                  Selecionar imagem
                </button>
              </div>
            </div>
          )}
          {running && <>
            <div className="liveBadge"><span />AO VIVO</div>
            <div className="moduleBadge">{moduleTitles[moduleId]}</div>
            {moduleId === 'altura' && <div className={zoneRisk ? 'riskBadge alert' : 'riskBadge'}>{zoneRisk ? 'ZONA DE RISCO' : 'ZONA MONITORADA'}</div>}
          </>}
          {!running && imageUrl && <>
            <div className="imageBadge">{imageAnalyzing ? 'ANALISANDO IMAGEM' : 'IMAGEM ANALISADA'}</div>
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
          <div className={running || imageUrl ? 'engineState online' : 'engineState'}>
            <span />
            {running ? 'Análise ativa' : imageAnalyzing ? 'Analisando imagem' : imageUrl ? 'Imagem analisada' : 'em espera'}
          </div>
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
            {ppeEntriesForModule(ppe, moduleId).map((entry) => (
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
            ? <div className="emptyFinding"><span className="pulseRing" /><p>{imageUrl ? 'Aguardando análise da imagem.' : 'Aguardando câmera ou imagem.'}</p></div>
            : findings.map((entry) => <article key={entry.code} className={'finding ' + entry.severity}><span className="findingTag">{severityLabel(entry.severity)}</span><strong>{entry.title}</strong><p>{entry.detail}</p></article>)}
        </div>

        <div className="privacyCard">
          <strong>Privacidade</strong>
          <p>Imagens da câmera e fotos selecionadas são usadas somente durante a análise e não ficam armazenadas.</p>
        </div>
      </aside>
    </section>
  );
}
