'use client';

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react';
import { primaryPose } from '@/lib/geometry';
import { inspectPpe, inspectPpeWithoutPose } from '@/lib/ppe';
import type { PoseLike, PpeAssessment, PpeItem } from '@/lib/types';

type FacingMode = 'user' | 'environment';
type VisionSource = HTMLVideoElement | HTMLImageElement;
type PoseDetectorLike = {
  estimatePoses: (
    input: VisionSource,
    config?: { maxPoses?: number; flipHorizontal?: boolean },
  ) => Promise<PoseLike[]>;
  dispose?: () => void;
};

type CombinedResponse = {
  items?: Partial<PpeAssessment>;
  person_detected?: boolean;
  engines?: {
    tensorflow?: boolean;
    opencv?: boolean;
    gemini?: boolean;
  };
  diagnostics?: {
    opencv_error?: string | null;
    gemini_error?: string | null;
  };
};

const EPI_KEYS: Array<keyof PpeAssessment> = [
  'capacete',
  'oculos',
  'colete',
  'luvas',
  'calcado',
];

const EPI_LABELS: Record<keyof PpeAssessment, string> = {
  capacete: 'Capacete',
  oculos: 'Óculos de proteção',
  colete: 'Colete/vestimenta refletiva',
  luvas: 'Luvas de proteção',
  calcado: 'Calçado fechado',
  cinturao: 'Cinturão paraquedista',
  talabarte: 'Talabarte',
  travaQuedas: 'Trava-quedas',
};

const STATUS_TEXT: Record<PpeItem['status'], string> = {
  detectado: 'Detectado',
  nao_detectado: 'Não detectado',
  incerto: 'Inconclusivo',
  nao_avaliavel: 'Não avaliável',
};

function unavailableItem(key: keyof PpeAssessment): PpeItem {
  return {
    label: EPI_LABELS[key],
    status: 'nao_avaliavel',
    confidence: 0,
    note: 'Não foi possível avaliar este item na imagem.',
  };
}

function emptyAssessment(): PpeAssessment {
  return {
    capacete: unavailableItem('capacete'),
    oculos: unavailableItem('oculos'),
    colete: unavailableItem('colete'),
    luvas: unavailableItem('luvas'),
    calcado: unavailableItem('calcado'),
    cinturao: unavailableItem('cinturao'),
    talabarte: unavailableItem('talabarte'),
    travaQuedas: unavailableItem('travaQuedas'),
  };
}

function mergeAssessment(
  local: PpeAssessment,
  remote?: Partial<PpeAssessment> | null,
): PpeAssessment {
  if (!remote) return local;

  const merged = { ...local };
  for (const key of EPI_KEYS) {
    const localItem = local[key];
    const remoteItem = remote[key];

    if (!remoteItem) continue;

    // Nunca perde uma detecção local forte por uma resposta remota inconclusiva.
    if (
      localItem.status === 'detectado' &&
      localItem.confidence >= 0.55 &&
      (remoteItem.status === 'incerto' || remoteItem.status === 'nao_avaliavel')
    ) {
      merged[key] = localItem;
      continue;
    }

    merged[key] = remoteItem;
  }
  return merged;
}

function detectedCount(ppe: PpeAssessment | null) {
  if (!ppe) return 0;
  return EPI_KEYS.filter((key) => ppe[key].status === 'detectado').length;
}

function situation(ppe: PpeAssessment | null) {
  if (!ppe) return 'Em espera';
  const items = EPI_KEYS.map((key) => ppe[key]);

  if (items.some((item) => item.status === 'nao_detectado')) return 'Alerta';
  if (items.some((item) => item.status === 'incerto')) return 'Atenção';
  if (items.every((item) => item.status === 'nao_avaliavel')) return 'Inconclusiva';
  if (items.some((item) => item.status === 'nao_avaliavel')) return 'Parcial';
  return 'Normal';
}

function normalizedPoseKeypoints(
  pose: PoseLike | null,
  width: number,
  height: number,
) {
  if (!pose || width <= 0 || height <= 0) return null;

  return Object.fromEntries(
    pose.keypoints
      .filter((point) => point.name)
      .map((point) => [
        point.name as string,
        {
          x: Math.max(0, Math.min(1, point.x / width)),
          y: Math.max(0, Math.min(1, point.y / height)),
          score: Math.max(0, Math.min(1, point.score ?? 0)),
        },
      ]),
  );
}

function imageForAnalysis(image: HTMLImageElement) {
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
  const dataUrl = canvas.toDataURL('image/jpeg', 0.86);
  const imageBase64 = dataUrl.split(',', 2)[1];

  if (!imageBase64) throw new Error('Não foi possível preparar a imagem.');

  return {
    imageBase64,
    mimeType: 'image/jpeg' as const,
  };
}

export default function VisionCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraDetectorRef = useRef<PoseDetectorLike | null>(null);
  const imageDetectorRef = useRef<PoseDetectorLike | null>(null);
  const animationRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const lastFrameRef = useRef(0);

  const [running, setRunning] = useState(false);
  const [facingMode, setFacingMode] = useState<FacingMode>('environment');
  const [loadingModel, setLoadingModel] = useState(false);
  const [imageAnalyzing, setImageAnalyzing] = useState(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageName, setImageName] = useState<string | null>(null);
  const [peopleCount, setPeopleCount] = useState(0);
  const [ppe, setPpe] = useState<PpeAssessment | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisNote, setAnalysisNote] = useState(
    'Envie uma imagem ou ative a câmera para iniciar a inspeção.',
  );

  const count = detectedCount(ppe);
  const currentSituation = situation(ppe);

  const statusCards = useMemo(
    () => [
      { label: 'Pessoas', value: String(peopleCount) },
      { label: 'Situação', value: currentSituation },
      { label: 'EPIs', value: ppe ? `${count} detectado${count === 1 ? '' : 's'}` : '—' },
      {
        label: 'Análise',
        value: imageAnalyzing ? 'Em andamento' : ppe ? 'Concluída' : 'Em espera',
      },
    ],
    [count, currentSituation, imageAnalyzing, peopleCount, ppe],
  );

  const clearImage = useCallback(() => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
    setImageUrl(null);
    setImageName(null);
    setImageAnalyzing(false);
  }, []);

  const stopCamera = useCallback(() => {
    runningRef.current = false;
    setRunning(false);

    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const ensureCameraDetector = useCallback(async () => {
    if (cameraDetectorRef.current) return cameraDetectorRef.current;

    setLoadingModel(true);
    try {
      const tf = await import('@tensorflow/tfjs-core');
      await import('@tensorflow/tfjs-backend-webgl');
      const poseDetection = await import('@tensorflow-models/pose-detection');

      await tf.setBackend('webgl');
      await tf.ready();

      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
        },
      );

      cameraDetectorRef.current = detector as unknown as PoseDetectorLike;
      return cameraDetectorRef.current;
    } finally {
      setLoadingModel(false);
    }
  }, []);

  const ensureImageDetector = useCallback(async () => {
    if (imageDetectorRef.current) return imageDetectorRef.current;

    setLoadingModel(true);
    try {
      const tf = await import('@tensorflow/tfjs-core');
      await import('@tensorflow/tfjs-backend-webgl');
      const poseDetection = await import('@tensorflow-models/pose-detection');

      await tf.setBackend('webgl');
      await tf.ready();

      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
          enableSmoothing: true,
        },
      );

      imageDetectorRef.current = detector as unknown as PoseDetectorLike;
      return imageDetectorRef.current;
    } finally {
      setLoadingModel(false);
    }
  }, []);

  const analyzeRemote = useCallback(async (
    image: HTMLImageElement,
    localPpe: PpeAssessment,
    pose: PoseLike | null,
  ) => {
    const prepared = imageForAnalysis(image);

    const response = await fetch('/api/analyze-ppe-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        module: 'epi',
        image_base64: prepared.imageBase64,
        mime_type: prepared.mimeType,
        local_ppe: localPpe,
        pose_keypoints: normalizedPoseKeypoints(
          pose,
          image.naturalWidth,
          image.naturalHeight,
        ),
      }),
    });

    if (!response.ok) {
      throw new Error(`Falha na análise de EPI (${response.status}).`);
    }

    const data = (await response.json()) as CombinedResponse;
    if (!data.items || Object.keys(data.items).length === 0) {
      throw new Error('A análise de EPI retornou um resultado vazio.');
    }

    return data;
  }, []);

  const analyzeImage = useCallback(async (image: HTMLImageElement) => {
    if (!image.naturalWidth || !image.naturalHeight) return;

    setImageAnalyzing(true);
    setError(null);
    setPpe(null);
    setPeopleCount(0);
    setAnalysisNote('Analisando a imagem e verificando os EPIs visíveis…');

    try {
      const detector = await ensureImageDetector();
      const poses = await detector.estimatePoses(image, {
        maxPoses: 1,
        flipHorizontal: false,
      });

      const pose = primaryPose(poses);
      const localPpe = pose
        ? inspectPpe(image, pose)
        : inspectPpeWithoutPose(image);

      let finalPpe = localPpe;
      let personDetected = Boolean(pose);

      try {
        const remote = await analyzeRemote(image, localPpe, pose);
        finalPpe = mergeAssessment(localPpe, remote.items);
        personDetected = personDetected || Boolean(remote.person_detected);

        const engines = remote.engines;
        const remoteWorked = Boolean(engines?.opencv || engines?.gemini);
        setAnalysisNote(
          remoteWorked
            ? 'Imagem analisada. Confira abaixo os EPIs identificados e os itens não avaliáveis.'
            : 'Imagem analisada localmente. A análise complementar não respondeu nesta tentativa.',
        );
      } catch (remoteError) {
        console.error('Falha na análise complementar de EPI', remoteError);
        setAnalysisNote(
          'A imagem foi analisada localmente, mas a análise complementar não respondeu. O resultado abaixo é parcial.',
        );
      }

      setPeopleCount(personDetected ? 1 : 0);
      setPpe(finalPpe);

      if (detectedCount(finalPpe) === 0) {
        const evaluable = EPI_KEYS.some(
          (key) => finalPpe[key].status !== 'nao_avaliavel',
        );
        if (!evaluable) {
          setError(
            'Não foi possível avaliar os EPIs nesta imagem. Tente uma foto mais nítida, com a pessoa maior no enquadramento.',
          );
        }
      }
    } catch (analysisError) {
      console.error('Falha ao analisar imagem', analysisError);
      setPpe(null);
      setPeopleCount(0);
      setError('Não foi possível concluir a análise desta imagem. Tente novamente com outra foto.');
      setAnalysisNote('Análise não concluída.');
    } finally {
      setImageAnalyzing(false);
    }
  }, [analyzeRemote, ensureImageDetector]);

  const handleImageSelection = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    event.target.value = '';

    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Selecione uma imagem válida.');
      return;
    }

    if (file.size > 12 * 1024 * 1024) {
      setError('A imagem deve ter no máximo 12 MB.');
      return;
    }

    stopCamera();
    clearImage();

    const objectUrl = URL.createObjectURL(file);
    objectUrlRef.current = objectUrl;
    setImageName(file.name);
    setImageUrl(objectUrl);
    setImageAnalyzing(true);
    setPpe(null);
    setPeopleCount(0);
    setError(null);
    setAnalysisNote('Preparando a imagem para análise…');
  }, [clearImage, stopCamera]);

  const handleImageLoad = useCallback(() => {
    if (imageRef.current) void analyzeImage(imageRef.current);
  }, [analyzeImage]);

  const runCameraFrame = useCallback(async (timestamp: number) => {
    if (!runningRef.current) return;

    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      animationRef.current = requestAnimationFrame(runCameraFrame);
      return;
    }

    if (timestamp - lastFrameRef.current < 300) {
      animationRef.current = requestAnimationFrame(runCameraFrame);
      return;
    }
    lastFrameRef.current = timestamp;

    try {
      const detector = cameraDetectorRef.current;
      if (!detector) {
        animationRef.current = requestAnimationFrame(runCameraFrame);
        return;
      }

      const poses = await detector.estimatePoses(video, {
        maxPoses: 1,
        flipHorizontal: facingMode === 'user',
      });

      const pose = primaryPose(poses);
      setPeopleCount(pose ? 1 : 0);

      if (pose) {
        const assessment = inspectPpe(video, pose);
        setPpe(assessment);
        setAnalysisNote('Análise ao vivo ativa.');
      } else {
        setPpe(null);
        setAnalysisNote('Posicione uma pessoa inteira ou parcialmente visível diante da câmera.');
      }
    } catch (cameraError) {
      console.error('Falha durante análise da câmera', cameraError);
    }

    animationRef.current = requestAnimationFrame(runCameraFrame);
  }, [facingMode]);

  const startCamera = useCallback(async (mode: FacingMode = facingMode) => {
    setError(null);
    clearImage();
    stopCamera();
    setPpe(null);
    setPeopleCount(0);
    setAnalysisNote('Preparando a câmera…');

    try {
      await ensureCameraDetector();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (!videoRef.current) throw new Error('Elemento de vídeo indisponível.');
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      runningRef.current = true;
      setRunning(true);
      setAnalysisNote('Análise ao vivo ativa.');
      animationRef.current = requestAnimationFrame(runCameraFrame);
    } catch (cameraError) {
      console.error('Falha ao abrir câmera', cameraError);
      stopCamera();
      setError('Não foi possível abrir a câmera. Verifique a permissão do navegador.');
      setAnalysisNote('Câmera indisponível.');
    }
  }, [clearImage, ensureCameraDetector, facingMode, runCameraFrame, stopCamera]);

  const switchCamera = useCallback(async () => {
    const next: FacingMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(next);

    if (runningRef.current) {
      await startCamera(next);
    }
  }, [facingMode, startCamera]);

  useEffect(() => () => {
    runningRef.current = false;
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    cameraDetectorRef.current?.dispose?.();
    imageDetectorRef.current?.dispose?.();
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
  }, []);

  return (
    <section className="visionGrid">
      {imageAnalyzing && (
        <div
          className="analysisLoadingOverlay"
          role="status"
          aria-live="polite"
          aria-busy="true"
        >
          <div className="analysisLoadingCard">
            <span className="analysisSpinner" aria-hidden="true" />
            <strong>Analisando imagem</strong>
            <p>Verificando capacete, óculos, vestimenta refletiva, luvas e calçado…</p>
          </div>
        </div>
      )}

      <div className="cameraCard">
        <div className="cameraToolbar">
          <div>
            <p className="sectionKicker">INSPEÇÃO DE EPI</p>
            <h3>Imagem ou câmera</h3>
          </div>

          <div className="cameraActions">
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelection}
              style={{ display: 'none' }}
              aria-label="Selecionar imagem para inspeção de EPI"
            />

            <button
              type="button"
              className="secondaryButton"
              onClick={() => inputRef.current?.click()}
            >
              Selecionar imagem
            </button>

            {running ? (
              <>
                <button type="button" className="secondaryButton" onClick={switchCamera}>
                  Trocar câmera
                </button>
                <button type="button" className="dangerButton" onClick={stopCamera}>
                  Encerrar
                </button>
              </>
            ) : (
              <button
                type="button"
                className="primaryButton"
                onClick={() => void startCamera()}
                disabled={loadingModel}
              >
                {loadingModel ? 'Preparando…' : 'Usar câmera'}
              </button>
            )}
          </div>
        </div>

        <div className={running || imageUrl ? 'cameraStage live' : 'cameraStage'}>
          <video
            ref={videoRef}
            muted
            playsInline
            className={imageUrl ? 'cameraVideo sourceHidden' : 'cameraVideo'}
          />

          {imageUrl && (
            <img
              ref={imageRef}
              src={imageUrl}
              alt={imageName ? `Imagem selecionada: ${imageName}` : 'Imagem para inspeção de EPI'}
              className="cameraImage"
              onLoad={handleImageLoad}
            />
          )}

          {!running && !imageUrl && (
            <div className="cameraEmpty">
              <div className="scannerIcon" aria-hidden="true">
                <span />
              </div>
              <strong>Inicie uma inspeção</strong>
              <p>
                Escolha uma foto ou use a câmera para identificar os EPIs visíveis.
              </p>
              <div className="emptyActions">
                <button
                  type="button"
                  className="primaryButton large"
                  onClick={() => void startCamera()}
                  disabled={loadingModel}
                >
                  {loadingModel ? 'Preparando…' : 'Abrir câmera'}
                </button>
                <button
                  type="button"
                  className="secondaryButton large"
                  onClick={() => inputRef.current?.click()}
                >
                  Selecionar imagem
                </button>
              </div>
            </div>
          )}

          {running && <div className="liveBadge"><span />AO VIVO</div>}
          {!running && imageUrl && (
            <div className="imageBadge">
              {imageAnalyzing ? 'ANALISANDO IMAGEM' : 'IMAGEM ANALISADA'}
            </div>
          )}
          {(running || imageUrl) && <div className="moduleBadge">Inspeção de EPI</div>}
        </div>

        {error && <div className="errorBanner">{error}</div>}

        <div className="statusStrip">
          {statusCards.map((card) => (
            <div key={card.label}>
              <span>{card.label}</span>
              <strong>{card.value}</strong>
            </div>
          ))}
        </div>
      </div>

      <aside className="inspectorCard">
        <div className="inspectorHeader">
          <div>
            <p className="sectionKicker">RESULTADO</p>
            <h3>EPIs identificados</h3>
          </div>
          <div className={running || imageUrl ? 'engineState online' : 'engineState'}>
            <span />
            {imageAnalyzing ? 'Analisando' : ppe ? 'Concluído' : 'Em espera'}
          </div>
        </div>

        <div className="summaryBox">
          <span>Leitura atual</span>
          <strong>{analysisNote}</strong>
        </div>

        <div className="ppeList">
          {EPI_KEYS.map((key) => {
            const item = ppe?.[key] ?? unavailableItem(key);
            return (
              <div key={key} className={`ppeRow ${item.status}`}>
                <span className="ppeState" />
                <div>
                  <strong>{EPI_LABELS[key]}</strong>
                  <small>{item.note ?? 'Aguardando análise.'}</small>
                </div>
                <b>{STATUS_TEXT[item.status]}</b>
              </div>
            );
          })}
        </div>

        <div className="privacyCard">
          <strong>Como interpretar</strong>
          <p>
            “Não avaliável” significa que o item ou a região necessária não está visível
            o suficiente. Isso não deve ser tratado como ausência do EPI.
          </p>
        </div>
      </aside>
    </section>
  );
}
