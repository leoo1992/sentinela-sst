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
  const [fps, setFps] = useState(0);
  const [peopleCount, setPeopleCount] = useState(0);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [backendLatency, setBackendLatency] = useState<number | null>(null);
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
        value: alertCount > 0 ? 'Alerta' : attentionCount > 0 ? 'Atenção' : running ? 'Normal' : 'Em espera',
      },
      { label: 'Alertas', value: String(alertCount + attentionCount) },
      { label: 'EPIs', value: moduleId === 'epi' || moduleId === 'altura' ? ppeLabel(ppe) : '—' },
    ];
  }, [findings, moduleId, peopleCount, ppe, running]);
}
