export type ModuleId = 'epi';
export type Severity = 'ok' | 'info' | 'attention' | 'alert';
export type DetectionStatus = 'detectado' | 'nao_detectado' | 'incerto' | 'nao_avaliavel';

export interface Keypoint {
  x: number;
  y: number;
  score?: number;
  name?: string;
}

export interface PoseLike {
  keypoints: Keypoint[];
  score?: number;
}

export interface BodyBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PoseMetrics {
  trunkInclination: number | null;
  neckInclination: number | null;
  leftKneeAngle: number | null;
  rightKneeAngle: number | null;
  leftHipAngle: number | null;
  rightHipAngle: number | null;
  shoulderTilt: number | null;
  hipTilt: number | null;
  kneeAsymmetry: number | null;
  hipAsymmetry: number | null;
  bodyBox: BodyBox | null;
  visibility: number;
}

export interface PpeItem {
  label: string;
  status: DetectionStatus;
  confidence: number;
  note?: string;
}

export interface PpeAssessment {
  capacete: PpeItem;
  oculos: PpeItem;
  luvas: PpeItem;
  protetorAuricular: PpeItem;
}

export interface Finding {
  code: string;
  title: string;
  detail: string;
  severity: Severity;
}

export interface EvaluationPayload {
  module: ModuleId;
  metrics: PoseMetrics | null;
  ppe: PpeAssessment | null;
  zoneRisk: false;
  liftingPhase: null;
}

export interface EvaluationResponse {
  module: ModuleId;
  findings: Finding[];
  summary: string;
}
