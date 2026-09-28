import type { BodyBox, Keypoint, PoseLike, PoseMetrics } from './types';

const MIN_SCORE = 0.28;

export function namedPoint(pose: PoseLike, name: string): Keypoint | null {
  const point = pose.keypoints.find((item) => item.name === name);
  if (!point || (point.score ?? 1) < MIN_SCORE) return null;
  return point;
}

export function angleBetween(a: Keypoint | null, b: Keypoint | null, c: Keypoint | null): number | null {
  if (!a || !b || !c) return null;
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const cbx = c.x - b.x;
  const cby = c.y - b.y;
  const denominator = Math.hypot(abx, aby) * Math.hypot(cbx, cby);
  if (denominator === 0) return null;
  const cosine = Math.min(1, Math.max(-1, (abx * cbx + aby * cby) / denominator));
  return Math.round((Math.acos(cosine) * 180) / Math.PI);
}

export function midpoint(a: Keypoint | null, b: Keypoint | null): Keypoint | null {
  if (!a || !b) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, score: Math.min(a.score ?? 1, b.score ?? 1) };
}

export function verticalInclination(upper: Keypoint | null, lower: Keypoint | null): number | null {
  if (!upper || !lower) return null;
  const dx = upper.x - lower.x;
  const dy = lower.y - upper.y;
  if (dx === 0 && dy === 0) return null;
  return Math.round(Math.abs((Math.atan2(dx, dy) * 180) / Math.PI));
}

export function lineTilt(a: Keypoint | null, b: Keypoint | null): number | null {
  if (!a || !b) return null;
  return Math.round(Math.abs((Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI));
}

export function bodyBox(pose: PoseLike): BodyBox | null {
  const valid = pose.keypoints.filter((point) => (point.score ?? 1) >= MIN_SCORE && Number.isFinite(point.x) && Number.isFinite(point.y));
  if (valid.length < 5) return null;
  const xs = valid.map((point) => point.x);
  const ys = valid.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  return { x: minX, y: minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) };
}

export function poseArea(pose: PoseLike): number {
  const box = bodyBox(pose);
  return box ? box.width * box.height : 0;
}

export function primaryPose(poses: PoseLike[]): PoseLike | null {
  if (!poses.length) return null;
  return [...poses].sort((a, b) => poseArea(b) - poseArea(a))[0] ?? null;
}

export function calculatePoseMetrics(pose: PoseLike): PoseMetrics {
  const leftShoulder = namedPoint(pose, 'left_shoulder');
  const rightShoulder = namedPoint(pose, 'right_shoulder');
  const leftHip = namedPoint(pose, 'left_hip');
  const rightHip = namedPoint(pose, 'right_hip');
  const leftKnee = namedPoint(pose, 'left_knee');
  const rightKnee = namedPoint(pose, 'right_knee');
  const leftAnkle = namedPoint(pose, 'left_ankle');
  const rightAnkle = namedPoint(pose, 'right_ankle');
  const nose = namedPoint(pose, 'nose');
  const shoulders = midpoint(leftShoulder, rightShoulder);
  const hips = midpoint(leftHip, rightHip);
  const leftHipAngle = angleBetween(leftShoulder, leftHip, leftKnee);
  const rightHipAngle = angleBetween(rightShoulder, rightHip, rightKnee);
  const leftKneeAngle = angleBetween(leftHip, leftKnee, leftAnkle);
  const rightKneeAngle = angleBetween(rightHip, rightKnee, rightAnkle);
  const visible = pose.keypoints.filter((point) => (point.score ?? 1) >= MIN_SCORE).length;
  const total = Math.max(1, pose.keypoints.length);

  return {
    trunkInclination: verticalInclination(shoulders, hips),
    neckInclination: verticalInclination(nose, shoulders),
    leftKneeAngle,
    rightKneeAngle,
    leftHipAngle,
    rightHipAngle,
    shoulderTilt: lineTilt(leftShoulder, rightShoulder),
    hipTilt: lineTilt(leftHip, rightHip),
    kneeAsymmetry: leftKneeAngle !== null && rightKneeAngle !== null ? Math.abs(leftKneeAngle - rightKneeAngle) : null,
    hipAsymmetry: leftHipAngle !== null && rightHipAngle !== null ? Math.abs(leftHipAngle - rightHipAngle) : null,
    bodyBox: bodyBox(pose),
    visibility: Math.round((visible / total) * 100),
  };
}

export function inferLiftingPhase(metrics: PoseMetrics | null): string | null {
  if (!metrics) return null;
  const knees = [metrics.leftKneeAngle, metrics.rightKneeAngle].filter((value): value is number => value !== null);
  const knee = knees.length ? knees.reduce((sum, value) => sum + value, 0) / knees.length : null;
  const trunk = metrics.trunkInclination;
  if (knee === null || trunk === null) return 'Aguardando corpo completo';
  if (knee < 105) return 'Agachamento / preparação';
  if (knee < 150 && trunk > 18) return 'Transição da carga';
  if (trunk > 30) return 'Inclinação com carga';
  return 'Postura ereta';
}

export function isPoseNearRiskZone(metrics: PoseMetrics | null, frameWidth: number, side: 'left' | 'right', zonePercent: number): boolean {
  if (!metrics?.bodyBox || frameWidth <= 0) return false;
  const zoneWidth = frameWidth * (zonePercent / 100);
  const box = metrics.bodyBox;
  return side === 'left' ? box.x < zoneWidth : box.x + box.width > frameWidth - zoneWidth;
}
