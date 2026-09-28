import { describe, expect, it } from 'vitest';
import { angleBetween, inferLiftingPhase } from '@/lib/geometry';
import type { PoseMetrics } from '@/lib/types';

describe('geometry', () => {
  it('calcula ângulo reto', () => {
    expect(angleBetween({ x: 0, y: 0 }, { x: 0, y: 1 }, { x: 1, y: 1 })).toBe(90);
  });

  it('classifica agachamento pela flexão dos joelhos', () => {
    const metrics: PoseMetrics = {
      trunkInclination: 20, neckInclination: 8, leftKneeAngle: 92, rightKneeAngle: 98,
      leftHipAngle: 90, rightHipAngle: 92, shoulderTilt: 3, hipTilt: 2,
      kneeAsymmetry: 6, hipAsymmetry: 2, bodyBox: null, visibility: 90,
    };
    expect(inferLiftingPhase(metrics)).toBe('Agachamento / preparação');
  });
});
