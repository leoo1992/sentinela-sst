import { describe, expect, it } from 'vitest';
import { evaluateLocally } from '@/lib/localRules';

describe('local rules', () => {
  it('gera alerta para alta inclinação do tronco', () => {
    const result = evaluateLocally({
      module: 'ergonomia',
      metrics: {
        trunkInclination: 51, neckInclination: 12, leftKneeAngle: 165, rightKneeAngle: 164,
        leftHipAngle: 148, rightHipAngle: 150, shoulderTilt: 3, hipTilt: 2,
        kneeAsymmetry: 1, hipAsymmetry: 2, bodyBox: null, visibility: 95,
      },
      ppe: null, zoneRisk: false, liftingPhase: null,
    });
    expect(result.findings.some((entry) => entry.code === 'ergo-trunk-high')).toBe(true);
  });

  it('gera alerta ao entrar na zona de altura', () => {
    const result = evaluateLocally({
      module: 'altura',
      metrics: {
        trunkInclination: 10, neckInclination: 5, leftKneeAngle: 170, rightKneeAngle: 170,
        leftHipAngle: 170, rightHipAngle: 170, shoulderTilt: 0, hipTilt: 0,
        kneeAsymmetry: 0, hipAsymmetry: 0, bodyBox: { x: 100, y: 100, width: 200, height: 500 }, visibility: 95,
      },
      ppe: null, zoneRisk: true, liftingPhase: null,
    });
    expect(result.findings[0].severity).toBe('alert');
  });
});
