import { describe, expect, it } from 'vitest';
import { evaluateLocally } from '../lib/localRules';

describe('inspeção local de EPI', () => {
  it('mantém avaliação dos três EPIs mesmo sem pose corporal completa', () => {
    const result = evaluateLocally({
      module: 'epi',
      metrics: null,
      ppe: {
        capacete: { label: 'Capacete', status: 'detectado', confidence: 0.95 },
        oculos: { label: 'Óculos', status: 'detectado', confidence: 0.88 },
        luvas: { label: 'Luvas', status: 'detectado', confidence: 0.86 },
      },
      zoneRisk: false,
      liftingPhase: null,
    });

    expect(result.findings.some((entry) => entry.code === 'epi-capacete-ok')).toBe(true);
    expect(result.findings.some((entry) => entry.code === 'epi-oculos-ok')).toBe(true);
    expect(result.findings.some((entry) => entry.code === 'epi-luvas-ok')).toBe(true);
  });
});
