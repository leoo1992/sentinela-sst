import { describe, expect, it } from 'vitest';
import { evaluateLocally } from '../lib/localRules';

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


it('mantém avaliação de EPI mesmo sem pose corporal', () => {
  const result = evaluateLocally({
    module: 'epi',
    metrics: null,
    ppe: {
      capacete: { label: 'Capacete', status: 'detectado', confidence: 0.95 },
      oculos: { label: 'Óculos de proteção', status: 'detectado', confidence: 0.88 },
      colete: { label: 'Colete refletivo', status: 'detectado', confidence: 0.9 },
      luvas: { label: 'Luvas', status: 'detectado', confidence: 0.86 },
      calcado: { label: 'Calçado fechado', status: 'nao_avaliavel', confidence: 0.1 },
      cinturao: { label: 'Cinturão paraquedista', status: 'nao_avaliavel', confidence: 0 },
      talabarte: { label: 'Talabarte', status: 'nao_avaliavel', confidence: 0 },
      travaQuedas: { label: 'Trava-quedas', status: 'nao_avaliavel', confidence: 0 },
    },
    zoneRisk: false,
    liftingPhase: null,
  });

  expect(result.findings.some((entry) => entry.code === 'epi-capacete-ok')).toBe(true);
  expect(result.findings.some((entry) => entry.code === 'epi-oculos-ok')).toBe(true);
});
