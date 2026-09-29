import type { EvaluationPayload, EvaluationResponse, Finding, PpeAssessment, PpeItem } from './types';

function finding(code: string, title: string, detail: string, severity: Finding['severity']): Finding {
  return { code, title, detail, severity };
}

function statusFinding(key: keyof PpeAssessment, value: PpeItem): Finding {
  if (value.status === 'detectado') return finding('epi-' + key + '-ok', value.label + ' detectado', value.note ?? 'EPI visualmente identificado.', 'ok');
  if (value.status === 'nao_detectado') return finding('epi-' + key + '-missing', value.label + ' não detectado', value.note ?? 'EPI não foi identificado na imagem.', 'alert');
  if (value.status === 'incerto') return finding('epi-' + key + '-uncertain', value.label + ' inconclusivo', value.note ?? 'É necessária confirmação visual.', 'attention');
  return finding('epi-' + key + '-unavailable', value.label + ' não avaliável', value.note ?? 'A imagem não fornece informação suficiente.', 'info');
}

export function evaluateLocally(payload: EvaluationPayload): EvaluationResponse {
  const findings: Finding[] = [];

  if (!payload.metrics) {
    findings.push(finding(
      'pose-missing',
      'Pessoa parcialmente localizada',
      'A pose corporal não pôde ser calculada por completo, mas os EPIs visíveis ainda podem ser avaliados.',
      'info',
    ));
  }

  if (!payload.ppe) {
    findings.push(finding(
      'ppe-waiting',
      'Preparando inspeção',
      'Mantenha cabeça, olhos e mãos visíveis.',
      'info',
    ));
  } else {
    const epiKeys: Array<keyof PpeAssessment> = ['capacete', 'oculos', 'luvas'];
    epiKeys.forEach((key) => findings.push(statusFinding(key, payload.ppe![key])));
  }

  const alertCount = findings.filter((entry) => entry.severity === 'alert').length;
  const attentionCount = findings.filter((entry) => entry.severity === 'attention').length;

  return {
    module: 'epi',
    findings,
    summary: alertCount > 0
      ? alertCount + ' EPI(s) não identificado(s).'
      : attentionCount > 0
        ? attentionCount + ' item(ns) inconclusivo(s).'
        : 'Inspeção visual concluída.',
  };
}
