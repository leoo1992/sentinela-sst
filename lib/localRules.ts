import type { EvaluationPayload, EvaluationResponse, Finding, PpeAssessment, PpeItem } from './types';

function finding(code: string, title: string, detail: string, severity: Finding['severity']): Finding {
  return { code, title, detail, severity };
}

function statusFinding(key: keyof PpeAssessment, value: PpeItem): Finding {
  if (value.status === 'detectado') return finding('epi-' + key + '-ok', value.label + ' detectado', value.note ?? 'EPI visualmente identificado.', 'ok');
  if (value.status === 'nao_detectado') return finding('epi-' + key + '-missing', value.label + ' não detectado', value.note ?? 'EPI não foi identificado na imagem.', 'alert');
  if (value.status === 'incerto') return finding('epi-' + key + '-uncertain', value.label + ' inconclusivo', value.note ?? 'É necessária confirmação visual.', 'attention');
  return finding('epi-' + key + '-unavailable', value.label + ' não avaliável', value.note ?? 'A câmera não fornece informação suficiente.', 'info');
}

export function evaluateLocally(payload: EvaluationPayload): EvaluationResponse {
  const findings: Finding[] = [];
  const metrics = payload.metrics;

  if (!metrics) {
    findings.push(finding('pose-missing', 'Pessoa não localizada', 'Posicione o corpo dentro do enquadramento para iniciar a análise.', 'info'));
    return { module: payload.module, findings, summary: 'Aguardando uma pessoa no enquadramento.' };
  }

  if (payload.module === 'epi') {
    if (!payload.ppe) {
      findings.push(finding('ppe-waiting', 'Preparando inspeção', 'Mantenha cabeça, tronco, mãos e pés visíveis por alguns instantes.', 'info'));
    } else {
      (Object.keys(payload.ppe) as Array<keyof PpeAssessment>).forEach((key) => findings.push(statusFinding(key, payload.ppe![key])));
    }
  }

  if (payload.module === 'altura') {
    findings.push(payload.zoneRisk
      ? finding('height-zone', 'Pessoa dentro da zona de risco', 'A projeção do corpo alcançou a área virtual de borda configurada.', 'alert')
      : finding('height-zone-ok', 'Fora da zona de risco', 'A pessoa está fora da área virtual de borda configurada.', 'ok'));
    if (payload.ppe?.capacete) findings.push(statusFinding('capacete', payload.ppe.capacete));
    findings.push(finding('height-harness', 'Cinturão e talabarte', 'A confirmação de ancoragem exige inspeção visual específica e não é certificada por este protótipo.', 'info'));
  }

  if (payload.module === 'ergonomia') {
    if (metrics.trunkInclination !== null) {
      if (metrics.trunkInclination >= 45) findings.push(finding('ergo-trunk-high', 'Inclinação elevada do tronco', 'Tronco estimado em ' + metrics.trunkInclination + '° em relação à vertical.', 'alert'));
      else if (metrics.trunkInclination >= 22) findings.push(finding('ergo-trunk-medium', 'Inclinação moderada do tronco', 'Tronco estimado em ' + metrics.trunkInclination + '°.', 'attention'));
      else findings.push(finding('ergo-trunk-ok', 'Tronco próximo da vertical', 'Inclinação estimada em ' + metrics.trunkInclination + '°.', 'ok'));
    }
    if (metrics.neckInclination !== null && metrics.neckInclination >= 25) findings.push(finding('ergo-neck', 'Inclinação do pescoço', 'Inclinação estimada em ' + metrics.neckInclination + '°.', metrics.neckInclination >= 40 ? 'alert' : 'attention'));
    if (metrics.kneeAsymmetry !== null && metrics.kneeAsymmetry >= 16) findings.push(finding('ergo-knee-asymmetry', 'Assimetria entre joelhos', 'Diferença angular estimada de ' + metrics.kneeAsymmetry + '°.', 'attention'));
  }

  if (payload.module === 'cargas') {
    if (payload.liftingPhase) findings.push(finding('lift-phase', payload.liftingPhase, 'Fase estimada a partir dos ângulos de quadril, joelho e tronco.', 'info'));
    if (metrics.trunkInclination !== null && metrics.trunkInclination >= 42) findings.push(finding('lift-trunk', 'Flexão elevada do tronco', 'Inclinação estimada em ' + metrics.trunkInclination + '° durante o movimento.', 'alert'));
    else if (metrics.trunkInclination !== null) findings.push(finding('lift-trunk-ok', 'Controle do tronco', 'Inclinação estimada em ' + metrics.trunkInclination + '°.', metrics.trunkInclination >= 25 ? 'attention' : 'ok'));

    const knees = [metrics.leftKneeAngle, metrics.rightKneeAngle].filter((value): value is number => value !== null);
    if (knees.length) {
      const average = Math.round(knees.reduce((sum, value) => sum + value, 0) / knees.length);
      if (average > 158 && (metrics.trunkInclination ?? 0) > 28) findings.push(finding('lift-knees', 'Pouca flexão dos joelhos', 'Ângulo médio dos joelhos em ' + average + '° com inclinação do tronco.', 'attention'));
      else findings.push(finding('lift-knees-ok', 'Participação dos joelhos', 'Ângulo médio estimado em ' + average + '°.', 'ok'));
    }
    if (metrics.hipAsymmetry !== null && metrics.hipAsymmetry >= 14) findings.push(finding('lift-asymmetry', 'Assimetria durante o levantamento', 'Diferença angular de quadril estimada em ' + metrics.hipAsymmetry + '°.', 'attention'));
  }

  if (!findings.length) findings.push(finding('analysis-normal', 'Análise ativa', 'Nenhum indicador adicional foi gerado neste instante.', 'ok'));

  const alertCount = findings.filter((entry) => entry.severity === 'alert').length;
  const attentionCount = findings.filter((entry) => entry.severity === 'attention').length;
  return {
    module: payload.module,
    findings,
    summary: alertCount > 0 ? alertCount + ' alerta(s) visual(is) no quadro atual.' : attentionCount > 0 ? attentionCount + ' ponto(s) para atenção.' : 'Nenhum alerta visual crítico no quadro atual.',
  };
}
