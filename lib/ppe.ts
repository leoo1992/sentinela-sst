import { bodyBox, namedPoint } from './geometry';
import type { DetectionStatus, Keypoint, PoseLike, PpeAssessment, PpeItem } from './types';

interface Region {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface RegionStats {
  skin: number;
  dark: number;
  saturated: number;
  edge: number;
  variance: number;
}

function item(label: string, status: DetectionStatus, confidence: number, note?: string): PpeItem {
  return { label, status, confidence: Math.max(0, Math.min(1, confidence)), note };
}

function unavailable(label: string, note: string) {
  return item(label, 'nao_avaliavel', 0, note);
}

function emptyAssessment(): PpeAssessment {
  return {
    oculos: unavailable('Óculos', 'Região dos olhos não localizada.'),
    capacete: unavailable('Capacete', 'Região da cabeça não localizada.'),
    luvas: unavailable('Luvas', 'Região das mãos não localizada.'),
    protetorAuricular: unavailable('Protetor auricular', 'Região das orelhas não localizada.'),
  };
}

function rgbToHsv(r: number, g: number, b: number) {
  const rp = r / 255;
  const gp = g / 255;
  const bp = b / 255;
  const max = Math.max(rp, gp, bp);
  const min = Math.min(rp, gp, bp);
  const delta = max - min;
  let h = 0;

  if (delta) {
    if (max === rp) h = ((gp - bp) / delta) % 6;
    else if (max === gp) h = (bp - rp) / delta + 2;
    else h = (rp - gp) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
  }

  return { h, s: max === 0 ? 0 : delta / max, v: max };
}

function isSkin(r: number, g: number, b: number) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  return r > 70 && g > 35 && b > 20 && max - min > 12 && r > g * 0.95 && r > b * 0.95;
}

function clampRegion(region: Region, width: number, height: number): Region {
  const x = Math.max(0, Math.min(width - 1, region.x));
  const y = Math.max(0, Math.min(height - 1, region.y));
  const right = Math.max(x + 1, Math.min(width, region.x + region.width));
  const bottom = Math.max(y + 1, Math.min(height, region.y + region.height));
  return { x, y, width: right - x, height: bottom - y };
}

function regionStats(
  context: CanvasRenderingContext2D,
  region: Region,
  width: number,
  height: number,
): RegionStats {
  const safe = clampRegion(region, width, height);
  const w = Math.max(2, Math.floor(safe.width));
  const h = Math.max(2, Math.floor(safe.height));
  const image = context.getImageData(Math.floor(safe.x), Math.floor(safe.y), w, h);
  const data = image.data;

  let skin = 0;
  let dark = 0;
  let saturated = 0;
  let count = 0;
  let sum = 0;
  let sumSq = 0;
  let edges = 0;
  let edgeChecks = 0;

  const luminance = new Float32Array(w * h);

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const pixel = (y * w + x) * 4;
      const r = data[pixel];
      const g = data[pixel + 1];
      const b = data[pixel + 2];
      const hsv = rgbToHsv(r, g, b);
      const luma = r * 0.299 + g * 0.587 + b * 0.114;

      luminance[y * w + x] = luma;
      if (isSkin(r, g, b)) skin += 1;
      if (hsv.v < 0.30) dark += 1;
      if (hsv.s > 0.30) saturated += 1;
      sum += luma;
      sumSq += luma * luma;
      count += 1;
    }
  }

  for (let y = 1; y < h; y += 2) {
    for (let x = 1; x < w; x += 2) {
      const here = luminance[y * w + x];
      const left = luminance[y * w + x - 1];
      const top = luminance[(y - 1) * w + x];
      if (Math.abs(here - left) > 34 || Math.abs(here - top) > 34) edges += 1;
      edgeChecks += 1;
    }
  }

  const mean = count ? sum / count : 0;
  const variance = count ? Math.max(0, sumSq / count - mean * mean) : 0;

  return {
    skin: count ? skin / count : 0,
    dark: count ? dark / count : 0,
    saturated: count ? saturated / count : 0,
    edge: edgeChecks ? edges / edgeChecks : 0,
    variance,
  };
}

function pointRegion(point: Keypoint | null, radius: number): Region | null {
  if (!point) return null;
  return {
    x: point.x - radius,
    y: point.y - radius,
    width: radius * 2,
    height: radius * 2,
  };
}

function detectHelmet(stats: RegionStats): PpeItem {
  const objectSignal =
    stats.edge * 1.6 +
    stats.saturated * 0.8 +
    stats.dark * 0.45 +
    Math.min(1, stats.variance / 1800) * 0.7;

  if (stats.skin < 0.42 && objectSignal > 0.62) {
    return item(
      'Capacete',
      'detectado',
      Math.min(0.92, 0.58 + objectSignal * 0.22),
      'Objeto consistente com capacete identificado na região da cabeça, sem exigir cor ou modelo específico.',
    );
  }

  if (objectSignal > 0.42) {
    return item('Capacete', 'incerto', 0.48, 'Há um objeto na região da cabeça, mas a confirmação é inconclusiva.');
  }

  return item('Capacete', 'nao_detectado', 0.56, 'Nenhum objeto consistente com capacete foi identificado na cabeça.');
}

function detectGlasses(stats: RegionStats): PpeItem {
  const signal = stats.edge * 1.7 + stats.dark * 0.65 + Math.min(1, stats.variance / 1300) * 0.55;

  if (signal > 0.72) {
    return item(
      'Óculos',
      'detectado',
      Math.min(0.88, 0.57 + signal * 0.20),
      'Estrutura compatível com óculos identificada ao redor dos olhos, independentemente do tipo.',
    );
  }

  if (signal > 0.52) {
    return item('Óculos', 'incerto', 0.46, 'Há contornos na região dos olhos, mas a identificação não é conclusiva.');
  }

  return item('Óculos', 'nao_detectado', 0.54, 'Óculos não foram identificados na região dos olhos.');
}

function detectGloves(regions: RegionStats[]): PpeItem {
  if (!regions.length) return unavailable('Luvas', 'Mãos não localizadas.');

  const skin = regions.reduce((sum, value) => sum + value.skin, 0) / regions.length;
  const material = regions.reduce(
    (sum, value) =>
      sum +
      value.saturated * 0.65 +
      value.dark * 0.45 +
      value.edge * 0.75 +
      Math.min(1, value.variance / 1600) * 0.45,
    0,
  ) / regions.length;

  if (skin < 0.38 && material > 0.54) {
    return item(
      'Luvas',
      'detectado',
      Math.min(0.90, 0.58 + material * 0.20),
      'Material diferente de pele identificado nas mãos, independentemente do tipo de luva.',
    );
  }

  if (skin > 0.56) {
    return item('Luvas', 'nao_detectado', 0.62, 'As mãos apresentam forte padrão de pele exposta.');
  }

  return item('Luvas', 'incerto', 0.44, 'As mãos estão visíveis, mas não foi possível confirmar luvas.');
}

function detectHearingProtection(regions: RegionStats[]): PpeItem {
  if (!regions.length) return unavailable('Protetor auricular', 'Orelhas não localizadas.');

  const signals = regions.map(
    (value) =>
      (1 - value.skin) * 0.45 +
      value.saturated * 0.55 +
      value.dark * 0.45 +
      value.edge * 0.85 +
      Math.min(1, value.variance / 1700) * 0.40,
  );

  const strongest = Math.max(...signals);
  if (strongest > 0.76) {
    return item(
      'Protetor auricular',
      'detectado',
      Math.min(0.88, 0.56 + strongest * 0.22),
      'Objeto/material compatível com plug ou abafador identificado na região de uma ou ambas as orelhas.',
    );
  }

  if (strongest > 0.56) {
    return item(
      'Protetor auricular',
      'incerto',
      0.46,
      'Há alteração visual na região das orelhas, mas não foi possível confirmar plug ou abafador.',
    );
  }

  return item('Protetor auricular', 'nao_detectado', 0.55, 'Proteção auditiva não foi identificada na região das orelhas.');
}

export function inspectPpe(source: HTMLVideoElement | HTMLImageElement, pose: PoseLike): PpeAssessment {
  const result = emptyAssessment();
  const box = bodyBox(pose);
  const sourceWidth = 'videoWidth' in source ? source.videoWidth : source.naturalWidth;
  const sourceHeight = 'videoHeight' in source ? source.videoHeight : source.naturalHeight;

  if (!box || sourceWidth <= 0 || sourceHeight <= 0) return result;

  const targetWidth = 420;
  const scale = targetWidth / sourceWidth;
  const targetHeight = Math.max(1, Math.round(sourceHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return result;

  context.drawImage(source, 0, 0, targetWidth, targetHeight);

  const mapPoint = (name: string): Keypoint | null => {
    const value = namedPoint(pose, name);
    return value ? { ...value, x: value.x * scale, y: value.y * scale } : null;
  };

  const bodyHeight = Math.max(80, box.height * scale);
  const nose = mapPoint('nose');
  const leftEye = mapPoint('left_eye');
  const rightEye = mapPoint('right_eye');
  const leftEar = mapPoint('left_ear');
  const rightEar = mapPoint('right_ear');
  const leftWrist = mapPoint('left_wrist');
  const rightWrist = mapPoint('right_wrist');

  const faceCenter = nose ?? leftEye ?? rightEye;
  if (faceCenter) {
    const helmetStats = regionStats(
      context,
      {
        x: faceCenter.x - bodyHeight * 0.12,
        y: faceCenter.y - bodyHeight * 0.21,
        width: bodyHeight * 0.24,
        height: bodyHeight * 0.18,
      },
      targetWidth,
      targetHeight,
    );
    result.capacete = detectHelmet(helmetStats);
  }

  if (leftEye && rightEye) {
    const eyeDistance = Math.max(12, Math.abs(rightEye.x - leftEye.x));
    const glassesStats = regionStats(
      context,
      {
        x: Math.min(leftEye.x, rightEye.x) - eyeDistance * 0.50,
        y: Math.min(leftEye.y, rightEye.y) - eyeDistance * 0.42,
        width: eyeDistance * 2.0,
        height: eyeDistance * 0.90,
      },
      targetWidth,
      targetHeight,
    );
    result.oculos = detectGlasses(glassesStats);
  }

  const handRadius = Math.max(12, bodyHeight * 0.065);
  const handStats = [leftWrist, rightWrist]
    .map((point) => pointRegion(point, handRadius))
    .filter((region): region is Region => Boolean(region))
    .map((region) => regionStats(context, region, targetWidth, targetHeight));
  result.luvas = detectGloves(handStats);

  const earRadius = Math.max(10, bodyHeight * 0.055);
  const earStats = [leftEar, rightEar]
    .map((point) => pointRegion(point, earRadius))
    .filter((region): region is Region => Boolean(region))
    .map((region) => regionStats(context, region, targetWidth, targetHeight));
  result.protetorAuricular = detectHearingProtection(earStats);

  return result;
}

export function inspectPpeWithoutPose(source: HTMLImageElement): PpeAssessment {
  const result = emptyAssessment();
  const width = source.naturalWidth;
  const height = source.naturalHeight;
  if (width <= 0 || height <= 0) return result;

  const targetWidth = 420;
  const scale = targetWidth / width;
  const targetHeight = Math.max(1, Math.round(height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) return result;

  context.drawImage(source, 0, 0, targetWidth, targetHeight);

  // Sem pose, só fazemos uma triagem conservadora de capacete e luvas.
  const top = regionStats(
    context,
    { x: targetWidth * 0.20, y: 0, width: targetWidth * 0.60, height: targetHeight * 0.34 },
    targetWidth,
    targetHeight,
  );
  const lower = regionStats(
    context,
    { x: targetWidth * 0.14, y: targetHeight * 0.48, width: targetWidth * 0.72, height: targetHeight * 0.38 },
    targetWidth,
    targetHeight,
  );

  result.capacete = detectHelmet(top);
  if (lower.skin < 0.34 && (lower.saturated + lower.dark + lower.edge) > 0.62) {
    result.luvas = item('Luvas', 'incerto', 0.44, 'Há material compatível com luvas, mas as mãos não foram localizadas.');
  }

  return result;
}
