import { bodyBox, namedPoint } from './geometry';
import type { DetectionStatus, Keypoint, PoseLike, PpeAssessment, PpeItem } from './types';

interface Region { x: number; y: number; width: number; height: number; }
interface SampleStats { highVis: number; dark: number; skin: number; saturated: number; count: number; }

function item(label: string, status: DetectionStatus, confidence: number, note?: string): PpeItem {
  return { label, status, confidence: Math.max(0, Math.min(1, confidence)), note };
}

function clampRegion(region: Region, width: number, height: number): Region {
  const x = Math.max(0, Math.min(width - 1, region.x));
  const y = Math.max(0, Math.min(height - 1, region.y));
  const right = Math.max(x + 1, Math.min(width, region.x + region.width));
  const bottom = Math.max(y + 1, Math.min(height, region.y + region.height));
  return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y) };
}

function rgbToHsv(r: number, g: number, b: number) {
  const rp = r / 255, gp = g / 255, bp = b / 255;
  const max = Math.max(rp, gp, bp), min = Math.min(rp, gp, bp), delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === rp) h = ((gp - bp) / delta) % 6;
    else if (max === gp) h = (bp - rp) / delta + 2;
    else h = (rp - gp) / delta + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  return { h, s: max === 0 ? 0 : delta / max, v: max };
}

function isSkin(r: number, g: number, b: number) {
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  return r > 70 && g > 35 && b > 20 && max - min > 12 && Math.abs(r - g) > 8 && r > g && r > b;
}

function statsForRegion(context: CanvasRenderingContext2D, region: Region, canvasWidth: number, canvasHeight: number): SampleStats {
  const safe = clampRegion(region, canvasWidth, canvasHeight);
  const data = context.getImageData(Math.floor(safe.x), Math.floor(safe.y), Math.max(1, Math.floor(safe.width)), Math.max(1, Math.floor(safe.height))).data;
  let highVis = 0, dark = 0, skin = 0, saturated = 0, count = 0;
  for (let index = 0; index < data.length; index += 16) {
    const r = data[index], g = data[index + 1], b = data[index + 2];
    const hsv = rgbToHsv(r, g, b);
    const yellow = hsv.h >= 40 && hsv.h <= 78 && hsv.s > 0.45 && hsv.v > 0.5;
    const orange = hsv.h >= 8 && hsv.h <= 38 && hsv.s > 0.55 && hsv.v > 0.5;
    const lime = hsv.h >= 79 && hsv.h <= 145 && hsv.s > 0.45 && hsv.v > 0.45;
    if (yellow || orange || lime) highVis += 1;
    if (hsv.v < 0.28) dark += 1;
    if (hsv.s > 0.38) saturated += 1;
    if (isSkin(r, g, b)) skin += 1;
    count += 1;
  }
  return { highVis, dark, skin, saturated, count };
}

function ratio(value: number, total: number) { return total ? value / total : 0; }

function pointRegion(point: Keypoint | null, bodyHeight: number, widthFactor: number, heightFactor: number): Region | null {
  if (!point) return null;
  return { x: point.x - bodyHeight * widthFactor * 0.5, y: point.y - bodyHeight * heightFactor * 0.5, width: bodyHeight * widthFactor, height: bodyHeight * heightFactor };
}

export function inspectPpe(video: HTMLVideoElement, pose: PoseLike): PpeAssessment {
  const box = bodyBox(pose);
  const unavailable = (label: string, note: string) => item(label, 'nao_avaliavel', 0, note);
  if (!box || video.videoWidth <= 0 || video.videoHeight <= 0) {
    return {
      capacete: unavailable('Capacete', 'Corpo insuficientemente visível.'),
      oculos: unavailable('Óculos de proteção', 'Face insuficientemente visível.'),
      colete: unavailable('Colete refletivo', 'Tronco insuficientemente visível.'),
      luvas: unavailable('Luvas', 'Mãos insuficientemente visíveis.'),
      calcado: unavailable('Calçado fechado', 'Pés insuficientemente visíveis.'),
    };
  }

  const targetWidth = 360;
  const scale = targetWidth / video.videoWidth;
  const targetHeight = Math.max(1, Math.round(video.videoHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });

  if (!context) {
    return {
      capacete: unavailable('Capacete', 'Canvas indisponível.'),
      oculos: unavailable('Óculos de proteção', 'Canvas indisponível.'),
      colete: unavailable('Colete refletivo', 'Canvas indisponível.'),
      luvas: unavailable('Luvas', 'Canvas indisponível.'),
      calcado: unavailable('Calçado fechado', 'Canvas indisponível.'),
    };
  }

  context.drawImage(video, 0, 0, targetWidth, targetHeight);

  const mapPoint = (name: string): Keypoint | null => {
    const value = namedPoint(pose, name);
    return value ? { ...value, x: value.x * scale, y: value.y * scale } : null;
  };

  const bodyHeight = box.height * scale;
  const leftShoulder = mapPoint('left_shoulder'), rightShoulder = mapPoint('right_shoulder');
  const leftHip = mapPoint('left_hip'), rightHip = mapPoint('right_hip');
  const nose = mapPoint('nose'), leftEye = mapPoint('left_eye'), rightEye = mapPoint('right_eye');
  const leftWrist = mapPoint('left_wrist'), rightWrist = mapPoint('right_wrist');
  const leftAnkle = mapPoint('left_ankle'), rightAnkle = mapPoint('right_ankle');

  const headCenter = nose ?? leftEye ?? rightEye;
  let capacete = unavailable('Capacete', 'Cabeça não localizada.');
  if (headCenter) {
    const stats = statsForRegion(context, {
      x: headCenter.x - bodyHeight * 0.11,
      y: headCenter.y - bodyHeight * 0.18,
      width: bodyHeight * 0.22,
      height: bodyHeight * 0.16,
    }, targetWidth, targetHeight);
    const high = ratio(stats.highVis, stats.count), sat = ratio(stats.saturated, stats.count);
    if (high > 0.085) capacete = item('Capacete', 'detectado', Math.min(0.95, 0.62 + high * 2.2), 'Cor de alta visibilidade na região da cabeça.');
    else if (sat > 0.42) capacete = item('Capacete', 'incerto', 0.48, 'Objeto/cor na região da cabeça requer confirmação visual.');
    else capacete = item('Capacete', 'nao_detectado', 0.64, 'Nenhum padrão de capacete de alta visibilidade foi identificado.');
  }

  let colete = unavailable('Colete refletivo', 'Ombros/quadril não localizados.');
  if (leftShoulder && rightShoulder && leftHip && rightHip) {
    const minX = Math.min(leftShoulder.x, rightShoulder.x, leftHip.x, rightHip.x);
    const maxX = Math.max(leftShoulder.x, rightShoulder.x, leftHip.x, rightHip.x);
    const minY = Math.min(leftShoulder.y, rightShoulder.y);
    const maxY = Math.max(leftHip.y, rightHip.y);
    const stats = statsForRegion(context, { x: minX, y: minY, width: Math.max(8, maxX - minX), height: Math.max(8, maxY - minY) }, targetWidth, targetHeight);
    const high = ratio(stats.highVis, stats.count), sat = ratio(stats.saturated, stats.count);
    if (high > 0.095) colete = item('Colete refletivo', 'detectado', Math.min(0.97, 0.65 + high * 1.8), 'Padrão de alta visibilidade detectado no tronco.');
    else if (sat > 0.5) colete = item('Colete refletivo', 'incerto', 0.5, 'Roupa saturada no tronco; confirmar se é EPI.');
    else colete = item('Colete refletivo', 'nao_detectado', 0.7, 'Colete de alta visibilidade não identificado.');
  }

  let oculos = unavailable('Óculos de proteção', 'Olhos não localizados.');
  if (leftEye && rightEye) {
    const minX = Math.min(leftEye.x, rightEye.x), maxX = Math.max(leftEye.x, rightEye.x);
    const eyeDistance = Math.max(8, maxX - minX);
    const stats = statsForRegion(context, {
      x: minX - eyeDistance * 0.45,
      y: Math.min(leftEye.y, rightEye.y) - eyeDistance * 0.45,
      width: eyeDistance * 1.9,
      height: eyeDistance * 0.9,
    }, targetWidth, targetHeight);
    const dark = ratio(stats.dark, stats.count);
    oculos = dark > 0.34
      ? item('Óculos de proteção', 'incerto', 0.54, 'Estrutura escura na região dos olhos; confirmação visual recomendada.')
      : item('Óculos de proteção', 'nao_avaliavel', 0.2, 'Óculos transparentes não podem ser confirmados com segurança por esta heurística.');
  }

  const wristResults = [leftWrist, rightWrist]
    .map((point) => pointRegion(point, bodyHeight, 0.12, 0.12))
    .filter((region): region is Region => region !== null)
    .map((region) => statsForRegion(context, region, targetWidth, targetHeight));

  let luvas = unavailable('Luvas', 'Punhos/mãos não localizados.');
  if (wristResults.length) {
    const skinAverage = wristResults.reduce((sum, stats) => sum + ratio(stats.skin, stats.count), 0) / wristResults.length;
    const saturationAverage = wristResults.reduce((sum, stats) => sum + ratio(stats.saturated, stats.count), 0) / wristResults.length;
    if (skinAverage < 0.14 && saturationAverage > 0.28) luvas = item('Luvas', 'detectado', 0.58, 'Baixa presença de pele e material colorido nas mãos.');
    else if (skinAverage > 0.42) luvas = item('Luvas', 'nao_detectado', 0.6, 'Padrão de pele aparente nas mãos.');
    else luvas = item('Luvas', 'incerto', 0.42, 'Mãos parcialmente visíveis ou iluminação insuficiente.');
  }

  const ankleResults = [leftAnkle, rightAnkle]
    .map((point) => pointRegion(point, bodyHeight, 0.16, 0.14))
    .filter((region): region is Region => region !== null)
    .map((region) => statsForRegion(context, region, targetWidth, targetHeight));

  let calcado = unavailable('Calçado fechado', 'Pés/tornozelos não localizados.');
  if (ankleResults.length) {
    const skinAverage = ankleResults.reduce((sum, stats) => sum + ratio(stats.skin, stats.count), 0) / ankleResults.length;
    const darkAverage = ankleResults.reduce((sum, stats) => sum + ratio(stats.dark, stats.count), 0) / ankleResults.length;
    if (skinAverage < 0.18 && darkAverage > 0.18) calcado = item('Calçado fechado', 'detectado', 0.55, 'Material não semelhante à pele detectado nos pés.');
    else if (skinAverage > 0.34) calcado = item('Calçado fechado', 'nao_detectado', 0.52, 'Região dos pés com pele aparente.');
    else calcado = item('Calçado fechado', 'incerto', 0.4, 'Não é possível diferenciar calçado de segurança de calçado comum.');
  }

  return { capacete, oculos, colete, luvas, calcado };
}
