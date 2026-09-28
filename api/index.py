import json
import os
import urllib.error
import urllib.request
from enum import Enum
from typing import Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.ppe_cv import analyze_ppe_cv

app = FastAPI(
    title="Sentinela SST API",
    version="1.0.0",
    description="Regras em tempo real para métricas derivadas de visão computacional.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

class DetectionStatus(str, Enum):
    detectado = "detectado"
    nao_detectado = "nao_detectado"
    incerto = "incerto"
    nao_avaliavel = "nao_avaliavel"

class PpeItem(BaseModel):
    label: str
    status: DetectionStatus
    confidence: float = Field(ge=0, le=1)
    note: Optional[str] = None

class PoseMetrics(BaseModel):
    trunk_inclination: Optional[float] = None
    neck_inclination: Optional[float] = None
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    left_hip_angle: Optional[float] = None
    right_hip_angle: Optional[float] = None
    shoulder_tilt: Optional[float] = None
    hip_tilt: Optional[float] = None
    knee_asymmetry: Optional[float] = None
    hip_asymmetry: Optional[float] = None
    visibility: float = Field(default=0, ge=0, le=100)

class EvaluationRequest(BaseModel):
    module: Literal["epi", "altura", "ergonomia", "cargas"]
    metrics: Optional[PoseMetrics] = None
    ppe: Optional[Dict[str, PpeItem]] = None
    zone_risk: bool = False
    lifting_phase: Optional[str] = None

class PosePoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    score: float = Field(default=0, ge=0, le=1)

class GeminiImageRequest(BaseModel):
    module: Literal["epi", "altura"]
    image_base64: str = Field(min_length=100, max_length=6_000_000)
    mime_type: Literal["image/jpeg", "image/png", "image/webp"] = "image/jpeg"
    local_ppe: Optional[Dict[str, PpeItem]] = None
    pose_keypoints: Optional[Dict[str, PosePoint]] = None

class Finding(BaseModel):
    code: str
    title: str
    detail: str
    severity: Literal["ok", "info", "attention", "alert"]

class EvaluationResponse(BaseModel):
    module: Literal["epi", "altura", "ergonomia", "cargas"]
    findings: list[Finding]
    summary: str

def make_finding(code: str, title: str, detail: str, severity: str) -> Finding:
    return Finding(code=code, title=title, detail=detail, severity=severity)

def ppe_finding(key: str, value: PpeItem) -> Finding:
    if value.status == DetectionStatus.detectado:
        return make_finding(f"epi-{key}-ok", f"{value.label} detectado", value.note or "EPI visualmente identificado.", "ok")
    if value.status == DetectionStatus.nao_detectado:
        return make_finding(f"epi-{key}-missing", f"{value.label} não detectado", value.note or "EPI não foi identificado na imagem.", "alert")
    if value.status == DetectionStatus.incerto:
        return make_finding(f"epi-{key}-uncertain", f"{value.label} inconclusivo", value.note or "É necessária confirmação visual.", "attention")
    return make_finding(f"epi-{key}-unavailable", f"{value.label} não avaliável", value.note or "A câmera não fornece informação suficiente.", "info")

def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    findings: list[Finding] = []
    metrics = request.metrics

    if metrics is None:
        findings.append(make_finding(
            "pose-missing",
            "Postura não localizada",
            "A postura corporal não pôde ser calculada, mas os EPIs visíveis ainda podem ser avaliados."
            if request.module in ("epi", "altura")
            else "Posicione o corpo dentro do enquadramento para iniciar a análise.",
            "info",
        ))
        if request.module not in ("epi", "altura"):
            return EvaluationResponse(
                module=request.module,
                findings=findings,
                summary="Aguardando uma pessoa no enquadramento.",
            )

    if request.module == "epi":
        if request.ppe:
            epi_keys = ["capacete", "oculos", "colete", "luvas", "calcado"]
            findings.extend(
                ppe_finding(key, request.ppe[key])
                for key in epi_keys
                if request.ppe.get(key)
            )
        else:
            findings.append(make_finding("ppe-waiting", "Preparando inspeção", "Mantenha cabeça, tronco, mãos e pés visíveis por alguns instantes.", "info"))

    if request.module == "altura":
        findings.append(make_finding(
            "height-zone" if request.zone_risk else "height-zone-ok",
            "Pessoa dentro da zona de risco" if request.zone_risk else "Fora da zona de risco",
            "A projeção do corpo alcançou a área virtual de borda configurada." if request.zone_risk else "A pessoa está fora da área virtual de borda configurada.",
            "alert" if request.zone_risk else "ok",
        ))
        if request.ppe:
            height_keys = ["capacete", "cinturao", "talabarte", "travaQuedas"]
            findings.extend(
                ppe_finding(key, request.ppe[key])
                for key in height_keys
                if request.ppe.get(key)
            )
        else:
            findings.append(make_finding(
                "height-ppe-waiting",
                "Proteção contra quedas",
                "Mantenha cabeça, tronco, cintura e sistema de conexão visíveis para avaliação.",
                "info",
            ))

    if request.module == "ergonomia" and metrics is not None:
        trunk = metrics.trunk_inclination
        if trunk is not None:
            if trunk >= 45:
                findings.append(make_finding("ergo-trunk-high", "Inclinação elevada do tronco", f"Tronco estimado em {round(trunk)}° em relação à vertical.", "alert"))
            elif trunk >= 22:
                findings.append(make_finding("ergo-trunk-medium", "Inclinação moderada do tronco", f"Tronco estimado em {round(trunk)}°.", "attention"))
            else:
                findings.append(make_finding("ergo-trunk-ok", "Tronco próximo da vertical", f"Inclinação estimada em {round(trunk)}°.", "ok"))
        neck = metrics.neck_inclination
        if neck is not None and neck >= 25:
            findings.append(make_finding("ergo-neck", "Inclinação do pescoço", f"Inclinação estimada em {round(neck)}°.", "alert" if neck >= 40 else "attention"))
        if metrics.knee_asymmetry is not None and metrics.knee_asymmetry >= 16:
            findings.append(make_finding("ergo-knee-asymmetry", "Assimetria entre joelhos", f"Diferença angular estimada de {round(metrics.knee_asymmetry)}°.", "attention"))

    if request.module == "cargas" and metrics is not None:
        if request.lifting_phase:
            findings.append(make_finding("lift-phase", request.lifting_phase, "Fase estimada a partir dos ângulos de quadril, joelho e tronco.", "info"))
        trunk = metrics.trunk_inclination
        if trunk is not None and trunk >= 42:
            findings.append(make_finding("lift-trunk", "Flexão elevada do tronco", f"Inclinação estimada em {round(trunk)}° durante o movimento.", "alert"))
        elif trunk is not None:
            findings.append(make_finding("lift-trunk-ok", "Controle do tronco", f"Inclinação estimada em {round(trunk)}°.", "attention" if trunk >= 25 else "ok"))
        knees = [value for value in [metrics.left_knee_angle, metrics.right_knee_angle] if value is not None]
        if knees:
            average = round(sum(knees) / len(knees))
            if average > 158 and (trunk or 0) > 28:
                findings.append(make_finding("lift-knees", "Pouca flexão dos joelhos", f"Ângulo médio dos joelhos em {average}° com inclinação do tronco.", "attention"))
            else:
                findings.append(make_finding("lift-knees-ok", "Participação dos joelhos", f"Ângulo médio estimado em {average}°.", "ok"))
        if metrics.hip_asymmetry is not None and metrics.hip_asymmetry >= 14:
            findings.append(make_finding("lift-asymmetry", "Assimetria durante o levantamento", f"Diferença angular de quadril estimada em {round(metrics.hip_asymmetry)}°.", "attention"))

    if not findings:
        findings.append(make_finding("analysis-normal", "Análise ativa", "Nenhum indicador adicional foi gerado neste instante.", "ok"))

    alert_count = sum(1 for entry in findings if entry.severity == "alert")
    attention_count = sum(1 for entry in findings if entry.severity == "attention")
    summary = f"{alert_count} alerta(s) visual(is) no quadro atual." if alert_count else (f"{attention_count} ponto(s) para atenção." if attention_count else "Nenhum alerta visual crítico no quadro atual.")
    return EvaluationResponse(module=request.module, findings=findings, summary=summary)

PPE_LABELS = {
    "capacete": "Capacete",
    "oculos": "Óculos de proteção",
    "colete": "Colete/vestimenta refletiva",
    "luvas": "Luvas",
    "calcado": "Calçado fechado",
    "cinturao": "Cinturão paraquedista",
    "talabarte": "Talabarte",
    "travaQuedas": "Trava-quedas",
}

def _gemini_output_text(data: dict) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    parts: list[str] = []
    for step in data.get("steps", []) or []:
        if step.get("type") != "model_output":
            continue
        for content in step.get("content", []) or []:
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts).strip()

def _plain_item(raw, key_name: str) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, PpeItem):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return None

    status = raw.get("status", "nao_avaliavel")
    if isinstance(status, DetectionStatus):
        status = status.value
    if status not in {"detectado", "nao_detectado", "incerto", "nao_avaliavel"}:
        status = "nao_avaliavel"

    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0

    note = raw.get("note")
    return {
        "label": str(raw.get("label") or PPE_LABELS[key_name]),
        "status": status,
        "confidence": confidence,
        "note": note if isinstance(note, str) and note.strip() else None,
    }

def _fuse_ppe_item(key_name: str, local_raw, opencv_raw, gemini_raw) -> dict:
    sources = {
        "TensorFlow": _plain_item(local_raw, key_name),
        "OpenCV": _plain_item(opencv_raw, key_name),
        "Gemini": _plain_item(gemini_raw, key_name),
    }
    available = [(name, value) for name, value in sources.items() if value]
    detected = [(name, value) for name, value in available if value["status"] == "detectado"]
    missing = [(name, value) for name, value in available if value["status"] == "nao_detectado"]
    uncertain = [(name, value) for name, value in available if value["status"] == "incerto"]

    strong_detected = []
    for name, value in detected:
        threshold = 0.55 if name == "Gemini" else 0.68
        if name == "OpenCV":
            threshold = 0.72 if key_name in {"capacete", "colete"} else 0.82
        if value["confidence"] >= threshold:
            strong_detected.append((name, value))

    if strong_detected:
        best_name, best = max(strong_detected, key=lambda pair: pair[1]["confidence"])
        confirmations = [name for name, value in detected if value["confidence"] >= 0.45]
        source_text = ", ".join(confirmations) if confirmations else best_name
        return {
            "label": PPE_LABELS[key_name],
            "status": "detectado",
            "confidence": max(value["confidence"] for _, value in strong_detected),
            "note": f"Identificado pela análise combinada ({source_text}). " + (best.get("note") or ""),
        }

    if len(detected) >= 2:
        confidence = sum(value["confidence"] for _, value in detected) / len(detected)
        return {
            "label": PPE_LABELS[key_name],
            "status": "detectado",
            "confidence": max(0.58, min(0.88, confidence)),
            "note": "Duas fontes independentes encontraram sinais compatíveis com este EPI.",
        }

    if detected:
        name, best = max(detected, key=lambda pair: pair[1]["confidence"])
        return {
            "label": PPE_LABELS[key_name],
            "status": "incerto",
            "confidence": best["confidence"],
            "note": f"{name} encontrou indício do EPI, mas faltou confirmação de outra análise.",
        }

    gemini_missing = next((value for name, value in missing if name == "Gemini" and value["confidence"] >= 0.82), None)
    if gemini_missing and not uncertain:
        return {
            "label": PPE_LABELS[key_name],
            "status": "nao_detectado",
            "confidence": gemini_missing["confidence"],
            "note": gemini_missing.get("note") or "O EPI não foi identificado na região visível.",
        }

    if len(missing) >= 2:
        confidence = sum(value["confidence"] for _, value in missing) / len(missing)
        return {
            "label": PPE_LABELS[key_name],
            "status": "nao_detectado",
            "confidence": min(0.86, confidence),
            "note": "Mais de uma análise não identificou este EPI na região visível.",
        }

    if uncertain:
        name, best = max(uncertain, key=lambda pair: pair[1]["confidence"])
        return {
            "label": PPE_LABELS[key_name],
            "status": "incerto",
            "confidence": best["confidence"],
            "note": best.get("note") or f"{name} encontrou informação visual inconclusiva.",
        }

    return {
        "label": PPE_LABELS[key_name],
        "status": "nao_avaliavel",
        "confidence": 0.1,
        "note": "Não houve informação visual suficiente para avaliar este item.",
    }

def _call_gemini(request: GeminiImageRequest, keys: list[str]) -> tuple[Optional[dict], Optional[str]]:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        return None, "GOOGLE_API_KEY não configurada."

    item_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["detectado", "nao_detectado", "incerto", "nao_avaliavel"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "note": {"type": "string"},
        },
        "required": ["status", "confidence", "note"],
    }
    response_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "object",
                "properties": {key_name: item_schema for key_name in keys},
                "required": keys,
            },
            "summary": {"type": "string"},
        },
        "required": ["items", "summary"],
    }

    prompt = """
Você analisa uma FOTO enviada para inspeção visual de EPI em Segurança do Trabalho.
Ignore marcas d'água, textos, logos e elementos de interface. Avalie a pessoa real da foto.
Analise SOMENTE o que está visualmente presente. Não invente EPI oculto.

Classificação:
- detectado: o EPI está claramente identificável.
- nao_detectado: a região está claramente visível e o EPI claramente não está presente.
- incerto: há sinal compatível, mas falta detalhe para confirmar.
- nao_avaliavel: a região está cortada, oculta, distante ou sem definição.

Regras:
- Capacete: capacete de segurança industrial na cabeça.
- Óculos: óculos de proteção/goggles cobrindo ou protegendo os olhos.
- Colete/vestimenta refletiva: aceite colete OU uniforme/jaqueta de alta visibilidade com faixas refletivas.
- Luvas: luvas de proteção cobrindo as mãos.
- Calçado: só avalie quando os pés estiverem visíveis.
- Cinturão paraquedista: exige tiras próprias de arnês no tronco/quadril/pernas; não confunda colete, suspensório ou uniforme.
- Talabarte: exige fita/cabo e conectores visíveis ligados ao sistema antiqueda.
- Trava-quedas: exige o dispositivo e sua conexão à linha de vida/cabo/corda/trilho visíveis.
Use confiança alta apenas quando o item estiver realmente visível.
Retorne notas curtas em português.
""".strip()

    body = {
        "model": "gemini-3.8-flash",
        "store": False,
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": request.image_base64,
                "mime_type": request.mime_type,
            },
        ],
        "generation_config": {"thinking_level": "low"},
        "response_format": {
            "type": "text",
            "mime_type": "application/json",
            "schema": response_schema,
        },
    }

    gemini_request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
            "Api-Revision": "2026-05-20",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(gemini_request, timeout=35) as response:
            data = json.loads(response.read().decode("utf-8"))
        output_text = _gemini_output_text(data)
        parsed = json.loads(output_text)
        raw_items = parsed.get("items", {}) if isinstance(parsed, dict) else {}
        return raw_items if isinstance(raw_items, dict) else {}, None
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print("Gemini HTTP error", error.code, detail[:600])
        return None, f"Gemini HTTP {error.code}"
    except Exception as error:
        print("Gemini PPE analysis failed", repr(error))
        return None, error.__class__.__name__

@app.post("/api/analyze-ppe-image")
def analyze_ppe_image(request: GeminiImageRequest):
    keys = (
        ["capacete", "cinturao", "talabarte", "travaQuedas"]
        if request.module == "altura"
        else ["capacete", "oculos", "colete", "luvas", "calcado"]
    )

    pose = (
        {name: point.model_dump() for name, point in request.pose_keypoints.items()}
        if request.pose_keypoints
        else None
    )

    opencv_items: dict = {}
    person_detected = bool(pose)
    opencv_error: Optional[str] = None
    try:
        cv_result = analyze_ppe_cv(request.image_base64, request.module, pose)
        opencv_items = cv_result.get("items", {})
        person_detected = bool(cv_result.get("person_detected")) or person_detected
    except Exception as error:
        print("OpenCV PPE analysis failed", repr(error))
        opencv_error = error.__class__.__name__

    gemini_items, gemini_error = _call_gemini(request, keys)
    local_items = request.local_ppe or {}

    fused = {
        key_name: _fuse_ppe_item(
            key_name,
            local_items.get(key_name),
            opencv_items.get(key_name),
            gemini_items.get(key_name) if gemini_items else None,
        )
        for key_name in keys
    }

    detected_count = sum(1 for value in fused.values() if value["status"] == "detectado")
    return {
        "items": fused,
        "summary": f"{detected_count}/{len(keys)} proteção(ões) visualmente identificada(s).",
        "person_detected": person_detected,
        "engines": {
            "tensorflow": bool(local_items),
            "opencv": bool(opencv_items),
            "gemini": bool(gemini_items),
        },
        "diagnostics": {
            "opencv_error": opencv_error,
            "gemini_error": gemini_error,
        },
    }

@app.get("/api")
def root():
    return {"name": "Sentinela SST API", "status": "ok", "storage": "disabled", "video_received": False}

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "storage": "disabled",
        "processing": "derived-metrics-only",
        "gemini_configured": bool(os.environ.get("GOOGLE_API_KEY", "").strip()),
        "opencv_enabled": True,
        "image_fusion": "tensorflow+opencv+gemini",
    }

@app.get("/api/modules")
def modules():
    return [
        {"id": "epi", "name": "Inspeção de EPI"},
        {"id": "altura", "name": "Segurança em Altura"},
        {"id": "ergonomia", "name": "Análise Ergonômica"},
        {"id": "cargas", "name": "Levantamento de Cargas"},
    ]

@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate_route(request: EvaluationRequest):
    return evaluate(request)
