import json
import os
import urllib.error
import urllib.request
from enum import Enum
from typing import Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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

class GeminiImageRequest(BaseModel):
    module: Literal["epi", "altura"]
    image_base64: str = Field(min_length=100, max_length=6_000_000)
    mime_type: Literal["image/jpeg", "image/png", "image/webp"] = "image/jpeg"

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
    "colete": "Colete refletivo",
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
        for content in step.get("content", []) or []:
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts).strip()

@app.post("/api/analyze-ppe-image")
def analyze_ppe_image(request: GeminiImageRequest):
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="Análise complementar de imagem não configurada.")

    keys = (
        ["capacete", "cinturao", "talabarte", "travaQuedas"]
        if request.module == "altura"
        else ["capacete", "oculos", "colete", "luvas", "calcado"]
    )

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
Você é uma segunda camada de inspeção visual de EPI para Segurança do Trabalho.
Analise SOMENTE o que está realmente visível na foto. Não invente EPI oculto.

Regras obrigatórias:
- detectado: o item está visualmente identificável.
- nao_detectado: use apenas quando a região necessária está claramente visível e o item está claramente ausente.
- incerto: há indício visual, mas não é possível confirmar.
- nao_avaliavel: item/região está cortado, oculto, distante ou sem definição suficiente.
- Capacete: capacete de segurança industrial visível.
- Óculos: óculos de proteção visíveis; não confunda óculos comuns quando não houver evidência.
- Colete: vestimenta de alta visibilidade/refletiva visível.
- Luvas: luvas de proteção nas mãos.
- Calçado: só avalie quando os pés estiverem visíveis.
- Cinturão paraquedista: exige tiras/fitas de arnês próprias do sistema antiqueda no tronco/quadril/pernas. Colete refletivo, suspensório de jardineira, uniforme ou faixa refletiva NÃO são cinturão.
- Talabarte: exige cabo/fita de conexão e/ou conectores visíveis ligados ao cinturão/sistema. Não deduza apenas pela roupa.
- Trava-quedas: exige o dispositivo de trava-quedas e sua conexão à linha de vida/cabo/corda/trilho visíveis. Não deduza pela presença de cinturão.
Retorne notas curtas em português.
""".strip()

    body = {
        "model": "gemini-3.8-flash",
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
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print("Gemini HTTP error", error.code, detail[:500])
        raise HTTPException(status_code=502, detail="Falha na análise complementar da imagem.") from error
    except Exception as error:
        print("Gemini request failed", repr(error))
        raise HTTPException(status_code=502, detail="Falha na comunicação com a análise complementar.") from error

    output_text = _gemini_output_text(data)
    try:
        parsed = json.loads(output_text)
    except (TypeError, json.JSONDecodeError) as error:
        print("Gemini invalid structured output", output_text[:500])
        raise HTTPException(status_code=502, detail="Resposta inválida da análise complementar.") from error

    normalized: dict[str, dict] = {}
    raw_items = parsed.get("items", {}) if isinstance(parsed, dict) else {}

    for key_name in keys:
        raw = raw_items.get(key_name, {}) if isinstance(raw_items, dict) else {}
        status = raw.get("status", "nao_avaliavel")
        if status not in {"detectado", "nao_detectado", "incerto", "nao_avaliavel"}:
            status = "nao_avaliavel"
        try:
            confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
        except (TypeError, ValueError):
            confidence = 0.0
        note = raw.get("note")
        normalized[key_name] = {
            "label": PPE_LABELS[key_name],
            "status": status,
            "confidence": confidence,
            "note": note if isinstance(note, str) and note.strip() else "Sem observação adicional.",
        }

    return {
        "items": normalized,
        "summary": parsed.get("summary", "") if isinstance(parsed, dict) else "",
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
