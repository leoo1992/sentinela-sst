from enum import Enum
from typing import Dict, Literal, Optional

from fastapi import FastAPI
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
        return EvaluationResponse(
            module=request.module,
            findings=[make_finding("pose-missing", "Pessoa não localizada", "Posicione o corpo dentro do enquadramento para iniciar a análise.", "info")],
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

    if request.module == "ergonomia":
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

    if request.module == "cargas":
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

@app.get("/api")
def root():
    return {"name": "Sentinela SST API", "status": "ok", "storage": "disabled", "video_received": False}

@app.get("/api/health")
def health():
    return {"status": "ok", "storage": "disabled", "processing": "derived-metrics-only"}

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
