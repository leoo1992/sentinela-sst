from enum import Enum
from typing import Dict, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.ppe_cv import analyze_ppe_cv
from backend.ppe_ensemble import analyze_ppe_ensemble, model_runtime_status

app = FastAPI(
    title="Sentinela SST API",
    version="3.0.0",
    description="Inspeção visual de EPI com MoveNet, OpenCV e ensemble de detectores PPE.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

EPI_KEYS = ["oculos", "capacete", "luvas"]
PPE_LABELS = {
    "oculos": "Óculos",
    "capacete": "Capacete",
    "luvas": "Luvas",
}


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
    visibility: float = Field(default=0, ge=0, le=100)


class EvaluationRequest(BaseModel):
    module: Literal["epi"] = "epi"
    metrics: Optional[PoseMetrics] = None
    ppe: Optional[Dict[str, PpeItem]] = None
    zone_risk: bool = False
    lifting_phase: Optional[str] = None


class PosePoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    score: float = Field(default=0, ge=0, le=1)


class PpeImageRequest(BaseModel):
    module: Literal["epi"] = "epi"
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
    module: Literal["epi"]
    findings: list[Finding]
    summary: str


def _finding(code: str, title: str, detail: str, severity: str) -> Finding:
    return Finding(code=code, title=title, detail=detail, severity=severity)


def _ppe_finding(key: str, value: PpeItem) -> Finding:
    if value.status == DetectionStatus.detectado:
        return _finding(f"epi-{key}-ok", f"{value.label} detectado", value.note or "EPI visualmente identificado.", "ok")
    if value.status == DetectionStatus.nao_detectado:
        return _finding(f"epi-{key}-missing", f"{value.label} não detectado", value.note or "EPI não identificado.", "alert")
    if value.status == DetectionStatus.incerto:
        return _finding(f"epi-{key}-uncertain", f"{value.label} inconclusivo", value.note or "Confirmação visual necessária.", "attention")
    return _finding(f"epi-{key}-unavailable", f"{value.label} não avaliável", value.note or "Informação visual insuficiente.", "info")


def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    findings: list[Finding] = []
    if request.metrics is None:
        findings.append(_finding(
            "pose-missing",
            "Pessoa parcialmente localizada",
            "A pose corporal não foi calculada por completo, mas os EPIs visíveis continuam sendo avaliados.",
            "info",
        ))

    if request.ppe:
        for key in EPI_KEYS:
            value = request.ppe.get(key)
            if value:
                findings.append(_ppe_finding(key, value))
    else:
        findings.append(_finding(
            "ppe-waiting",
            "Preparando inspeção",
            "Mantenha cabeça, olhos e mãos visíveis.",
            "info",
        ))

    alerts = sum(1 for value in findings if value.severity == "alert")
    attention = sum(1 for value in findings if value.severity == "attention")
    summary = (
        f"{alerts} EPI(s) não identificado(s)." if alerts
        else f"{attention} item(ns) inconclusivo(s)." if attention
        else "Inspeção visual concluída."
    )
    return EvaluationResponse(module="epi", findings=findings, summary=summary)


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


def _probability(raw, key_name: str) -> Optional[float]:
    value = _plain_item(raw, key_name)
    if not value or value["status"] == "nao_avaliavel":
        return None
    if value["status"] == "detectado":
        return value["confidence"]
    if value["status"] == "nao_detectado":
        return 1.0 - value["confidence"]
    return 0.5 + (value["confidence"] - 0.5) * 0.20


def _fuse_ppe_item(key_name: str, local_raw, opencv_raw, ensemble_raw) -> dict:
    sources = {
        "local": (_probability(local_raw, key_name), 0.15),
        "opencv": (_probability(opencv_raw, key_name), 0.25),
        "ensemble": (_probability(ensemble_raw, key_name), 0.60),
    }
    used = [(name, probability, weight) for name, (probability, weight) in sources.items() if probability is not None]
    if not used:
        return {
            "label": PPE_LABELS[key_name],
            "status": "nao_avaliavel",
            "confidence": 0.0,
            "note": "Informação visual insuficiente para avaliar este item.",
        }

    total_weight = sum(weight for _, _, weight in used)
    probability = sum(probability * weight for _, probability, weight in used) / max(total_weight, 1e-9)
    evidence_count = len(used)

    if probability >= 0.58:
        status = "detectado"
        note = f"EPI identificado por {evidence_count} análise(s), com probabilidade combinada de {probability:.0%}."
    elif probability <= 0.28 and evidence_count >= 2:
        status = "nao_detectado"
        note = f"As análises convergiram para ausência do EPI ({1 - probability:.0%} de confiança)."
    else:
        status = "incerto"
        note = f"Resultado inconclusivo após combinar {evidence_count} análise(s) ({probability:.0%})."

    return {
        "label": PPE_LABELS[key_name],
        "status": status,
        "confidence": min(1.0, max(0.0, probability)),
        "note": note,
    }


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate_endpoint(request: EvaluationRequest):
    return evaluate(request)


@app.post("/api/analyze-ppe-image")
def analyze_ppe_image(request: PpeImageRequest):
    pose = (
        {name: point.model_dump() for name, point in request.pose_keypoints.items()}
        if request.pose_keypoints else None
    )

    opencv_items: dict = {}
    opencv_error: Optional[str] = None
    person_detected = bool(pose)
    try:
        cv_result = analyze_ppe_cv(request.image_base64, request.module, pose)
        opencv_items = cv_result.get("items", {})
        person_detected = bool(cv_result.get("person_detected")) or person_detected
    except Exception as error:
        print("OpenCV PPE analysis failed", repr(error))
        opencv_error = error.__class__.__name__

    ensemble_items: dict = {}
    ensemble_error: Optional[str] = None
    ensemble_result: dict = {}
    try:
        ensemble_result = analyze_ppe_ensemble(request.image_base64, pose)
        ensemble_items = ensemble_result.get("items", {})
    except Exception as error:
        print("PPE ensemble analysis failed", repr(error))
        ensemble_error = error.__class__.__name__

    local_items = request.local_ppe or {}
    fused = {
        key: _fuse_ppe_item(
            key, local_items.get(key), opencv_items.get(key), ensemble_items.get(key)
        )
        for key in EPI_KEYS
    }

    detected_count = sum(1 for value in fused.values() if value["status"] == "detectado")
    active_models = ensemble_result.get("active_models", []) if ensemble_result else []

    return {
        "items": fused,
        "summary": f"{detected_count}/{len(EPI_KEYS)} EPI(s) identificado(s).",
        "person_detected": person_detected,
        "engines": {
            "tensorflow": bool(local_items),
            "opencv": bool(opencv_items),
            "trained_models": active_models,
            "sliced_inference": bool(ensemble_result.get("sliced_inference")) if ensemble_result else False,
            "weighted_boxes_fusion": bool(ensemble_result.get("wbf")) if ensemble_result else False,
        },
        "diagnostics": {
            "opencv_error": opencv_error,
            "ensemble_error": ensemble_error,
            "model_status": ensemble_result.get("model_status", model_runtime_status()),
        },
    }


@app.get("/api")
def root():
    return {"name": "Sentinela SST API", "status": "ok"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "storage": "disabled",
        "opencv_enabled": True,
        "image_fusion": "movenet+opencv+yolo8n+yolo10n+wbf+sliced-inference",
        "trained_model_status": model_runtime_status(),
        "items": EPI_KEYS,
    }


@app.get("/api/modules")
def modules():
    return [{"id": "epi", "name": "Inspeção de EPI"}]
