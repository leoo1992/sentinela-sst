from api.index import EvaluationRequest, PoseMetrics, PpeItem, evaluate

def test_ergonomia_alerta_inclinacao_tronco():
    result = evaluate(EvaluationRequest(module="ergonomia", metrics=PoseMetrics(trunk_inclination=52, neck_inclination=10, visibility=95)))
    assert any(item.code == "ergo-trunk-high" for item in result.findings)
    assert any(item.severity == "alert" for item in result.findings)

def test_altura_alerta_zona_risco():
    result = evaluate(EvaluationRequest(module="altura", metrics=PoseMetrics(trunk_inclination=8, visibility=90), zone_risk=True))
    assert result.findings[0].code == "height-zone"
    assert result.findings[0].severity == "alert"

def test_sem_pose_retorna_estado_aguardando():
    result = evaluate(EvaluationRequest(module="cargas", metrics=None))
    assert result.findings[0].code == "pose-missing"


def test_epi_can_be_evaluated_without_pose():
    payload = EvaluationRequest(
        module="epi",
        metrics=None,
        ppe={
            "capacete": PpeItem(label="Capacete", status="detectado", confidence=0.95),
            "oculos": PpeItem(label="Óculos de proteção", status="detectado", confidence=0.9),
            "colete": PpeItem(label="Colete refletivo", status="detectado", confidence=0.9),
            "luvas": PpeItem(label="Luvas", status="detectado", confidence=0.85),
            "calcado": PpeItem(label="Calçado fechado", status="nao_avaliavel", confidence=0.1),
        },
        zone_risk=False,
        lifting_phase=None,
    )
    result = evaluate(payload)
    assert any(item.code == "epi-capacete-ok" for item in result.findings)
    assert any(item.code == "epi-oculos-ok" for item in result.findings)
