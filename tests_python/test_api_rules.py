from api.index import EvaluationRequest, PpeItem, evaluate


def test_epi_avalia_apenas_tres_itens():
    payload = EvaluationRequest(
        module="epi",
        metrics=None,
        ppe={
            "capacete": PpeItem(label="Capacete", status="detectado", confidence=0.95),
            "oculos": PpeItem(label="Óculos", status="detectado", confidence=0.90),
            "luvas": PpeItem(label="Luvas", status="detectado", confidence=0.85),
        },
    )

    result = evaluate(payload)

    assert any(item.code == "epi-capacete-ok" for item in result.findings)
    assert any(item.code == "epi-oculos-ok" for item in result.findings)
    assert any(item.code == "epi-luvas-ok" for item in result.findings)
