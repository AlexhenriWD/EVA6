from types import SimpleNamespace

from context import ContextBuilder, MODO_INICIATIVA, PREFIXO_IDEIA


def test_montar_inclui_bloco_de_iniciativa_no_cabecalho():
    cfg = SimpleNamespace(
        memoria=SimpleNamespace(
            max_fatos=5,
            max_episodios=5,
            max_procedimentos=5,
            max_personalidade=5,
            janela_historico=5,
        ),
        llm=SimpleNamespace(formato_contexto="json"),
    )
    builder = ContextBuilder(cfg)

    contexto = builder.montar(
        plano=SimpleNamespace(intencao=""),
        memorias={},
        resultados_ferramentas={},
        estado=None,
        historico=[],
        contexto_visual="um cenário",
        iniciativa="puxar assunto",
    )

    assert MODO_INICIATIVA in contexto.system
    assert PREFIXO_IDEIA + "puxar assunto" in contexto.system
