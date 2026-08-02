"""
Consolidacao -- memorias episodicas antigas viram resumo semantico.

POR QUE ISSO EXISTE
--------------------
Sem consolidacao, memoria so cresce. Depois de meses de conversa, dezenas
de fatos quase-redundantes sobre o mesmo tema competem no contexto por
espaco -- "gosto de correr de manha", "fui correr hoje", "treino toda
terca de manha" sao três linhas quando poderiam ser uma: "corre
regularmente pela manha". O Context Builder ja limita quantos fatos entram
por turno (`max_fatos` em config.py); consolidacao ataca o problema na
raiz, reduzindo o que existe, nao so o que e mostrado.

COMO AGRUPA
-----------
Por similaridade de embedding (cosine), nao por sobreposicao de palavra.
"gosto de correr de manha" e "adoro me exercitar cedo" tem quase nenhuma
palavra em comum mas sao semanticamente o mesmo fato -- Jaccard nunca
agruparia os dois, cosine agrupa. So funciona para memorias que JA TEM
embedding gerado (ver embeddings.py e a coluna `embedding` em store.py);
memoria sem embedding fica de fora da consolidacao ate ganhar um.

QUANDO RODA
-----------
Sob demanda via `consolidar_usuario()`, chamado pelo orquestrador em
segundo plano (mesmo padrao de _disparar_extracao_llm em orchestrator.py
-- thread solta no caminho sincrono, create_task no caminho async). NAO
roda em todo turno: e trabalho pesado (embedding de N memorias, O(n²)
comparacoes de similaridade) que so faz sentido de tempos em tempos, nao
a cada mensagem. O chamador decide a cadencia (ex: a cada 50 turnos, ou
uma vez por dia por usuario).

O QUE NAO FAZ
-------------
Nao apaga a memoria original -- `marcar_consolidadas` so seta
`consolidado_em`, `ativo` continua 1. O resumo e uma camada ADICIONAL. Se
isso se provar errado (contexto poluido com resumo E detalhe ao mesmo
tempo), o ajuste fica em como o Context Builder prioriza um sobre o outro,
nao em apagar a fonte.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

from .embeddings import blob_para_vetor, cosseno
from .store import BancoMemoria, Memoria


def agrupar_por_similaridade(
    memorias: list[Memoria], embeddings_blob: dict[int, bytes],
    limiar: float = 0.82,
) -> list[list[Memoria]]:
    """Agrupa memorias cujo embedding tem cosseno >= limiar entre si.

    Algoritmo guloso simples (nao clustering hierarquico de verdade): pega
    a primeira memoria sem grupo, junta a ela tudo que for similar o
    suficiente, repete. Nao e otimo global, mas e previsivel e barato --
    para volume de dezenas de memorias por rodada de consolidacao, um
    algoritmo mais sofisticado nao muda o resultado que importa.

    `limiar=0.82` mais alto que o 0.75 usado no rerank de busca (ver
    store.py) -- juntar duas memorias em uma so precisa de confianca maior
    que so trazer as duas como resultado relacionado. Errar pra menos
    (nao agrupar o que deveria) e mais seguro que errar pra mais (fundir
    fatos que na verdade eram diferentes).
    """
    vetores: dict[int, list[float]] = {}
    for mem in memorias:
        blob = embeddings_blob.get(mem.id)
        if blob is None:
            continue
        try:
            vetores[mem.id] = blob_para_vetor(blob)
        except Exception:
            continue

    usados: set[int] = set()
    grupos: list[list[Memoria]] = []

    for mem in memorias:
        if mem.id in usados or mem.id not in vetores:
            continue
        grupo = [mem]
        usados.add(mem.id)

        for outra in memorias:
            if outra.id in usados or outra.id not in vetores:
                continue
            sim = cosseno(vetores[mem.id], vetores[outra.id])
            if sim >= limiar:
                grupo.append(outra)
                usados.add(outra.id)

        if len(grupo) >= 2:  # grupo de 1 nao e consolidacao, e so a memoria
            grupos.append(grupo)

    return grupos


def sintetizar(grupo: list[Memoria]) -> str:
    """Produz o texto do resumo a partir de um grupo de memorias similares.

    NAO usa LLM -- de proposito. Consolidacao roda em lote, potencialmente
    dezenas de grupos por vez; uma chamada de LLM por grupo tornaria isso
    caro e lento pra uma tarefa de fundo que ja e best-effort. A sintese
    aqui e mecanica: pega a memoria de MAIOR confianca do grupo como base
    (ela e presumivelmente a mais bem-formada) e anexa quantas evidencias
    existem, para o resumo carregar a informacao "isso foi dito N vezes"
    sem precisar reescrever a frase.

    Se a qualidade do resumo mecanico se provar ruim demais no uso real,
    o proximo passo e uma variante que chama o LLM extrator (o mesmo de
    memory/extractor.py) UMA VEZ para o lote inteiro de grupos, nao um
    por grupo -- mantendo o custo controlado.
    """
    grupo_ordenado = sorted(grupo, key=lambda m: m.confianca, reverse=True)
    base = grupo_ordenado[0].conteudo
    if len(grupo) == 1:
        return base
    return f"{base} (mencionado {len(grupo)} vezes)"


def consolidar_usuario(
    banco: BancoMemoria, *, usuario: str, tipo: str | None = None,
    dias_minimo: float = 14.0, limite_candidatos: int = 50,
    limiar_similaridade: float = 0.82,
) -> list[str]:
    """Roda uma passada de consolidacao para um usuario. Devolve os
    textos dos resumos criados (para log/debug de quem chamou).

    Sincrona de proposito -- mesmo padrao de BancoMemoria em si (sincrono
    por dentro, thread/task por fora e responsabilidade de quem chama).
    Ver orchestrator.py para o padrao de disparo em segundo plano.
    """
    candidatos = banco.candidatos_para_consolidar(
        usuario=usuario, tipo=tipo, dias_minimo=dias_minimo,
        limite=limite_candidatos,
    )
    if len(candidatos) < 2:
        return []

    # candidatos_para_consolidar ja filtra por embedding IS NOT NULL, mas
    # o embedding em si (o BLOB) precisa ser lido -- Memoria nao carrega
    # o vetor por padrao (seria pesado passar em toda leitura comum).
    embeddings_blob = _ler_embeddings(banco, [m.id for m in candidatos if m.id])

    grupos = agrupar_por_similaridade(
        candidatos, embeddings_blob, limiar=limiar_similaridade)

    resumos: list[str] = []
    for grupo in grupos:
        tipo_grupo = grupo[0].tipo  # agrupamento e sempre do mesmo tipo (query já filtra)
        texto = sintetizar(grupo)
        confianca = min(1.0, 0.5 + 0.1 * len(grupo))
        banco.adicionar(
            tipo=tipo_grupo, conteudo=texto, usuario=usuario,
            fonte="consolidacao", confianca=confianca,
            evitar_duplicata=True, gerar_embedding=True,
        )
        banco.marcar_consolidadas([m.id for m in grupo if m.id])
        resumos.append(texto)

    return resumos


def _ler_embeddings(banco: BancoMemoria, ids: list[int]) -> dict[int, bytes]:
    """Le os BLOBs de embedding para um conjunto de ids.

    Funcao a parte (nao metodo de BancoMemoria) porque e uso interno desta
    tarefa especifica -- ler embedding bruto em lote nao e operacao que o
    resto do sistema precisa, entao nao polui a interface publica do
    store com um metodo de proposito unico.
    """
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    with banco._lock:
        linhas = list(banco.con.execute(
            f"SELECT id, embedding FROM memorias WHERE id IN ({placeholders})",
            ids,
        ))
    return {r["id"]: r["embedding"] for r in linhas if r["embedding"] is not None}
