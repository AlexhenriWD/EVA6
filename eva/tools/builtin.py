"""
Ferramentas embutidas.

Todas devolvem JSON. As que dependem de rede ou API externa degradam para
{"erro": ...} quando nao configuradas, em vez de falhar -- e a EVA sabe
lidar com isso, foi treinada para nao inventar quando a ferramenta falha.
"""

from __future__ import annotations

import ast
import operator
import os
from datetime import datetime, timedelta

from .registry import registro

# ---------------------------------------------------------------- tempo

DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
        "sexta-feira", "sábado", "domingo"]
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
         "agosto", "setembro", "outubro", "novembro", "dezembro"]


@registro.adicionar(
    "hora_atual",
    "Data e hora atuais. Use quando a pergunta depende de 'hoje', 'agora', "
    "'que dia é', ou de calcular prazo.",
)
def hora_atual() -> dict:
    agora = datetime.now()
    return {
        "iso": agora.isoformat(timespec="seconds"),
        "data": agora.strftime("%d/%m/%Y"),
        "hora": agora.strftime("%H:%M"),
        "dia_semana": DIAS[agora.weekday()],
        "mes": MESES[agora.month - 1],
        "periodo": (
            "madrugada" if agora.hour < 6 else
            "manhã" if agora.hour < 12 else
            "tarde" if agora.hour < 18 else "noite"
        ),
    }


# ---------------------------------------------------------- calculadora

# Avaliacao segura: so as operacoes listadas. eval() puro em texto do
# usuario seria execucao arbitraria de codigo.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _avaliar(no):
    if isinstance(no, ast.Constant):
        if not isinstance(no.value, (int, float)):
            raise ValueError("só números")
        return no.value
    if isinstance(no, ast.BinOp):
        op = _OPS.get(type(no.op))
        if not op:
            raise ValueError("operação não permitida")
        return op(_avaliar(no.left), _avaliar(no.right))
    if isinstance(no, ast.UnaryOp):
        op = _OPS.get(type(no.op))
        if not op:
            raise ValueError("operação não permitida")
        return op(_avaliar(no.operand))
    raise ValueError("expressão não permitida")


@registro.adicionar(
    "calcular",
    "Calcula uma expressão aritmética. Use para qualquer conta -- modelos de "
    "linguagem erram aritmética com frequência.",
    {"expressao": "expressão como '2 + 3 * 4'"},
)
def calcular(expressao: str) -> dict:
    limpa = expressao.replace("×", "*").replace("÷", "/").replace(",", ".").replace("^", "**")
    try:
        arvore = ast.parse(limpa, mode="eval")
        valor = _avaliar(arvore.body)
    except ZeroDivisionError:
        return {"erro": "divisao_por_zero", "expressao": expressao}
    except Exception as e:
        return {"erro": "expressao_invalida", "detalhe": str(e)[:100], "expressao": expressao}

    if isinstance(valor, float) and valor.is_integer():
        valor = int(valor)
    elif isinstance(valor, float):
        valor = round(valor, 10)
    return {"expressao": expressao, "resultado": valor}


# ---------------------------------------------------------------- clima


@registro.adicionar(
    "clima",
    "Previsão do tempo para uma cidade.",
    {"cidade": "nome da cidade"},
    cara=True,
)
def clima(cidade: str) -> dict:
    """Usa Open-Meteo (sem chave). Geocodifica e busca a previsao."""
    try:
        import urllib.parse
        import urllib.request
        import json as _json

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?name="
            + urllib.parse.quote(cidade) + "&count=1&language=pt&format=json"
        )
        with urllib.request.urlopen(geo_url, timeout=8) as r:
            geo = _json.loads(r.read())
        if not geo.get("results"):
            return {"erro": "cidade_nao_encontrada", "cidade": cidade}

        loc = geo["results"][0]
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={loc['latitude']}"
            f"&longitude={loc['longitude']}&current=temperature_2m,relative_humidity_2m,"
            "precipitation,weather_code&timezone=auto"
        )
        with urllib.request.urlopen(url, timeout=8) as r:
            dados = _json.loads(r.read())

        atual = dados.get("current", {})
        return {
            "cidade": loc.get("name"),
            "temperatura": atual.get("temperature_2m"),
            "umidade": atual.get("relative_humidity_2m"),
            "precipitacao": atual.get("precipitation"),
            "codigo_tempo": atual.get("weather_code"),
        }
    except Exception as e:
        return {"erro": "falha_rede", "detalhe": str(e)[:120], "cidade": cidade}


# ----------------------------------------------------------------- busca


@registro.adicionar(
    "buscar",
    "Busca informação atual na web. Use para fatos que mudam (notícias, "
    "preços, dados recentes) ou que você não sabe.",
    {"consulta": "o que buscar"},
    cara=True,
)
def buscar(consulta: str) -> dict:
    """Busca via SearXNG local (metabusca open-source, self-hosted).

    SUBSTITUIU o DuckDuckGo Instant Answer, que tinha dois problemas
    reais em uso: cobertura estreita demais (só resolvia entidade/definição
    curta -- "Python" sozinho funcionava, "Python linguagem de programação"
    já não trazia nada, e frase/pergunta natural quase nunca trazia), e a
    API oficial (api.duckduckgo.com) passou a devolver 403 em alguns
    ambientes de rede, sem relação com chave ou rate limit.

    SearXNG resolve os dois: agrega vários motores (DuckDuckGo, Wikipedia,
    Brave, Bing -- configurados em searxng/settings.yml) numa consulta só,
    então a cobertura de qualquer um deles individualmente falhando não
    derruba a busca inteira. E por rodar local (docker-compose.yml na raiz
    do projeto), não depende de política de bloqueio de terceiro.

    Requer o container rodando: docker compose up -d (a partir da raiz do
    projeto). Se a instância não responder, o erro diz exatamente isso, em
    vez de um traceback de conexão recusada sem contexto.
    """
    base_url = os.environ.get("EVA_SEARXNG_URL", "http://127.0.0.1:8080")
    consulta = (consulta or "").strip()
    print(f"[busca] consulta recebida: {consulta[:120]!r}")
    if not consulta or len(consulta) < 2:
        return {"erro": "consulta_vazia", "consulta": consulta}

    try:
        import urllib.error
        import urllib.parse
        import urllib.request
        import json as _json

        url = (base_url.rstrip("/") + "/search?q=" + urllib.parse.quote(consulta)
               + "&format=json&language=pt-BR")
        req = urllib.request.Request(url, headers={"User-Agent": "EVA/1.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            dados = _json.loads(r.read())

        resultados = dados.get("results") or []
        if not resultados:
            return {"consulta": consulta, "resultados": [], "aviso": "sem_resultado"}

        # SearXNG devolve muitos campos por resultado (engine, score,
        # template, etc.) que só interessam para debug -- o modelo
        # conversacional só precisa do essencial, e passar tudo infla o
        # contexto sem ganho (o Context Builder já filtra chave começando
        # com "_", mas aqui é melhor nem gerar o excesso).
        primeiro = resultados[0]
        resumo = primeiro.get("content") or primeiro.get("title") or None
        print(f"[busca] resultado: resumo={(resumo or '')[:150]!r}")
        relacionados = [
            r.get("title", "") for r in resultados[1:4] if r.get("title")
        ]

        return {
            "consulta": consulta,
            "resumo": resumo,
            "fonte": primeiro.get("url") or None,
            "relacionados": relacionados,
        }
    except urllib.error.URLError as e:
        return {
            "erro": "searxng_indisponivel",
            "detalhe": f"não consegui falar com {base_url}. "
                       f"O container está rodando? docker compose up -d",
            "consulta": consulta,
        }
    except Exception as e:
        return {"erro": "falha_busca", "detalhe": str(e)[:120], "consulta": consulta}


def carregar_ferramentas():
    """Garante que as ferramentas foram registradas. Retorna o registro."""
    try:
        from . import minecraft_tools  # noqa: F401 -- import registra via decorador, não usa o nome
    except Exception as e:
        # Minecraft é opcional -- sem o cliente/bridge disponível, as
        # ferramentas de minecraft_* simplesmente não existem, e todo o
        # resto do projeto continua funcionando normal. Mesma filosofia de
        # degradação graciosa do resto deste arquivo (clima, buscar).
        import sys
        print(f"[ferramentas] minecraft_tools não carregou ({e}) -- "
              f"ferramentas de Minecraft indisponíveis nesta sessão", file=sys.stderr)

    # EVA_ROBOT_ATIVO=0 pula o import por completo -- ANTES do decorador
    # rodar, então nenhuma ferramenta robo_* é registrada. Sem isso, ligar
    # o robô fisicamente era o único jeito de desligar o CATÁLOGO: mesmo
    # com o robô desconectado, o import de robot_tools já registrava
    # robo_estado/robo_olhar/robo_gesto/etc, e registro.descrever() (ver
    # decision.py::DecisorPorLLM.decidir) manda esse catálogo pro prompt
    # do decisor em TODA mensagem -- inclusive quando a call é só
    # conversa e ninguém pediu nada de robô. Se o decisor (LLM local ou
    # Groq) interpretar mal e escolher uma robo_*, a chamada real dispara
    # _iniciar_thread_robo() na hora (ver robot_tools.py), que tenta
    # conectar em EVA_ROBOT_HOST e fica reconectando pra sempre em
    # segundo plano -- e sozinho não indica erro no dashboard nem na
    # conversa, só no console (achado real: log cheio de "[robo] sem
    # conexão" numa sessão onde só se queria falar por texto/voz, sem
    # tocar em robô nenhum). Default "1" preserva o comportamento de
    # antes -- quem já usa o robô não precisa mudar nada.
    if os.environ.get("EVA_ROBOT_ATIVO", "1") == "1":
        try:
            from . import robot_tools  # noqa: F401 -- mesmo padrão de minecraft_tools acima
        except Exception as e:
            # Robô é opcional pelo mesmo motivo: sem eva_command_server.py
            # acessível, as ferramentas robo_* simplesmente não existem, e o
            # resto do projeto continua normal.
            import sys
            print(f"[ferramentas] robot_tools não carregou ({e}) -- "
                  f"ferramentas de robô indisponíveis nesta sessão", file=sys.stderr)
    else:
        print("[ferramentas] robot_tools não carregado (EVA_ROBOT_ATIVO=0) -- "
              "ferramentas robo_* fora do catálogo do decisor nesta sessão")

    return registro