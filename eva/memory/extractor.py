"""
Extracao de memorias a partir da conversa.

Duas estrategias, e a escolha entre elas importa:

REGRAS (padrao): padroes explicitos como "eu uso X", "meu nome e X",
"lembra que X". Sao precisas -- so disparam quando a pessoa de fato
declarou algo -- e nao custam nada. A limitacao e cobertura: nao pegam
informacao dita de forma indireta.

LLM (opcional): manda a conversa para o modelo e pede fatos em JSON.
Cobre muito mais, mas inventa. Um fato alucinado guardado na memoria e
pior que fato nenhum: ele volta como contexto em conversas futuras e a
EVA passa a agir com base em algo que a pessoa nunca disse.

Por isso o padrao e regras, e a extracao por LLM entra com confianca menor
e marcada na fonte, para ser auditavel depois.
"""

from __future__ import annotations

import re

# (padrao, tipo, template). O grupo 1 vira o conteudo.
REGRAS: list[tuple[re.Pattern, str, str]] = [
    # declaracoes explicitas de preferencia e habito
    (re.compile(r"\b(?:eu )?uso (?:o |a )?([\w\s\.\-]{3,40})", re.I), "semantica", "usa {0}"),
    (re.compile(r"\b(?:eu )?(?:sou|s[oô]) (?:um |uma )?([\w\s]{3,40})", re.I), "semantica", "é {0}"),
    (re.compile(r"\b(?:eu )?trabalho (?:com|como|na|no) ([\w\s]{3,40})", re.I), "semantica", "trabalha com {0}"),
    (re.compile(r"\b(?:eu )?moro (?:em|no|na) ([\w\s]{3,40})", re.I), "semantica", "mora em {0}"),
    (re.compile(r"\bmeu nome (?:é|eh|e) ([\w\s]{2,30})", re.I), "semantica", "se chama {0}"),
    (re.compile(r"\b(?:eu )?tenho (?:um|uma) ([\w\s]{3,40})", re.I), "semantica", "tem {0}"),
    (re.compile(r"\b(?:eu )?prefiro ([\w\s,]{3,60})", re.I), "procedural", "prefere {0}"),
    (re.compile(r"\b(?:eu )?(?:n[aã]o gosto|odeio|detesto) (?:de )?([\w\s]{3,40})", re.I),
     "procedural", "não gosta de {0}"),
    (re.compile(r"\b(?:eu )?gosto de ([\w\s]{3,40})", re.I), "personalidade", "gosta de {0}"),

    # pedido explicito de memorizacao -- alta confianca
    (re.compile(r"\b(?:lembra|lembre|guarda|anota) que ([^\.!?\n]{5,120})", re.I),
     "semantica", "{0}"),

    # eventos
    (re.compile(r"\b(?:eu )?comecei (?:a |o |um |uma )?([\w\s]{3,50})", re.I),
     "episodica", "começou {0}"),
    (re.compile(r"\b(?:eu )?terminei (?:a |o |um |uma )?([\w\s]{3,50})", re.I),
     "episodica", "terminou {0}"),
    (re.compile(r"\b(?:eu )?consegui (?:a |o |um |uma )?([\w\s]{3,50})", re.I),
     "episodica", "conseguiu {0}"),
]

# Trechos que indicam hipotese, negacao ou pergunta -- nao sao declaracao
# de fato e nao devem virar memoria.
BLOQUEIOS = re.compile(
    r"\b(se eu|caso eu|talvez|acho que|será que|e se|imagina se|queria|"
    r"gostaria|pretendo|vou tentar|não sei se|sonho em)\b", re.I
)

PROMPT_LLM = """Extraia fatos duradouros sobre o usuário a partir da conversa.

Regras:
- Só extraia o que o usuário afirmou sobre si. Nada de suposição.
- Ignore hipóteses, perguntas e desejos ("queria", "talvez", "e se").
- Ignore o que é passageiro ("estou com fome agora").
- Escreva na terceira pessoa, curto e direto.
- Se não houver nada duradouro, chame a ferramenta com uma lista vazia.

Conversa:
{conversa}"""

# Schema OpenAI para tool-calling nativo -- ver ClienteLLM.completar_com_
# ferramenta em llm.py. Substituiu o formato antigo (pedir "responda em
# JSON" na instrução e parsear o texto solto da resposta): o modelo é
# treinado especificamente para produzir tool_calls estruturado quando
# recebe um schema assim, e não necessariamente para seguir uma instrução
# textual pedindo um formato JSON arbitrário -- daí a diferença real de
# confiabilidade entre os dois caminhos.
FERRAMENTA_EXTRACAO = {
    "type": "function",
    "function": {
        "name": "registrar_fatos",
        "description": "Registra fatos duradouros extraídos sobre o usuário.",
        "parameters": {
            "type": "object",
            "properties": {
                "fatos": {
                    "type": "array",
                    "description": "Lista de fatos extraídos. Vazia se não houver nada duradouro.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tipo": {
                                "type": "string",
                                "enum": ["semantica", "episodica", "procedural", "personalidade"],
                            },
                            "conteudo": {
                                "type": "string",
                                "description": "O fato, em terceira pessoa, curto.",
                            },
                        },
                        "required": ["tipo", "conteudo"],
                    },
                },
            },
            "required": ["fatos"],
        },
    },
}


def _limpar(texto: str) -> str:
    t = re.sub(r"\s+", " ", texto).strip(" .,;:!?")
    # corta em conectivo, para nao capturar meia frase seguinte
    t = re.split(r"\b(?:e|mas|porque|porém|então|que|pra|para)\b", t, maxsplit=1)[0]
    return t.strip(" .,;:!?")


def extrair_por_regras(mensagem: str, min_palavras: int = 1) -> list[dict]:
    """Extrai memorias de uma mensagem do usuario usando padroes."""
    if BLOQUEIOS.search(mensagem):
        return []

    achados: list[dict] = []
    vistos: set[str] = set()

    for padrao, tipo, template in REGRAS:
        for m in padrao.finditer(mensagem):
            valor = _limpar(m.group(1))
            if not valor or len(valor.split()) < min_palavras or len(valor) < 3:
                continue
            conteudo = template.format(valor)
            chave = conteudo.lower()
            if chave in vistos:
                continue
            vistos.add(chave)
            achados.append({
                "tipo": tipo,
                "conteudo": conteudo,
                "fonte": "regra",
                "confianca": 0.85,
            })

    return achados


def extrair_por_llm(conversa: list[dict], cliente, max_fatos: int = 5) -> list[dict]:
    """Extrai memorias usando o LLM. Mais cobertura, menos precisao.

    Usa tool-calling nativo (completar_com_ferramenta), não instrução de
    prompt pedindo JSON solto -- é o caminho que o modelo foi de fato
    treinado a seguir com confiabilidade, ver o schema FERRAMENTA_EXTRACAO
    acima e o docstring de ClienteLLM.completar_com_ferramenta.

    Marca fonte='llm' e confianca menor de proposito -- assim da pra
    auditar depois o que veio de inferencia e nao de declaracao direta.
    """
    texto = "\n".join(
        f"{'Usuário' if t['role'] == 'user' else 'EVA'}: {t['content']}"
        for t in conversa[-8:]
    )
    prompt = PROMPT_LLM.replace("{conversa}", texto)

    try:
        argumentos = cliente.completar_com_ferramenta(
            [{"role": "user", "content": prompt}],
            FERRAMENTA_EXTRACAO,
            temperatura=0.0, max_tokens=300,
        )
    except Exception:
        return []

    if argumentos is None:
        return []  # modelo não chamou a ferramenta -- nada a extrair

    dados_fatos = argumentos.get("fatos", [])
    if not isinstance(dados_fatos, list):
        return []  # "fatos" não é lista -- formato irreconhecível, desiste

    saida = []
    for f in dados_fatos[:max_fatos]:
        # O schema pede {"tipo":..., "conteudo":...}, mas mesmo com
        # tool-calling nativo um backend pode devolver a string do fato
        # direto, sem envelope, ocasionalmente -- mantém a tolerância dos
        # dois formatos como rede de segurança, não assume que o schema
        # sozinho garante 100% de adesão.
        if isinstance(f, dict):
            conteudo = str(f.get("conteudo", "")).strip()
            tipo = f.get("tipo", "semantica")
        elif isinstance(f, str):
            conteudo = f.strip()
            tipo = "semantica"
        else:
            continue  # formato irreconhecível (número, lista aninhada, etc) -- pula esse item, não trava os outros

        if conteudo and tipo in ("semantica", "episodica", "procedural", "personalidade"):
            saida.append({
                "tipo": tipo,
                "conteudo": conteudo,
                "fonte": "llm",
                "confianca": 0.55,
            })
    return saida