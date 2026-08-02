"""
Decision Engine -- o "lobo frontal" da EVA.

Regra da arquitetura: ele NUNCA conversa. Só decide, e devolve um plano
estruturado. Quem escreve português é a EVA; quem escolhe o que buscar,
lembrar e executar é este módulo.

Essa separação é o que permite treinar os dois de forma independente: o
conversacional é otimizado para diálogo, o decisor para consistência de
formato. Um modelo bom em conversa é ruim em produzir JSON estável, e
vice-versa.

Duas implementações:

REGRAS (padrão): heurísticas sobre o texto. Determinísticas, instantâneas
e de graça. Cobrem bem os casos frequentes -- que são a maioria.

LLM (opcional): manda a mensagem para um modelo pequeno e pede o plano em
JSON. Cobre mais casos, ao custo de latência e de variabilidade. No plano
original do projeto, este seria um modelo de 100M-500M treinado só para
isso.

A saída é a mesma nos dois casos, então trocar não afeta o resto.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field


@dataclass
class Plano:
    """O que o Decision Engine devolve."""
    intencao: str = "conversa"
    precisa_memoria: bool = True
    precisa_ferramenta: bool = False
    ferramentas: list[dict] = field(default_factory=list)  # [{"nome":..., "args":{...}}]
    consulta_memoria: str = ""
    prioridade: str = "normal"           # normal | alta
    carga_emocional: float = 0.0
    novidade: float = 0.5
    complexidade: float = 0.5
    guardar_memoria: bool = True
    motivo: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------- padrões

# Sinais de conteúdo emocional. Peso maior para o que indica sofrimento
# real, porque isso muda a prioridade e desliga o humor.
EMOCIONAL_FORTE = re.compile(
    r"\b(sozinh|solid[ãa]o|deprim|ansiedad|ansios|p[âa]nico|angustia|angústia|"
    r"desesper|sem sentido|n[ãa]o aguento|cansad[oa] de tudo|vontade de sumir|"
    r"morrer|morte|luto|faleceu|morreu|perdi meu|perdi minha|chorar|chorei|"
    r"medo|assustad|traum)\w*", re.I
)
EMOCIONAL_MEDIO = re.compile(
    r"\b(triste|magoa|magoada|magoado|frustrad|irritad|raiva|nervos|preocupad|"
    r"estress|culpa|vergonha|arrepend|sauda)\w*", re.I
)

# Sinais de crise -- exigem tratamento prioritário e nunca humor.
CRISE = re.compile(
    r"\b(me matar|suic[íi]d|acabar com tudo|n[ãa]o quero mais viver|"
    r"sumir de vez|me cortar|me machucar|tirar minha vida)\w*", re.I
)

TEMPORAL = re.compile(
    r"\b(hoje|agora|amanh[ãa]|ontem|que dia|que horas|semana que vem|"
    r"esse m[êe]s|neste momento|atualmente)\b", re.I
)

CONTA = re.compile(r"\d+\s*[\+\-\*/×÷\^]\s*\d+|quanto\s+[ée]\s+\d|calcul\w*", re.I)

CLIMA = re.compile(r"\b(clima|tempo|chuva|chover|temperatura|calor|frio|graus)\b", re.I)

# Pedido de busca. Exige forma IMPERATIVA ou pergunta direta -- verbo no
# passado ("pesquisei tanto e não achei sentido") é relato, não pedido, e
# tratá-lo como busca faz a EVA sair procurando na web enquanto a pessoa
# estava desabafando.
BUSCA = re.compile(
    r"(\b(pesquis[ae]|busca|procura|pesquise|busque|procure|me acha|acha a[íi]|"
    r"d[áa] uma olhada|v[êe] a[íi])\b"
    r"|\b(quem [ée]|o que [ée] o|quanto custa|qual o pre[çc]o|"
    r"[úu]ltimas not[íi]cias|not[íi]cias de hoje)\b)", re.I
)

# Formas no passado que parecem busca mas são relato pessoal
BUSCA_RELATO = re.compile(
    r"\b(pesquisei|busquei|procurei|andei pesquisando|tentei achar|"
    r"j[áa] pesquisei|j[áa] procurei)\b", re.I
)

# Perguntas sobre a própria EVA -- não precisam de memória do usuário nem
# de ferramenta, e buscar memória aqui só traz ruído.
SOBRE_SI = re.compile(
    r"\b(seu nome|voc[êe] [ée]|quem [ée] voc[êe]|o que voc[êe] [ée]|"
    r"voc[êe] sente|voc[êe] gosta|voc[êe] tem consci[êe]ncia|"
    r"quem te criou|voc[êe] [ée] humana|voc[êe] lembra)\b", re.I
)

SAUDACAO = re.compile(
    r"^\s*(oi|ol[áa]|e a[íi]|bom dia|boa tarde|boa noite|tudo bem|opa|hey)\b[\s!?.]*$",
    re.I
)

CIDADE = re.compile(
    r"\b(?:em|no|na|de|para|pra)\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+(?:\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][\wÀ-ÿ]+)?)"
)


class DecisorPorRegras:
    """Decisor determinístico baseado em padrões de texto."""

    def decidir(self, mensagem: str, historico: list[dict] | None = None) -> Plano:
        p = Plano()
        texto = mensagem.strip()
        p.consulta_memoria = texto

        # --- crise tem precedência sobre tudo ---
        if CRISE.search(texto):
            p.intencao = "crise"
            p.prioridade = "alta"
            p.carga_emocional = 1.0
            p.complexidade = 1.0
            p.precisa_ferramenta = False
            # não guardamos memória automática aqui: o momento pede atenção,
            # não coleta de dados sobre a pessoa
            p.guardar_memoria = False
            p.motivo = "sinais de crise detectados"
            return p

        # --- saudação simples: nada de contexto pesado ---
        if SAUDACAO.match(texto) and len(texto) < 25:
            p.intencao = "saudacao"
            p.precisa_memoria = False
            p.novidade = 0.2
            p.complexidade = 0.1
            p.guardar_memoria = False
            p.motivo = "saudação"
            return p

        # --- pergunta sobre a própria EVA ---
        if SOBRE_SI.search(texto):
            p.intencao = "sobre_si"
            p.precisa_memoria = False
            p.complexidade = 0.5
            p.guardar_memoria = False
            p.motivo = "pergunta sobre a própria EVA"
            return p

        # --- carga emocional ---
        if EMOCIONAL_FORTE.search(texto):
            p.intencao = "emocional"
            p.carga_emocional = 0.8
            p.prioridade = "alta"
            p.complexidade = 0.8
            p.motivo = "conteúdo emocional forte"
        elif EMOCIONAL_MEDIO.search(texto):
            p.intencao = "emocional"
            p.carga_emocional = 0.5
            p.complexidade = 0.6
            p.motivo = "conteúdo emocional"

        # --- ferramentas ---
        ferramentas = []

        if CONTA.search(texto):
            expr = self._extrair_expressao(texto)
            if expr:
                ferramentas.append({"nome": "calcular", "args": {"expressao": expr}})

        if CLIMA.search(texto):
            cidade = self._extrair_cidade(texto)
            if cidade:
                ferramentas.append({"nome": "clima", "args": {"cidade": cidade}})

        if TEMPORAL.search(texto):
            ferramentas.append({"nome": "hora_atual", "args": {}})

        # Busca só com pedido explícito, sem carga emocional alta e sem ser
        # relato no passado. "pesquisei tanto e não achei sentido" é desabafo.
        if BUSCA.search(texto) and p.carga_emocional < 0.5 and not BUSCA_RELATO.search(texto):
            ferramentas.append({"nome": "buscar", "args": {"consulta": texto[:200]}})

        if ferramentas:
            p.precisa_ferramenta = True
            p.ferramentas = ferramentas
            if p.intencao == "conversa":
                p.intencao = "tarefa"
            p.motivo = (p.motivo + "; " if p.motivo else "") + \
                       f"ferramentas: {', '.join(f['nome'] for f in ferramentas)}"

        # --- complexidade e novidade por tamanho e histórico ---
        n_palavras = len(texto.split())
        if p.complexidade == 0.5:
            p.complexidade = min(1.0, 0.25 + n_palavras / 80)

        if historico:
            p.novidade = self._novidade(texto, historico)

        if not p.motivo:
            p.motivo = "conversa comum"
        return p

    def _extrair_expressao(self, texto: str) -> str | None:
        m = re.search(r"[\d\.,]+(?:\s*[\+\-\*/×÷\^]\s*[\d\.,]+)+", texto)
        return m.group(0) if m else None

    def _extrair_cidade(self, texto: str) -> str | None:
        m = CIDADE.search(texto)
        if m:
            return m.group(1)
        return None

    def _novidade(self, texto: str, historico: list[dict]) -> float:
        """Quanto o assunto difere do que já foi conversado."""
        palavras = set(re.findall(r"\w{4,}", texto.lower()))
        if not palavras:
            return 0.3
        anteriores: set[str] = set()
        for t in historico[-6:]:
            anteriores |= set(re.findall(r"\w{4,}", t.get("content", "").lower()))
        if not anteriores:
            return 0.8
        sobreposicao = len(palavras & anteriores) / len(palavras)
        return max(0.1, min(1.0, 1.0 - sobreposicao))


PROMPT_DECISOR = """Você é um planejador. Não converse, não responda ao usuário.
Analise a mensagem e devolva APENAS um JSON:

{{"intencao": "conversa|tarefa|emocional|sobre_si|saudacao",
 "precisa_memoria": true/false,
 "ferramentas": [{{"nome": "...", "args": {{}}}}],
 "carga_emocional": 0.0-1.0,
 "novidade": 0.0-1.0,
 "complexidade": 0.0-1.0}}

Ferramentas disponíveis:
{ferramentas}

Mensagem: {mensagem}"""


class DecisorPorLLM:
    """Decisor que usa um modelo. Cai para regras se o modelo falhar."""

    def __init__(self, cliente, registro, fallback: DecisorPorRegras | None = None):
        self.cliente = cliente
        self.registro = registro
        self.fallback = fallback or DecisorPorRegras()

    def decidir(self, mensagem: str, historico: list[dict] | None = None) -> Plano:
        prompt = PROMPT_DECISOR.format(
            ferramentas=self.registro.descrever(), mensagem=mensagem
        )
        try:
            bruto = self.cliente.completar(
                [{"role": "user", "content": prompt}], temperatura=0.0, max_tokens=200
            )
            m = re.search(r"\{.*\}", bruto, re.S)
            if not m:
                raise ValueError("sem JSON na resposta")
            d = json.loads(m.group(0))
        except Exception:
            # Decisor quebrado não pode derrubar a conversa -- as regras
            # cobrem o caso e o usuário nem percebe.
            plano = self.fallback.decidir(mensagem, historico)
            plano.motivo = "fallback: decisor LLM falhou"
            return plano

        # crise sempre é reavaliada por regra: é grave demais para depender
        # de o modelo ter classificado certo
        base = self.fallback.decidir(mensagem, historico)
        if base.intencao == "crise":
            return base

        ferramentas = [
            f for f in d.get("ferramentas", [])
            if isinstance(f, dict) and self.registro.get(f.get("nome", ""))
        ]
        return Plano(
            intencao=d.get("intencao", "conversa"),
            precisa_memoria=bool(d.get("precisa_memoria", True)),
            precisa_ferramenta=bool(ferramentas),
            ferramentas=ferramentas,
            consulta_memoria=mensagem,
            prioridade="alta" if float(d.get("carga_emocional", 0)) > 0.7 else "normal",
            carga_emocional=float(d.get("carga_emocional", 0.0)),
            novidade=float(d.get("novidade", 0.5)),
            complexidade=float(d.get("complexidade", 0.5)),
            motivo="decisor LLM",
        )
