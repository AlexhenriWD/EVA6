"""
Decision Engine -- o "lobo frontal" da EVA.

Regra da arquitetura: ele NUNCA conversa. S├│ decide, e devolve um plano
estruturado. Quem escreve portugu├¬s ├® a EVA; quem escolhe o que buscar,
lembrar e executar ├® este m├│dulo.

Essa separa├º├úo ├® o que permite treinar os dois de forma independente: o
conversacional ├® otimizado para di├ílogo, o decisor para consist├¬ncia de
formato. Um modelo bom em conversa ├® ruim em produzir JSON est├ível, e
vice-versa.

Duas implementa├º├Áes:

REGRAS (padr├úo): heur├¡sticas sobre o texto. Determin├¡sticas, instant├óneas
e de gra├ºa. Cobrem bem os casos frequentes -- que s├úo a maioria.

LLM (opcional): manda a mensagem para um modelo pequeno e pede o plano em
JSON. Cobre mais casos, ao custo de lat├¬ncia e de variabilidade. No plano
original do projeto, este seria um modelo de 100M-500M treinado s├│ para
isso.

A sa├¡da ├® a mesma nos dois casos, ent├úo trocar n├úo afeta o resto.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace


@dataclass
class Plano:
    """O que o Decision Engine devolve."""
    intencao: str = "conversa"
    precisa_memoria: bool = True
    precisa_ferramenta: bool = False
    precisa_visao: bool = False
    precisa_jogo: bool = False
    ferramentas: list[dict] = field(default_factory=list)  # [{"nome":..., "args":{...}}]
    consulta_memoria: str = ""
    prioridade: str = "normal"           # normal | alta
    carga_emocional: float = 0.0
    novidade: float = 0.5
    complexidade: float = 0.5
    guardar_memoria: bool = True
    motivo: str = ""

    # Sinal de "isso pode estar fora do que a EVA sabe" -- n├úo dispara
    # ferramenta no turno atual (diferente de precisa_ferramenta/buscar,
    # que j├í busca IMEDIATO para responder). Este ├® para o orquestrador
    # pesquisar em SEGUNDO PLANO e guardar o achado para trazer ├á tona
    # depois, na iniciativa, via Consciencia.pesquisa_pronta(). Ver
    # LACUNA_CONHECIMENTO em decision.py.
    possivel_lacuna: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------- padr├Áes

# Sinais de conte├║do emocional. Peso maior para o que indica sofrimento
# real, porque isso muda a prioridade e desliga o humor.
EMOCIONAL_FORTE = re.compile(
    r"\b(sozinh|solid[├úa]o|deprim|ansiedad|ansios|p[├óa]nico|angustia|ang├║stia|"
    r"desesper|sem sentido|n[├úa]o aguento|cansad[oa] de tudo|vontade de sumir|"
    r"morrer|morte|luto|faleceu|morreu|perdi meu|perdi minha|chorar|chorei|"
    r"medo|assustad|traum)\w*", re.I
)
EMOCIONAL_MEDIO = re.compile(
    r"\b(triste|magoa|magoada|magoado|frustrad|irritad|raiva|nervos|preocupad|"
    r"estress|culpa|vergonha|arrepend|sauda)\w*", re.I
)

# Sinais de crise -- exigem tratamento priorit├írio e nunca humor.
CRISE = re.compile(
    r"\b(me matar|suic[├¡i]d|acabar com tudo|n[├úa]o quero mais viver|"
    r"sumir de vez|me cortar|me machucar|tirar minha vida)\w*", re.I
)

TEMPORAL = re.compile(
    r"\b(hoje|agora|amanh[├úa]|ontem|que dia|que horas|semana que vem|"
    r"esse m[├¬e]s|neste momento|atualmente)\b", re.I
)

CONTA = re.compile(r"\d+\s*[\+\-\*/├ù├À\^]\s*\d+|quanto\s+[├®e]\s+\d|calcul\w*", re.I)

CLIMA = re.compile(r"\b(clima|tempo|chuva|chover|temperatura|calor|frio|graus)\b", re.I)

# ---------------------------------------------------------------- vis├úo
#
# N├úo decide se a EVA V├è algo -- isso ├® do SistemaVisual (vision/visao.py),
# que roda por fora, independente de decis├úo. Isto decide se, num turno
# em que existe uma cena capturada, vale a pena INJETAR ela no prompt.
#
# Sem este port├úo, contexto_visual entraria em TODO turno sempre que a
# vis├úo estivesse ligada -- inclusive conversa sem nenhuma rela├º├úo com a
# tela. ├ë o mesmo risco de "narradora" da consci├¬ncia (falar sobre tudo
# que muda vira insuport├ível), s├│ que aplicado ao n├¡vel da conversa
# inteira em vez da fala espont├ónea: contexto visual perene polui o
# prompt e pode puxar a resposta pra comentar a tela sem vir ao caso.
#
# Refer├¬ncia DE├ìTICA (aponta pra tela: "olha isso", "v├¬ aqui") ou men├º├úo
# a artefato de tela (jogo, c├│digo, documento) -- os casos em que a
# pessoa est├í claramente falando do que est├í vendo. Falso negativo
# (n├úo detectar quando devia) ├® prefer├¡vel a falso positivo aqui: melhor
# a EVA ocasionalmente "esquecer" de olhar a tela do que ficar comentando
# a tela em toda resposta sem necessidade.
VISAO = re.compile(
    r"\b(v[├¬e] (isso|aqui|a[├¡i])|olha (isso|aqui|a tela)|olhando (isso|aqui)|"
    r"enxerg\w*|"                    # NOVO: "enxergar a tela", "consegue enxergar"
    r"\btela\b|"                     # NOVO: qualquer men├º├úo a "tela" j├í ├® sinal forte
    r"sua vis[├úa]o|"                 # NOVO: "problema com a sua vis├úo"
    r"o que (eu )?(t[├┤o]|estou) (fazendo|jogando|vendo|mostrando)|"
    r"(esse|essa) (jogo|c[├│o]digo|documento|programa|janela)|"
    r"t[├ía] vendo (isso|aqui)|consegue ver|voc[├¬e] (t[├ía] |est[├ía] )?vendo)\b",
    re.I,
)

ROBO_OLHAR = re.compile(
    r"\b((pode|poderia)\s+dar|d[áa]) uma olhada em volta|"
    r"\b(olh(ar|e)(\s+um pouco)? em volta|olh(ar|e) ao redor|"
    r"olh(ar|e) para os lados|vir(a|e) a cabe[çc]a|olh(ar|e) em torno)\b",
    re.I,
)
ROBO_ESTADO = re.compile(
    r"\b(o que (voc[êe] )?(est[áa]|t[áa]) enxergando (pelo|no|do|com o|através do) rob[ôo]|"
    r"o que o rob[ôo] (est[áa]|t[áa]) vendo|"
    r"c[âa]mera (do|no) rob[ôo]|"
    r"estado do rob[ôo]|sensores do rob[ôo]|bateria do rob[ôo])\b",
    re.I,
)

JOGO_ESTADO = re.compile(
    r"\b(no minecraft|no jogo|no servidor|seu invent[áa]rio|quanto de vida)\b",
    re.I,
)


def visao_relevante(texto: str) -> bool:
    """Se vale injetar contexto_visual neste turno.

    Fun├º├úo p├║blica (n├úo s├│ uso interno de DecisorPorRegras) porque quem
    integra a vis├úo (bridge_client.py) precisa decidir isso ANTES de
    chamar EVA.responder() -- contexto_visual ├® um par├ómetro de entrada,
    n├úo algo que o orquestrador busca sozinho. Mesma regra dos dois
    lados: aqui alimenta Plano.precisa_visao (vis├¡vel em Resultado, pra
    quem quiser auditar a decis├úo depois), e bridge_client chama esta
    fun├º├úo direto no texto cru antes de decidir se passa a cena ou None.
    """
    return bool(VISAO.search(texto))


def robo_olhar_relevante(texto: str) -> bool:
    return bool(ROBO_OLHAR.search(texto))


def robo_estado_relevante(texto: str) -> bool:
    return bool(ROBO_ESTADO.search(texto))


def minecraft_estado_relevante(texto: str) -> bool:
    return bool(JOGO_ESTADO.search(texto))

# Pedido de busca. Exige forma IMPERATIVA ou pergunta direta -- verbo no
# passado ("pesquisei tanto e n├úo achei sentido") ├® relato, n├úo pedido, e
# trat├í-lo como busca faz a EVA sair procurando na web enquanto a pessoa
# estava desabafando.
# Pedido de busca. Exige forma IMPERATIVA, INFINITIVA (depois de modal:
# "pode/consegue/d├í pra pesquisar") ou pergunta direta -- verbo no passado
# ("pesquisei tanto e n├úo achei sentido") ├® relato, n├úo pedido, e trat├í-lo
# como busca faz a EVA sair procurando na web enquanto a pessoa estava
# desabafando.
#
# BUG REAL J├ü VISTO: "pesquis[ae]\b" casa "pesquisa"/"pesquise" mas N├âO
# "pesquisar" -- depois de "pesquisa" vem "r" colado, sem fronteira de
# palavra ali, ent├úo o \b falha. Isso faz "voc├¬ pode pesquisar pra mim"
# (forma mais natural e educada que o imperativo seco "pesquisa isso")
# nunca disparar a ferramenta -- silenciosamente, sem erro nenhum. A EVA
# respondia do que j├í sabia (desatualizado) e n├úo tinha como perceber
# que a busca nunca rodou. Por isso agora cada verbo cobre explicitamente
# a forma no infinitivo tamb├®m: pesquis(a|e|ar), n├úo s├│ pesquis[ae].
# BUG REAL JA VISTO (2a vez, conjugacao diferente): "pesquisar[ae]" so
# cobria imperativo e infinitivo. "eu gostaria que voc├¬ pesquisasse"
# (subjuntivo imperfeito, depois de verbo de vontade -- forma natural e
# comum em portugues) nao batia em nada, e a busca nunca disparava, em
# silencio. Agora cobre tambem subjuntivo (-asse) e condicional (-aria).
#
# Isso e enumeracao de forma verbal, e portugues tem mais conjugacao do
# que da pra enumerar por regex com confianca de cobertura total -- essa
# e a segunda lacuna encontrada, provavelmente nao a ultima. Se continuar
# acontecendo com outras frases naturais, o proximo passo nao e continuar
# remendando aqui: e ligar EVA_DECISION_LLM=1 (ja configuravel, aponta pro
# minicpm-v-4.6 que ja provou suportar tool-calling nativo) para que a
# deteccao de intencao use compreensao de linguagem de verdade em vez de
# casar padrao de texto.
BUSCA = re.compile(
    r"(\b(pesquis(a|e|ar|asse|aria)|busc(a|e|ar|asse|aria)|procur(a|e|ar|asse|aria)|"
    r"me acha|acha a[├¡i]|d[├ía] uma olhada|v[├¬e] a[├¡i])\b"
    r"|\b(quem [├®e]|o que [├®e] o|quanto custa|qual o pre[├ºc]o|"
    r"[├║u]ltimas not[├¡i]cias|not[├¡i]cias de hoje)\b)", re.I
)

# Formas no passado que parecem busca mas s├úo relato pessoal
BUSCA_RELATO = re.compile(
    r"\b(pesquisei|busquei|procurei|andei pesquisando|tentei achar|"
    r"j[├ía] pesquisei|j[├ía] procurei)\b", re.I
)

# ------------------------------------------------- consulta de busca real
#
# BUSCA s├│ detecta QUE existe pedido de busca -- n├úo separa o pedido do
# assunto. "pesquisa mais sobre isso" batia em BUSCA e a consulta virava
# essa frase literal, mandada pro SearXNG palavra por palavra. Motor de
# busca n├úo resolve "isso" (n├úo tem o resto da conversa), ent├úo voltava
# lixo ou nada -- e por fora parecia a EVA "se recusando a pesquisar",
# quando na verdade pesquisou a frase errada.
_GATILHO_BUSCA = re.compile(
    r"^\s*(voc[├¬e]\s+)?(pode|poderia|consegue|conseguiria|d[├ía]\s+pra|"
    r"d[├ía]\s+para|gostaria\s+que\s+voc[├¬e])?\s*"
    r"(pesquis(a|e|ar|asse|aria)|busc(a|e|ar|asse|aria)|procur(a|e|ar|asse|aria)|"
    r"me\s+ach(a|ar)|ach(a|ar)\s+a[├¡i]|d[├ía]\s+uma\s+olhada(\s+em)?|v[├¬e]\s+a[├¡i])\s*",
    re.I,
)

_SO_ENCHIMENTO = re.compile(
    r"^(sobre\s+|em\s+|no\s+|na\s+)?"
    r"(isso|isto|aquilo|esse\s+assunto|essa\s+coisa|a[├¡i]|"
    r"mais(\s+(sobre\s+)?(isso|isto|aquilo))?|pra\s+mim|por\s+favor|)"
    r"[\s.,!?]*$",
    re.I,
)


def extrair_consulta_busca(texto: str, historico: list[dict] | None = None) -> str | None:
    """Assunto de um pedido de busca -- nunca o pedido cru.

    Duas etapas: tira o verbo-gatilho e o modal que vem junto; se o que
    sobra ├® s├│ enchimento sem assunto (pronome, "mais sobre", "por
    favor"), tenta resolver olhando a ├ÜLTIMA mensagem do usu├írio no
    hist├│rico -- ├® o caso mais comum de pronome apontando pra fora da
    frase atual ("o que rolou de novo?" ... "pesquisa mais sobre isso").

    Devolve None quando n├úo h├í assunto extra├¡vel nem no hist├│rico -- quem
    chama decide o que fazer (busca expl├¡cita cai pro texto original como
    ├║ltimo recurso; lacuna em segundo plano simplesmente desiste).
    """
    resto = _GATILHO_BUSCA.sub("", texto.strip(), count=1).strip(" ,.!?")

    if resto and len(resto) >= 4 and not _SO_ENCHIMENTO.match(resto):
        return resto

    if historico:
        for turno in reversed(historico):
            if turno.get("role") != "user":
                continue
            candidato = (turno.get("content") or "").strip()
            if candidato and len(candidato) >= 4 and not _SO_ENCHIMENTO.match(candidato):
                return candidato[:200]

    return None

# ------------------------------------------------- lacuna de conhecimento
#
# Diferente de BUSCA (pedido expl├¡cito, busca AGORA para responder), isto
# ├® sobre tema que SOA como fato-que-muda mesmo sem pedido de busca --
# "o que voc├¬ acha do mercado de IA agora", "fale sobre a F├│rmula 1 esse
# ano". A EVA responde do que j├í sabe (treino), que pode estar
# desatualizado, e o usu├írio n├úo tem como perceber isso.
#
# N├úo ├® substituto de julgamento -- ├® rede grosseira e r├ípida (regex, zero
# custo), porque n├úo pode ter o delay de uma chamada de LLM em todo turno.
# Ela vai ERRAR: vai marcar coisa que n├úo precisava, vai perder coisa
# sutil. ├ë aceit├ível porque o efeito de marcar errado ├® pesquisa em
# segundo plano que talvez nunca seja usada (TTL expira, ver
# consciousness.py) -- n├úo ├® dito ao usu├írio, n├úo interrompe nada.
#
# Dois sinais, and n├úo or: sozinho, "mercado" ou "atual" aparecem demais
# em conversa comum. Junto -- tema que muda (mercado, tecnologia, vers├úo,
# not├¡cia, pre├ºo) E marcador de tempo presente/recente (agora, hoje, esse
# ano, atualmente) -- ├® bem mais espec├¡fico do que exige checagem.
LACUNA_TEMA = re.compile(
    r"\b(mercado|tecnologia|ia\b|intelig[├¬e]ncia artificial|not[├¡i]cia|"
    r"pre[├ºc]o|vers[├úa]o|lan[├ºc]amento|elei├º[├úa]o|governo|guerra|"
    r"empresa|startup|criptomoeda|bolsa|a[├ºc][├Áo]es)\b", re.I
)
LACUNA_TEMPORAL = re.compile(
    r"\b(agora|hoje|atualmente|esse ano|este ano|ultimamente|"
    r"recentemente|nos [├║u]ltimos|em 202\d)\b", re.I
)

# Perguntas sobre a pr├│pria EVA -- n├úo precisam de mem├│ria do usu├írio nem
# de ferramenta, e buscar mem├│ria aqui s├│ traz ru├¡do.
SOBRE_SI = re.compile(
    r"\b(seu nome|voc[├¬e] [├®e]|quem [├®e] voc[├¬e]|o que voc[├¬e] [├®e]|"
    r"voc[├¬e] sente|voc[├¬e] gosta|voc[├¬e] tem consci[├¬e]ncia|"
    r"quem te criou|voc[├¬e] [├®e] humana|voc[├¬e] lembra)\b", re.I
)

SAUDACAO = re.compile(
    r"^\s*(oi|ol[├ía]|e a[├¡i]|bom dia|boa tarde|boa noite|tudo bem|opa|hey)\b[\s!?.]*$",
    re.I
)

CIDADE = re.compile(
    r"\b(?:em|no|na|de|para|pra)\s+([A-Z├ü├ë├ì├ô├Ü├é├è├ö├â├ò├ç][\w├Ç-├┐]+(?:\s+[A-Z├ü├ë├ì├ô├Ü├é├è├ö├â├ò├ç][\w├Ç-├┐]+)?)"
)


class DecisorPorRegras:
    """Decisor determin├¡stico baseado em padr├Áes de texto."""

    def decidir(self, mensagem: str, historico: list[dict] | None = None) -> Plano:
        p = Plano()
        texto = mensagem.strip()
        p.consulta_memoria = texto
        # Setado ANTES de qualquer retorno antecipado (crise, sauda├º├úo,
        # sobre_si) -- refer├¬ncia ├á tela pode aparecer em mensagem curta
        # ("olha isso") que de outra forma sairia num desses atalhos sem
        # passar pelo resto da fun├º├úo.
        p.precisa_visao = visao_relevante(texto) and not robo_estado_relevante(texto)
        p.precisa_jogo = minecraft_estado_relevante(texto)

        # --- crise tem preced├¬ncia sobre tudo ---
        if CRISE.search(texto):
            p.intencao = "crise"
            p.prioridade = "alta"
            p.carga_emocional = 1.0
            p.complexidade = 1.0
            p.precisa_ferramenta = False
            # n├úo guardamos mem├│ria autom├ítica aqui: o momento pede aten├º├úo,
            # n├úo coleta de dados sobre a pessoa
            p.guardar_memoria = False
            p.motivo = "sinais de crise detectados"
            return p

        # --- sauda├º├úo simples: nada de contexto pesado ---
        if SAUDACAO.match(texto) and len(texto) < 25:
            p.intencao = "saudacao"
            p.precisa_memoria = False
            p.novidade = 0.2
            p.complexidade = 0.1
            p.guardar_memoria = False
            p.motivo = "sauda├º├úo"
            return p

        # --- pergunta sobre a pr├│pria EVA ---
        if SOBRE_SI.search(texto):
            p.intencao = "sobre_si"
            p.precisa_memoria = False
            p.complexidade = 0.5
            p.guardar_memoria = False
            p.motivo = "pergunta sobre a pr├│pria EVA"
            return p

        # --- carga emocional ---
        if EMOCIONAL_FORTE.search(texto):
            p.intencao = "emocional"
            p.carga_emocional = 0.8
            p.prioridade = "alta"
            p.complexidade = 0.8
            p.motivo = "conte├║do emocional forte"
        elif EMOCIONAL_MEDIO.search(texto):
            p.intencao = "emocional"
            p.carga_emocional = 0.5
            p.complexidade = 0.6
            p.motivo = "conte├║do emocional"

        # --- ferramentas ---
        ferramentas = []

        olhar_robo_acionado = robo_olhar_relevante(texto)
        if olhar_robo_acionado:
            ferramentas.append({"nome": "robo_olhar", "args": {}})
        if robo_estado_relevante(texto):
            ferramentas.append({"nome": "robo_estado", "args": {}})

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

        # Busca s├│ com pedido expl├¡cito, sem carga emocional alta e sem ser
        # relato no passado. "pesquisei tanto e n├úo achei sentido" ├® desabafo.
        busca_ja_acionada = olhar_robo_acionado
        if (not busca_ja_acionada and BUSCA.search(texto)
            and p.carga_emocional < 0.5 and not BUSCA_RELATO.search(texto)):
            consulta = extrair_consulta_busca(texto, historico)
            if consulta is None:
                consulta = texto[:200]
                print(f"[decisao] busca sem assunto extra├¡vel, usando texto cru: {texto[:80]!r}")
            ferramentas.append({"nome": "buscar", "args": {"consulta": consulta}})
            busca_ja_acionada = True

        # Lacuna de conhecimento: n├úo busca AGORA (n├úo atrasa a resposta),
        # s├│ marca para o orquestrador pesquisar em segundo plano. S├│ marca
        # se a busca imediata n├úo disparou -- sen├úo a mesma pesquisa
        # aconteceria duas vezes por motivos diferentes.
        if (not busca_ja_acionada and p.carga_emocional < 0.5
                and LACUNA_TEMA.search(texto) and LACUNA_TEMPORAL.search(texto)):
            # Oportunista: se n├úo achou assunto de verdade, n├úo pesquisa nada em
            # vez de gastar cota do Brave numa frase-gatilho sem conte├║do.
            p.possivel_lacuna = extrair_consulta_busca(texto, historico) or texto[:200]

        if ferramentas:
            p.precisa_ferramenta = True
            p.ferramentas = ferramentas
            if p.intencao == "conversa":
                p.intencao = "tarefa"
            p.motivo = (p.motivo + "; " if p.motivo else "") + \
                       f"ferramentas: {', '.join(f['nome'] for f in ferramentas)}"

        # --- complexidade e novidade por tamanho e hist├│rico ---
        n_palavras = len(texto.split())
        if p.complexidade == 0.5:
            p.complexidade = min(1.0, 0.25 + n_palavras / 80)

        if historico:
            p.novidade = self._novidade(texto, historico)

        if not p.motivo:
            p.motivo = "conversa comum"
        return p

    def _extrair_expressao(self, texto: str) -> str | None:
        m = re.search(r"[\d\.,]+(?:\s*[\+\-\*/├ù├À\^]\s*[\d\.,]+)+", texto)
        return m.group(0) if m else None

    def _extrair_cidade(self, texto: str) -> str | None:
        m = CIDADE.search(texto)
        if m:
            return m.group(1)
        return None

    def _novidade(self, texto: str, historico: list[dict]) -> float:
        """Quanto o assunto difere do que j├í foi conversado."""
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


PROMPT_DECISOR = """Voc├¬ ├® um planejador. N├úo converse, n├úo responda ao usu├írio.
Analise a mensagem e devolva APENAS um JSON:

{{"intencao": "conversa|tarefa|emocional|sobre_si|saudacao",
 "precisa_memoria": true/false,
 "ferramentas": [{{"nome": "...", "args": {{}}}}],
 "carga_emocional": 0.0-1.0,
 "novidade": 0.0-1.0,
 "complexidade": 0.0-1.0}}

Ferramentas dispon├¡veis:
{ferramentas}

Mensagem: {mensagem}"""


def clientes_decisao(cfg_decisao):
    """Monta o par (cliente principal, cliente reserva) pra qualquer
    decis├úo pequena e estruturada -- usado pelo decisor de ferramentas E
    pelo loop de tarefa do Minecraft. Um lugar s├│ de prop├│sito: bug real
    j├í aconteceu por causa disso n├úo existir antes -- o loop de tarefa
    tinha sua pr├│pria c├│pia da l├│gica, desatualizada em rela├º├úo ao
    decisor principal, e continuou tentando um modelo local que j├í tinha
    sido descarregado depois que o Groq virou principal ali mas n├úo aqui.

    Sem Groq configurado, devolve (cliente_local, None) -- reserva nula,
    sem fallback nenhum, exatamente como j├í era antes do Groq existir.
    """
    from .llm import ClienteLLM
    if cfg_decisao.groq_ativo and cfg_decisao.groq_key:
        cfg_groq = replace(
            cfg_decisao, base_url=cfg_decisao.groq_url,
            api_key=cfg_decisao.groq_key, modelo=cfg_decisao.groq_modelo,
        )
        return ClienteLLM(cfg_groq), ClienteLLM(cfg_decisao)
    return ClienteLLM(cfg_decisao), None


def completar_com_reserva(principal, reserva, prompt: str) -> str:
    """Tenta o cliente principal; se falhar (exce├º├úo) OU devolver algo
    sem JSON reconhec├¡vel, tenta a reserva antes de desistir.

    Achado real: resposta "com sucesso" mas sem JSON dentro (Groq ├ás
    vezes devolve isso sob instabilidade) n├úo ├® uma exce├º├úo -- sem essa
    checagem extra, esse caso nunca acionava a reserva, s├│ o de conex├úo/
    erro HTTP acionava. Os dois motivos de falha agora levam ao mesmo
    lugar. Deixa a exce├º├úo do ├ÜLTIMO que falhar propagar, pra quem chama
    decidir o que fazer com isso.
    """
    def _tem_json(texto: str) -> bool:
        return bool(re.search(r"\{.*\}", texto, re.S))

    try:
        resultado = principal.completar(
            [{"role": "user", "content": prompt}], temperatura=0.0,
            max_tokens=principal.cfg.max_tokens,
        )
        if reserva is None or _tem_json(resultado):
            return resultado
        print(f"[decisao] principal respondeu sem JSON reconhec├¡vel, tentando reserva local")
    except Exception as e:
        if reserva is None:
            raise
        print(f"[decisao] principal falhou ({e}), tentando reserva local")

    return reserva.completar(
        [{"role": "user", "content": prompt}], temperatura=0.0,
        max_tokens=reserva.cfg.max_tokens,
    )


class DecisorPorLLM:
    """Decisor que usa um modelo. Cai para regras se o modelo falhar.

    `cliente` ├® o backend PRINCIPAL (Groq, se configurado -- ver
    DecisionConfig.groq_ativo). `cliente_reserva` ├® opcional -- quando
    dado, ├® tentado automaticamente se o principal falhar (erro de rede,
    limite de uso, etc.) antes de cair pra decis├úo por regra. Existe pra
    manter o LM Studio local funcionando como rede de seguran├ºa sem
    precisar de interven├º├úo manual quando o Groq estiver indispon├¡vel.
    """

    def __init__(self, cliente, registro, fallback: DecisorPorRegras | None = None,
                 cliente_reserva=None):
        self.cliente = cliente
        self.cliente_reserva = cliente_reserva
        self.registro = registro
        self.fallback = fallback or DecisorPorRegras()

    def _completar_com_reserva(self, prompt: str) -> str:
        return completar_com_reserva(self.cliente, self.cliente_reserva, prompt)

    def decidir(self, mensagem: str, historico: list[dict] | None = None) -> Plano:
        base = self.fallback.decidir(mensagem, historico)
        if any(f.get("nome") in {"robo_estado", "robo_olhar"}
               for f in base.ferramentas):
            base.motivo = "regra física prioritária"
            return base

        prompt = PROMPT_DECISOR.format(
            ferramentas=self.registro.descrever(), mensagem=mensagem
        )
        try:
            bruto = self._completar_com_reserva(prompt)
            m = re.search(r"\{.*\}", bruto, re.S)
            if not m:
                print(f"[decisao] resposta sem JSON, primeiros 200 chars: {bruto[:200]!r}")
                raise ValueError("sem JSON na resposta")
            d = json.loads(m.group(0))
        except Exception as e:
            # Decisor quebrado n├úo pode derrubar a conversa -- as regras
            # cobrem o caso e o usu├írio nem percebe. Log fica no console,
            # n├úo vai pro usu├írio -- s├│ o motivo curto abaixo ├® vis├¡vel.
            print(f"[decisao] decisor LLM falhou, caindo pra regras: {e}")
            plano = self.fallback.decidir(mensagem, historico)
            plano.motivo = "fallback: decisor LLM falhou"
            return plano

        # crise sempre ├® reavaliada por regra: ├® grave demais para depender
        # de o modelo ter classificado certo
        if base.intencao == "crise":
            return base

        ferramentas = [
            f for f in d.get("ferramentas", [])
            if isinstance(f, dict) and self.registro.get(f.get("nome", ""))
        ]
        # Pedidos físicos inequívocos não podem depender da interpretação do
        # LLM, nem ser confundidos com busca web ou visão da tela.
        for ferramenta_base in base.ferramentas:
            if (ferramenta_base.get("nome") in {"robo_olhar", "robo_estado"}
                    and not any(f.get("nome") == ferramenta_base.get("nome")
                                for f in ferramentas)):
                ferramentas.append(ferramenta_base)
        for f in ferramentas:
            if f.get("nome") == "buscar":
                args = f.setdefault("args", {})
                bruta = str(args.get("consulta", ""))
                args["consulta"] = extrair_consulta_busca(bruta, historico) or bruta[:200] or mensagem[:200]

        plano = Plano(
            intencao=d.get("intencao", "conversa"),
            precisa_memoria=bool(d.get("precisa_memoria", True)),
            precisa_visao=base.precisa_visao,
            precisa_jogo=base.precisa_jogo,
            precisa_ferramenta=bool(ferramentas),
            ferramentas=ferramentas,
            consulta_memoria=mensagem,
            prioridade="alta" if float(d.get("carga_emocional", 0)) > 0.7 else "normal",
            carga_emocional=float(d.get("carga_emocional", 0.0)),
            novidade=float(d.get("novidade", 0.5)),
            complexidade=float(d.get("complexidade", 0.5)),
            motivo="decisor LLM",
        )
        # BUG REAL: possivel_lacuna nunca era copiada do decisor de regra pro
        # plano final do LLM (o campo nem existe no JSON pedido ao modelo). Com
        # EVA_DECISION_LLM=1, a pesquisa de segundo plano da iniciativa parava
        # de disparar por completo, em sil├¬ncio.
        plano.possivel_lacuna = base.possivel_lacuna
        return plano