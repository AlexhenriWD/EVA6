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
from dataclasses import asdict, dataclass, field, replace


@dataclass
class Plano:
    """O que o Decision Engine devolve."""
    intencao: str = "conversa"
    precisa_memoria: bool = True
    precisa_ferramenta: bool = False
    precisa_visao: bool = False
    ferramentas: list[dict] = field(default_factory=list)  # [{"nome":..., "args":{...}}]
    consulta_memoria: str = ""
    prioridade: str = "normal"           # normal | alta
    carga_emocional: float = 0.0
    novidade: float = 0.5
    complexidade: float = 0.5
    guardar_memoria: bool = True
    motivo: str = ""

    # Sinal de "isso pode estar fora do que a EVA sabe" -- não dispara
    # ferramenta no turno atual (diferente de precisa_ferramenta/buscar,
    # que já busca IMEDIATO para responder). Este é para o orquestrador
    # pesquisar em SEGUNDO PLANO e guardar o achado para trazer à tona
    # depois, na iniciativa, via Consciencia.pesquisa_pronta(). Ver
    # LACUNA_CONHECIMENTO em decision.py.
    possivel_lacuna: str = ""

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

# ---------------------------------------------------------------- visão
#
# Não decide se a EVA VÊ algo -- isso é do SistemaVisual (vision/visao.py),
# que roda por fora, independente de decisão. Isto decide se, num turno
# em que existe uma cena capturada, vale a pena INJETAR ela no prompt.
#
# Sem este portão, contexto_visual entraria em TODO turno sempre que a
# visão estivesse ligada -- inclusive conversa sem nenhuma relação com a
# tela. É o mesmo risco de "narradora" da consciência (falar sobre tudo
# que muda vira insuportável), só que aplicado ao nível da conversa
# inteira em vez da fala espontânea: contexto visual perene polui o
# prompt e pode puxar a resposta pra comentar a tela sem vir ao caso.
#
# Referência DEÍTICA (aponta pra tela: "olha isso", "vê aqui") ou menção
# a artefato de tela (jogo, código, documento) -- os casos em que a
# pessoa está claramente falando do que está vendo. Falso negativo
# (não detectar quando devia) é preferível a falso positivo aqui: melhor
# a EVA ocasionalmente "esquecer" de olhar a tela do que ficar comentando
# a tela em toda resposta sem necessidade.
VISAO = re.compile(
    r"\b(v[êe] (isso|aqui|a[íi])|olha (isso|aqui|a tela)|olhando (isso|aqui)|"
    r"enxerg\w*|"                    # NOVO: "enxergar a tela", "consegue enxergar"
    r"\btela\b|"                     # NOVO: qualquer menção a "tela" já é sinal forte
    r"sua vis[ãa]o|"                 # NOVO: "problema com a sua visão"
    r"o que (eu )?(t[ôo]|estou) (fazendo|jogando|vendo|mostrando)|"
    r"(esse|essa) (jogo|c[óo]digo|documento|programa|janela)|"
    r"t[áa] vendo (isso|aqui)|consegue ver|voc[êe] (t[áa] |est[áa] )?vendo)\b",
    re.I,
)


def visao_relevante(texto: str) -> bool:
    """Se vale injetar contexto_visual neste turno.

    Função pública (não só uso interno de DecisorPorRegras) porque quem
    integra a visão (bridge_client.py) precisa decidir isso ANTES de
    chamar EVA.responder() -- contexto_visual é um parâmetro de entrada,
    não algo que o orquestrador busca sozinho. Mesma regra dos dois
    lados: aqui alimenta Plano.precisa_visao (visível em Resultado, pra
    quem quiser auditar a decisão depois), e bridge_client chama esta
    função direto no texto cru antes de decidir se passa a cena ou None.
    """
    return bool(VISAO.search(texto))

# Pedido de busca. Exige forma IMPERATIVA ou pergunta direta -- verbo no
# passado ("pesquisei tanto e não achei sentido") é relato, não pedido, e
# tratá-lo como busca faz a EVA sair procurando na web enquanto a pessoa
# estava desabafando.
# Pedido de busca. Exige forma IMPERATIVA, INFINITIVA (depois de modal:
# "pode/consegue/dá pra pesquisar") ou pergunta direta -- verbo no passado
# ("pesquisei tanto e não achei sentido") é relato, não pedido, e tratá-lo
# como busca faz a EVA sair procurando na web enquanto a pessoa estava
# desabafando.
#
# BUG REAL JÁ VISTO: "pesquis[ae]\b" casa "pesquisa"/"pesquise" mas NÃO
# "pesquisar" -- depois de "pesquisa" vem "r" colado, sem fronteira de
# palavra ali, então o \b falha. Isso faz "você pode pesquisar pra mim"
# (forma mais natural e educada que o imperativo seco "pesquisa isso")
# nunca disparar a ferramenta -- silenciosamente, sem erro nenhum. A EVA
# respondia do que já sabia (desatualizado) e não tinha como perceber
# que a busca nunca rodou. Por isso agora cada verbo cobre explicitamente
# a forma no infinitivo também: pesquis(a|e|ar), não só pesquis[ae].
# BUG REAL JA VISTO (2a vez, conjugacao diferente): "pesquisar[ae]" so
# cobria imperativo e infinitivo. "eu gostaria que você pesquisasse"
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
    r"me acha|acha a[íi]|d[áa] uma olhada|v[êe] a[íi])\b"
    r"|\b(quem [ée]|o que [ée] o|quanto custa|qual o pre[çc]o|"
    r"[úu]ltimas not[íi]cias|not[íi]cias de hoje)\b)", re.I
)

# Formas no passado que parecem busca mas são relato pessoal
BUSCA_RELATO = re.compile(
    r"\b(pesquisei|busquei|procurei|andei pesquisando|tentei achar|"
    r"j[áa] pesquisei|j[áa] procurei)\b", re.I
)

# ------------------------------------------------- consulta de busca real
#
# BUSCA só detecta QUE existe pedido de busca -- não separa o pedido do
# assunto. "pesquisa mais sobre isso" batia em BUSCA e a consulta virava
# essa frase literal, mandada pro SearXNG palavra por palavra. Motor de
# busca não resolve "isso" (não tem o resto da conversa), então voltava
# lixo ou nada -- e por fora parecia a EVA "se recusando a pesquisar",
# quando na verdade pesquisou a frase errada.
_GATILHO_BUSCA = re.compile(
    r"^\s*(voc[êe]\s+)?(pode|poderia|consegue|conseguiria|d[áa]\s+pra|"
    r"d[áa]\s+para|gostaria\s+que\s+voc[êe])?\s*"
    r"(pesquis(a|e|ar|asse|aria)|busc(a|e|ar|asse|aria)|procur(a|e|ar|asse|aria)|"
    r"me\s+ach(a|ar)|ach(a|ar)\s+a[íi]|d[áa]\s+uma\s+olhada(\s+em)?|v[êe]\s+a[íi])\s*",
    re.I,
)

_SO_ENCHIMENTO = re.compile(
    r"^(sobre\s+|em\s+|no\s+|na\s+)?"
    r"(isso|isto|aquilo|esse\s+assunto|essa\s+coisa|a[íi]|"
    r"mais(\s+(sobre\s+)?(isso|isto|aquilo))?|pra\s+mim|por\s+favor|)"
    r"[\s.,!?]*$",
    re.I,
)


def extrair_consulta_busca(texto: str, historico: list[dict] | None = None) -> str | None:
    """Assunto de um pedido de busca -- nunca o pedido cru.

    Duas etapas: tira o verbo-gatilho e o modal que vem junto; se o que
    sobra é só enchimento sem assunto (pronome, "mais sobre", "por
    favor"), tenta resolver olhando a ÚLTIMA mensagem do usuário no
    histórico -- é o caso mais comum de pronome apontando pra fora da
    frase atual ("o que rolou de novo?" ... "pesquisa mais sobre isso").

    Devolve None quando não há assunto extraível nem no histórico -- quem
    chama decide o que fazer (busca explícita cai pro texto original como
    último recurso; lacuna em segundo plano simplesmente desiste).
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
# Diferente de BUSCA (pedido explícito, busca AGORA para responder), isto
# é sobre tema que SOA como fato-que-muda mesmo sem pedido de busca --
# "o que você acha do mercado de IA agora", "fale sobre a Fórmula 1 esse
# ano". A EVA responde do que já sabe (treino), que pode estar
# desatualizado, e o usuário não tem como perceber isso.
#
# Não é substituto de julgamento -- é rede grosseira e rápida (regex, zero
# custo), porque não pode ter o delay de uma chamada de LLM em todo turno.
# Ela vai ERRAR: vai marcar coisa que não precisava, vai perder coisa
# sutil. É aceitável porque o efeito de marcar errado é pesquisa em
# segundo plano que talvez nunca seja usada (TTL expira, ver
# consciousness.py) -- não é dito ao usuário, não interrompe nada.
#
# Dois sinais, and não or: sozinho, "mercado" ou "atual" aparecem demais
# em conversa comum. Junto -- tema que muda (mercado, tecnologia, versão,
# notícia, preço) E marcador de tempo presente/recente (agora, hoje, esse
# ano, atualmente) -- é bem mais específico do que exige checagem.
LACUNA_TEMA = re.compile(
    r"\b(mercado|tecnologia|ia\b|intelig[êe]ncia artificial|not[íi]cia|"
    r"pre[çc]o|vers[ãa]o|lan[çc]amento|eleiç[ãa]o|governo|guerra|"
    r"empresa|startup|criptomoeda|bolsa|a[çc][õo]es)\b", re.I
)
LACUNA_TEMPORAL = re.compile(
    r"\b(agora|hoje|atualmente|esse ano|este ano|ultimamente|"
    r"recentemente|nos [úu]ltimos|em 202\d)\b", re.I
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
        # Setado ANTES de qualquer retorno antecipado (crise, saudação,
        # sobre_si) -- referência à tela pode aparecer em mensagem curta
        # ("olha isso") que de outra forma sairia num desses atalhos sem
        # passar pelo resto da função.
        p.precisa_visao = visao_relevante(texto)

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
        busca_ja_acionada = False
        if BUSCA.search(texto) and p.carga_emocional < 0.5 and not BUSCA_RELATO.search(texto):
            consulta = extrair_consulta_busca(texto, historico)
            if consulta is None:
                consulta = texto[:200]
                print(f"[decisao] busca sem assunto extraível, usando texto cru: {texto[:80]!r}")
            ferramentas.append({"nome": "buscar", "args": {"consulta": consulta}})
            busca_ja_acionada = True

        # Lacuna de conhecimento: não busca AGORA (não atrasa a resposta),
        # só marca para o orquestrador pesquisar em segundo plano. Só marca
        # se a busca imediata não disparou -- senão a mesma pesquisa
        # aconteceria duas vezes por motivos diferentes.
        if (not busca_ja_acionada and p.carga_emocional < 0.5
                and LACUNA_TEMA.search(texto) and LACUNA_TEMPORAL.search(texto)):
            # Oportunista: se não achou assunto de verdade, não pesquisa nada em
            # vez de gastar cota do Brave numa frase-gatilho sem conteúdo.
            p.possivel_lacuna = extrair_consulta_busca(texto, historico) or texto[:200]

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


def clientes_decisao(cfg_decisao):
    """Monta o par (cliente principal, cliente reserva) pra qualquer
    decisão pequena e estruturada -- usado pelo decisor de ferramentas E
    pelo loop de tarefa do Minecraft. Um lugar só de propósito: bug real
    já aconteceu por causa disso não existir antes -- o loop de tarefa
    tinha sua própria cópia da lógica, desatualizada em relação ao
    decisor principal, e continuou tentando um modelo local que já tinha
    sido descarregado depois que o Groq virou principal ali mas não aqui.

    Sem Groq configurado, devolve (cliente_local, None) -- reserva nula,
    sem fallback nenhum, exatamente como já era antes do Groq existir.
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
    """Tenta o cliente principal; se falhar (exceção) OU devolver algo
    sem JSON reconhecível, tenta a reserva antes de desistir.

    Achado real: resposta "com sucesso" mas sem JSON dentro (Groq às
    vezes devolve isso sob instabilidade) não é uma exceção -- sem essa
    checagem extra, esse caso nunca acionava a reserva, só o de conexão/
    erro HTTP acionava. Os dois motivos de falha agora levam ao mesmo
    lugar. Deixa a exceção do ÚLTIMO que falhar propagar, pra quem chama
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
        print(f"[decisao] principal respondeu sem JSON reconhecível, tentando reserva local")
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

    `cliente` é o backend PRINCIPAL (Groq, se configurado -- ver
    DecisionConfig.groq_ativo). `cliente_reserva` é opcional -- quando
    dado, é tentado automaticamente se o principal falhar (erro de rede,
    limite de uso, etc.) antes de cair pra decisão por regra. Existe pra
    manter o LM Studio local funcionando como rede de segurança sem
    precisar de intervenção manual quando o Groq estiver indisponível.
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
            # Decisor quebrado não pode derrubar a conversa -- as regras
            # cobrem o caso e o usuário nem percebe. Log fica no console,
            # não vai pro usuário -- só o motivo curto abaixo é visível.
            print(f"[decisao] decisor LLM falhou, caindo pra regras: {e}")
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
        for f in ferramentas:
            if f.get("nome") == "buscar":
                args = f.setdefault("args", {})
                bruta = str(args.get("consulta", ""))
                args["consulta"] = extrair_consulta_busca(bruta, historico) or bruta[:200] or mensagem[:200]

        plano = Plano(
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
        # BUG REAL: possivel_lacuna nunca era copiada do decisor de regra pro
        # plano final do LLM (o campo nem existe no JSON pedido ao modelo). Com
        # EVA_DECISION_LLM=1, a pesquisa de segundo plano da iniciativa parava
        # de disparar por completo, em silêncio.
        plano.possivel_lacuna = base.possivel_lacuna
        return plano