"""
Consciência -- decide quando a EVA fala sem ser chamada.

O PRINCÍPIO
-----------
Duas metades, e a separação é o que faz isso funcionar:

    produzir impulso   "tem algo que eu poderia dizer"
    abrir o portão     "vale dizer AGORA"

O modelo de conversa não participa de nenhuma das duas. Se você perguntar a
ele "devo falar agora?", ele diz sim -- produzir resposta é a única coisa
que ele sabe fazer. Ele só entra depois que o portão já abriu, e aí escreve
o que vai ser dito.

O padrão é o silêncio. Fala não solicitada é irritante por definição, e o
custo dos dois erros é assimétrico: ficar quieta quando poderia falar passa
despercebido; falar quando não devia estraga a call. O portão erra para o
lado de não falar.

DE ONDE VÊM OS IMPULSOS
-----------------------
Em ordem de qualidade:

  fio      algo ficou pendurado numa conversa anterior. É o melhor porque
           é específico e não precisa ser inventado -- ela retoma, não
           puxa assunto do nada. Gente funciona assim.
  visual   mudou algo na tela (fase 3, o gancho já está aqui)
  pesquisa ela foi buscar algo em segundo plano e voltou com conteúdo
  vazio    silêncio longo e curiosidade alta. O mais fraco, e de propósito:
           quase sempre barrado pelo limiar. Existe para os casos em que
           tudo o mais está a favor.

Impulso EXPIRA. Voltar dois minutos depois com um assunto que já morreu é
pior que não voltar.

O ESTADO É O ORÇAMENTO
----------------------
`state.py` hoje entra no prompt como número e torce para o modelo reagir --
sem jeito de verificar. Aqui ele finalmente faz algo observável: curiosidade
e energia baixam o limiar, estresse sobe. `curiosidade: 0.91` passa a ter
consequência mensurável.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass, field

# Força de cada tipo de impulso. Não é ajuste fino: reflete o quanto o
# conteúdo é específico. Retomar algo que a pessoa disse vale mais que
# comentar que a tela mudou, que vale mais que falar por falar.
FORCA_PADRAO = {
    # Ordem calibrada com a preferência declarada do Alex (24/08/2026),
    # que estava QUASE INVERTIDA em relação aos valores antigos: "fio"
    # era o mais forte (0.70) e o mais indesejado; "vazio" era o segundo
    # mais desejado e o mais fraco (0.35) -- fraco a ponto de nunca passar
    # do limiar, o que se via em todo log como "impulso fraco (vazio)".
    "pesquisa": 0.72,      # 1ª preferência: trazer algo novo do mundo
    "visual_robo": 0.68,   # câmera do robô -- ver nota abaixo
    "corporal": 0.65,      # segurança do robô: urgência real, não gosto.
                           # Não entra na ordem de preferência de propósito.
    "jogo": 0.66,          # Minecraft: dano relevante, morte, conexão.
                           # Fala de jogador no chat do jogo vem com força
                           # explícita bem maior (FORCA_CHAT_JOGO = 0.80 em
                           # minecraft_tools) -- é uma pessoa se dirigindo
                           # a ela, não um incidente do mundo.
    "vazio": 0.62,         # 2ª preferência: puxar conversa no silêncio.
                           # Viável agora, mas ainda depende de silêncio
                           # longo + curiosidade alta pra ser escolhido.
    "visual": 0.58,        # 3ª: tela do PC
    "iniciativa": 0.55,
    "fio": 0.50,           # 4ª: retomar assunto pendente
}

# visual_robo > visual de propósito. Quando ela está operando o robô, o
# que a câmera dele mostra É o assunto -- é a única coisa que só ela está
# vendo, e comentar aquilo é a diferença entre operar um robô e narrar uma
# captura de tela. A tela do PC, por contraste, a pessoa também está vendo:
# comentar é redundante com mais frequência.
#
# Não há checagem de "robô conectado" aqui, e não precisa: sem conexão não
# chega quadro, sem quadro o SistemaVisualRobo não emite evento. A força
# alta só existe quando há robô de fato.

# Quanto tempo cada tipo continua valendo depois de criado.
TTL_PADRAO = {
    "fio": 900.0,      # 15 min -- fio é durável, o assunto não azeda rápido
    "visual": 60.0,    # a tela já mudou de novo
    "visual_robo": 45.0,  # cena do robô muda mais rápido que tela: ele se
                          # move, e comentar o que já saiu do quadro é pior
                          # que ficar quieto
    "corporal": 90.0,  # estado físico muda rápido -- comentar "acabei de
                       # tomar um susto" 3min depois já ficou esquisito
    "jogo": 75.0,      # responder a algo do jogo 2min depois já perdeu
                       # o sentido -- ela já se moveu, o mob já sumiu
    "iniciativa": 400.0,
    "pesquisa": 180.0,
    "vazio": 60.0,
}


@dataclass
class Impulso:
    tipo: str
    conteudo: str              # descrição em português do que ela quer dizer
    forca: float = 0.5
    criado_em: float = field(default_factory=time.time)
    ttl: float = 120.0
    origem: str | None = None  # user_id de quem gerou, quando aplicável

    def expirado(self, agora: float | None = None) -> bool:
        # `if agora is None`, nunca `agora or time.time()`: 0.0 é falsy, e o
        # simulador roda em tempo virtual começando em zero. Em produção o
        # timestamp nunca é 0, então o bug só aparecia no teste -- que é
        # justamente onde o comportamento é calibrado.
        if agora is None:
            agora = time.time()
        return agora - self.criado_em > self.ttl

    def __str__(self) -> str:
        return f"[{self.tipo} {self.forca:.2f}] {self.conteudo}"


def criar_impulso(tipo: str, conteudo: str, forca: float | None = None,
                  origem: str | None = None) -> Impulso:
    return Impulso(
        tipo=tipo,
        conteudo=conteudo.strip(),
        forca=FORCA_PADRAO.get(tipo, 0.5) if forca is None else forca,
        ttl=TTL_PADRAO.get(tipo, 120.0),
        origem=origem,
    )


# --------------------------------------------------------------- fios

# Marcadores de intenção/plano. Quem diz "vou refazer o treino" deixa algo
# em aberto que dá para retomar depois -- e é exatamente o tipo de coisa que
# uma pessoa lembraria de perguntar.
PLANO = re.compile(
    r"\b(vou|pretendo|tô pensando em|to pensando em|quero|planejo|"
    r"semana que vem|amanhã|amanha|depois eu|mais tarde|preciso)\s+([^.!?;,]{6,70})",
    re.I,
)

# Coisa nova mencionada de passagem. O grupo capturado vira o assunto.
MENCAO = re.compile(
    r"\b(comecei|começei|terminei|acabei de|descobri|comprei|li sobre|"
    r"conheci|entrei n[oa]|saí d[oa]|sai d[oa])\s+([^.!?;,]{4,60})",
    re.I,
)

# Não vira fio: se for um pedido direto, ela já está respondendo agora.
NAO_E_FIO = re.compile(r"\?\s*$|^(me ajuda|como|qual|quando|onde|quem|por que)", re.I)


@dataclass
class Fio:
    """Algo que ficou pendurado e dá para retomar."""
    assunto: str
    usuario: str
    criado_em: float = field(default_factory=time.time)
    usado: bool = False

    def idade_horas(self) -> float:
        return (time.time() - self.criado_em) / 3600


def extrair_fios(mensagem: str, usuario: str, plano=None) -> list[Fio]:
    """Tira fios de uma mensagem do usuário, por regra.

    Por regra e não por LLM porque isso roda em TODO turno: uma chamada de
    modelo aqui acrescentaria segundos a cada resposta para produzir algo
    que, na maioria dos turnos, não existe.

    Uma regra que eu NÃO implementei, e o motivo: "ela perguntou algo e a
    pessoa não respondeu" parece o melhor fio de todos, mas detectar se uma
    resposta responde à pergunta é caro e erra muito. Fio errado é pior que
    fio nenhum -- ela retoma algo que já foi resolvido e parece desatenta.
    """
    if NAO_E_FIO.search(mensagem.strip()):
        return []

    achados: list[Fio] = []
    vistos: set[str] = set()

    for padrao in (MENCAO, PLANO):
        for m in padrao.finditer(mensagem):
            assunto = m.group(0).strip().rstrip(".,;")
            chave = assunto.lower()
            if chave in vistos or len(assunto) < 8:
                continue
            vistos.add(chave)
            achados.append(Fio(assunto=assunto, usuario=usuario))

    return achados[:2]  # dois por turno já é generoso


# ------------------------------------------------------------- portão


@dataclass
class Veredito:
    passou: bool
    motivo: str
    limiar: float = 0.0
    forca: float = 0.0
    impulso: Impulso | None = None
    # True só nos dois caminhos que de fato calculam limiar/força (impulso
    # fraco, e a aprovação final) -- todos os outros retornos antecipados
    # (ocupada, sem impulso, conversa viva, etc) usam os defaults 0.0/0.0
    # do dataclass, e SEM esta flag o __str__ imprimia "força 0.00 vs
    # limiar 0.00" pra esses casos como se fosse uma comparação real que
    # aconteceu -- confuso de ler no log, parecia sempre a mesma conta
    # dando quase-zero quando na verdade a conta nunca rodou.
    avaliado: bool = False

    def __str__(self) -> str:
        marca = "FALA" if self.passou else "cala"
        if self.avaliado:
            return f"{marca}: {self.motivo} (força {self.forca:.2f} vs limiar {self.limiar:.2f})"
        return f"{marca}: {self.motivo}"


class PortaoFala:
    """Decide se um impulso vira voz.

    Todas as barreiras são verificadas antes do limiar, e nessa ordem: as
    de tempo são baratas e absolutas, o limiar é caro de raciocinar. Se ela
    falou há dez segundos, não importa quão bom seja o impulso.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def avaliar(
        self,
        impulso: Impulso | None,
        *,
        silencio: float,              # segundos desde alguém falar
        desde_fala_dela: float,       # segundos desde ELA falar
        falas_sem_resposta: int,
        ocupada: bool,
        estado,
        agora: float | None = None,
    ) -> Veredito:
        if agora is None:
            agora = time.time()
        c = self.cfg

        if not c.ativa:
            return Veredito(False, "consciência desligada")
        if ocupada:
            return Veredito(False, "já está falando ou processando")
        if impulso is None:
            return Veredito(False, "nenhum impulso na fila")
        if impulso.expirado(agora):
            return Veredito(False, f"impulso expirado ({impulso.tipo})")

        if silencio < c.silencio_minimo:
            return Veredito(False,
                            f"conversa viva ({silencio:.0f}s < {c.silencio_minimo:.0f}s)")
        if desde_fala_dela < c.cooldown_fala:
            return Veredito(False,
                            f"falou há pouco ({desde_fala_dela:.0f}s < {c.cooldown_fala:.0f}s)")

        limiar = self.limiar(estado, falas_sem_resposta)
        if impulso.forca < limiar:
            return Veredito(False, f"impulso fraco ({impulso.tipo})",
                            limiar, impulso.forca, impulso, avaliado=True)

        return Veredito(True, f"impulso {impulso.tipo}", limiar, impulso.forca,
                        impulso, avaliado=True)

    def limiar(self, estado, falas_sem_resposta: int = 0) -> float:
        """Quanta força um impulso precisa ter para virar fala.

        Curiosidade e energia altas baixam; estresse sobe. Os coeficientes
        são deliberadamente pequenos: o estado modula, não decide. Se ele
        pudesse zerar o limiar sozinho, um pico de curiosidade transformaria
        a EVA em tagarela.

        A escalada por fala sem resposta é a barreira mais importante daqui.
        Falar duas vezes seguidas e ninguém responder significa que ela leu
        a call errado -- insistir uma terceira vez é o comportamento que faz
        as pessoas desligarem o bot.
        """
        c = self.cfg
        v = c.limiar_base
        v -= 0.25 * (getattr(estado, "curiosidade", 0.5) - 0.5)
        v -= 0.15 * (getattr(estado, "energia", 0.5) - 0.5)
        v += 0.35 * getattr(estado, "estresse", 0.0)
        v += 0.25 * max(0, falas_sem_resposta)
        return max(0.25, min(0.99, v))


# --------------------------------------------------------- consciência


# Tipos cujo conteúdo é gerado (resumo de busca, fala de preenchimento)
# e portanto pode ser vazio de substância. Os outros descrevem um evento
# concreto que aconteceu -- a tela mudou, o robô recusou um comando -- e
# não precisam ser validados: o fato já é a substância.
TIPOS_VALIDADOS = {"pesquisa", "iniciativa"}

# "vazio" ficou DE FORA, e isso é correção de um erro de desenho, não
# afrouxamento: o conteúdo dele é o marcador "puxar assunto", não a fala.
# O texto real só nasce depois, no modelo de conversa. Validar substância
# de um placeholder não mede nada -- e no simulador isso disparava uma
# chamada ao Groq A CADA TICK (5s), estourando o rate limit de 8000 TPM
# em minutos. Se um dia o "vazio" passar a carregar texto de verdade, ele
# volta pra cá.

# Sinal barato de concretude: nome próprio, número, data, sigla. Um
# impulso de pesquisa sem NENHUM deles é quase sempre generalidade vazia
# ("preparar-se para os desafios do futuro exige visão de longo prazo" --
# caso real de log, 23/08/2026, que virou fala).
def _tem_concreto(texto: str) -> bool:
    """Número, sigla ou nome próprio -- sinal barato de que há algo real.

    Cuidado que custou um falso-negativo no teste: maiúscula de INÍCIO DE
    FRASE não é nome próprio. "Preparar-se para os desafios do futuro
    exige visão de longo prazo" (lixo real de log) passava como concreto
    só por causa do "P" inicial. Por isso a busca por nome próprio ignora
    a primeira palavra de cada frase.
    """
    if re.search(r"\b\d+\b", texto):          # número
        return True
    if re.search(r"\b[A-Z]{2,}\b", texto):     # sigla
        return True
    for frase in re.split(r"[.!?]\s+|^", texto):
        palavras = frase.split()
        for palavra in palavras[1:]:           # pula a primeira
            if re.match(r"^[A-ZÁÂÃÉÊÍÓÔÕÚÇ][a-záâãéêíóôõúç]{2,}", palavra):
                return True
    return False

# Abertura de generalidade. Não é lista de palavra proibida: é o formato
# "afirmação universal sem sujeito" que sai de resumo ruim.
_GENERICO = re.compile(
    r"\b(exige|requer|é fundamental|é importante|é essencial|"
    r"pode ajudar|costuma ser|em geral|de modo geral)\b", re.I)


class PortaoSubstancia:
    """Segundo portão: o primeiro decide QUANDO falar, este decide SE há
    o que dizer.

    Por que existe: o PortaoFala só olha tempo, estado e força -- nada
    disso enxerga o CONTEÚDO. Um resumo alucinado do que a pessoa disse
    virava busca, virava impulso com força de pesquisa, passava no portão
    e saía pela boca dela. O portão estava certo e a fala era lixo, porque
    ninguém tinha perguntado se havia algo real ali.

    Duas camadas, nessa ordem, e a ordem importa: regra primeiro porque é
    grátis e pega o caso óbvio; LLM só no que sobrou.

    O LLM aqui é o GROQ, não o modelo local, e isso é decisão de
    arquitetura, não conveniência: validar substância é chamada curta e
    sem estado, e o servidor local (8082) já disputa GPU com a visão e com
    o modelo de conversa -- medido em call real, visão sozinha leva 6s e
    30s quando concorre. Mandar isso pra fora da GPU custa latência de
    rede e não custa nada do orçamento que aperta.
    """

    _PROMPT = (
        "Você valida se um comentário tem conteúdo real antes de uma IA "
        "dizê-lo em voz alta numa conversa.\n\n"
        "REJEITE se for: generalidade sem sujeito, conselho motivacional, "
        "paráfrase do que a pessoa acabou de dizer, ou afirmação vaga que "
        "serviria pra qualquer assunto.\n"
        "ACEITE se trouxer: fato específico, dado, nome, evento concreto, "
        "ou observação particular sobre algo que está acontecendo.\n\n"
        "Última fala da pessoa: {ultima}\n"
        "Comentário a validar: {conteudo}\n\n"
        "Responda APENAS com um JSON: {{\"aceita\": true|false, "
        "\"motivo\": \"até 8 palavras\"}}"
    )

    # Quanto tempo um veredito continua valendo pro MESMO texto. Sem
    # isso, um impulso que fica na fila é revalidado a cada tick -- mesmo
    # texto, mesma resposta, uma chamada de rede cada vez.
    TTL_CACHE = 120.0

    def __init__(self, cfg):
        self.cfg = cfg
        self._principal = None
        self._reserva = None
        self._tentou_montar = False
        self._cache: dict[str, tuple[float, bool, str]] = {}

    def _clientes(self):
        # Montagem preguiçosa: sem Groq configurado o portão nunca chama
        # LLM, e não faz sentido construir cliente no __init__ por isso.
        if not self._tentou_montar:
            self._tentou_montar = True
            try:
                from .decision import clientes_decisao
                self._principal, self._reserva = clientes_decisao(
                    self.cfg.decisao)
            except Exception as e:  # noqa: BLE001
                print(f"[substancia] sem validador LLM: {e}")
        return self._principal, self._reserva

    def validar(self, impulso, ultima_fala: str = "") -> tuple[bool, str]:
        """(aceita, motivo). Nunca levanta: falha vira aceite.

        Falhar pra ACEITE e não pra rejeição é escolha deliberada. Este
        portão existe pra cortar lixo, não pra virar mais um jeito da EVA
        emudecer: se o Groq cair, o comportamento volta a ser exatamente o
        de antes deste portão existir, e não 'ela parou de falar sozinha'.
        """
        if impulso is None:
            return False, "sem impulso"
        if impulso.tipo == "vazio":
            # Marcador, não fala: o texto real nasce depois no modelo.
            return True, "vazio: texto ainda não existe"
        if impulso.tipo not in TIPOS_VALIDADOS:
            return True, f"{impulso.tipo} descreve evento concreto"

        texto = (impulso.conteudo or "").strip()
        if len(texto) < 12:
            return False, "conteúdo curto demais"

        agora = time.time()
        em_cache = self._cache.get(texto)
        if em_cache and agora - em_cache[0] < self.TTL_CACHE:
            return em_cache[1], em_cache[2]
        # limpeza preguiçosa -- o dict nunca passa de algumas dezenas
        if len(self._cache) > 64:
            self._cache = {k: v for k, v in self._cache.items()
                           if agora - v[0] < self.TTL_CACHE}

        # --- camada 1: regra, grátis ---
        # A regra só REJEITA sozinha quando há sinal POSITIVO de vazio
        # (fórmula de generalidade). Ausência de nome próprio/número não
        # basta: "saiu uma técnica nova de impressão 3D em titânio" é
        # concreto e não casa com nenhum dos padrões. Falso-negativo aqui
        # é pior que falso-positivo, porque a camada 2 ainda filtra --
        # rejeitar na camada 1 é definitivo.
        if _GENERICO.search(texto) and not _tem_concreto(texto):
            return self._lembrar(texto, False, "generalidade sem sujeito")
        if ultima_fala and self._e_eco(texto, ultima_fala):
            return self._lembrar(texto, False, "eco do que a pessoa disse")

        # --- camada 2: LLM (Groq), só no que sobrou ---
        principal, reserva = self._clientes()
        if principal is None:
            return True, "sem validador -- passou pela regra"
        try:
            from .decision import completar_com_reserva
            bruto = completar_com_reserva(
                principal, reserva,
                self._PROMPT.format(ultima=ultima_fala or "(nada ainda)",
                                    conteudo=texto))
            dados = json.loads(_extrair_json(bruto))
            if not dados.get("aceita", True):
                return self._lembrar(
                    texto, False, str(dados.get("motivo", "rejeitado"))[:60])
            return self._lembrar(texto, True, "validado")
        except Exception as e:  # noqa: BLE001
            print(f"[substancia] validador falhou, aceitando: {e}")
            return True, "validador indisponível"

    def _lembrar(self, texto: str, ok: bool, motivo: str) -> tuple[bool, str]:
        """Guarda o veredito e devolve. Só resultados DEFINITIVOS entram --
        falha de rede não vira cache, senão um Groq fora do ar por 3s
        deixaria tudo liberado pelos 120s seguintes."""
        self._cache[texto] = (time.time(), ok, motivo)
        return ok, motivo

    @staticmethod
    def _e_eco(texto: str, ultima: str) -> bool:
        a = {p for p in texto.lower().split() if len(p) > 3}
        b = {p for p in ultima.lower().split() if len(p) > 3}
        if not a or not b:
            return False
        return len(a & b) / len(a) > 0.65


def _extrair_json(bruto: str) -> str:
    """Groq às vezes embrulha em ```json -- mesmo tratamento do decisor."""
    t = (bruto or "").strip()
    if "```" in t:
        t = re.sub(r"```(?:json)?", "", t).strip()
    i, f = t.find("{"), t.rfind("}")
    return t[i:f + 1] if i >= 0 and f > i else t


class Consciencia:
    """Uma instância por canal de voz.

    Não roda laço próprio: quem chama faz `tick()` no ritmo que quiser. Isso
    deixa o comportamento inteiro testável sem asyncio, sem call e sem GPU --
    o simulador em ferramentas/simular_portao.py roda milhares de cenários
    em segundos.
    """

    def __init__(self, cfg, canal: str = ""):
        self.cfg = cfg
        self.canal = canal
        self.portao = PortaoFala(cfg.consciencia)
        # cfg inteiro (nao cfg.consciencia): o portao de substancia precisa
        # de cfg.decisao pra montar o cliente Groq.
        self.substancia = PortaoSubstancia(cfg)

        self.fila: deque[Impulso] = deque(maxlen=12)
        # user_id -> nome. Sem isso o impulso sai como "retomar o que
        # 481923... mencionou", o id cru vai para o prompt e o modelo às
        # vezes fala o número em voz alta.
        self.nomes: dict[str, str] = {}
        self.fios: deque[Fio] = deque(maxlen=cfg.consciencia.max_fios)

        agora = time.time()
        self.ultima_fala_alguem = agora
        self.ultima_fala_dela = agora
        self.ultimo_falante: str | None = None
        # texto da ultima fala da pessoa -- so pro teste de eco do
        # PortaoSubstancia. Nao entra em prompt nenhum.
        self._ultima_fala_texto: str = ""
        # Ver a trava em tick(): segura a proxima tentativa depois de um
        # impulso barrado por falta de substancia.
        self._bloqueado_ate: float = 0.0
        self.falas_sem_resposta = 0
        self.ocupada = False

        self.ultimo_veredito: Veredito | None = None

    # ---------------------------------------------------------- eventos

    def registrar_nome(self, usuario: str, nome: str | None) -> None:
        if nome:
            self.nomes[str(usuario)] = nome

    def audio_detectado(self) -> None:
        """Reseta só o relógio de silêncio, assim que chega áudio
        não-silencioso -- ANTES da transcrição terminar.

        BUG REAL encontrado em log de produção: `ultima_fala_alguem` só
        era atualizado aqui, dentro de `alguem_falou()`, que só roda
        DEPOIS do STT terminar (~1-2s de `await`, ponto onde o laço de
        `tick()` roda por baixo, mesmo event loop). Nesse intervalo, o
        portão via um silêncio já defasado -- se a pausa ANTES da pessoa
        recomeçar a falar já tivesse passado do limiar, um impulso de
        proatividade podia ser aprovado bem no meio dela começando a
        falar de novo, pulando na frente da resposta de verdade (que só
        chega depois, quando o STT termina). Log real: "eva espontânea"
        divagando sobre um assunto velho, e a resposta certa pro que a
        pessoa disse chegando SÓ DEPOIS, logo em seguida.

        Só mexe no relógio -- fila de impulsos e fios continuam sendo
        limpos/coletados em alguem_falou(), quando já se sabe que era
        fala de verdade (não ruído) e o texto já existe."""
        self.ultima_fala_alguem = time.time()

    def alguem_falou(self, usuario: str, mensagem: str = "", plano=None,
                     nome: str | None = None) -> None:
        """Registra fala humana. Zera a escalada e colhe fios."""
        self.ultima_fala_alguem = time.time()
        self.ultimo_falante = usuario
        self._ultima_fala_texto = mensagem or ""
        self.registrar_nome(usuario, nome)
        self.falas_sem_resposta = 0

        # Impulso pendente vira lixo quando a conversa retoma: ele foi
        # gerado para preencher um silêncio que acabou de terminar.
        self.fila.clear()

        if mensagem:
            for fio in extrair_fios(mensagem, usuario, plano):
                if not self._fio_repetido(fio):
                    self.fios.append(fio)

    def ela_falou(self, espontanea: bool = False) -> None:
        self.ultima_fala_dela = time.time()
        if espontanea:
            self.falas_sem_resposta += 1

    def adicionar(self, impulso: Impulso) -> None:
        self.fila.append(impulso)

    def evento_visual(self, descricao: str) -> None:
        """Tela do PC. A visão chama aqui quando algo muda de verdade."""
        self.adicionar(criar_impulso("visual", descricao))

    def evento_visual_robo(self, descricao: str) -> None:
        """Câmera do robô -- tipo SEPARADO de `visual`, não é detalhe.

        Antes os dois laços (_laco_visao e _laco_visao_robo) chamavam
        `evento_visual`, então os dois viravam o mesmo impulso com a mesma
        força e a EVA não tinha como preferir um. Sem tipo próprio não há
        como dar peso maior ao que ela vê pelo robô -- ver FORCA_PADRAO.
        """
        self.adicionar(criar_impulso("visual_robo", descricao))

    def evento_corporal(self, descricao: str) -> None:
        """Transição de segurança do corpo físico (entrou/saiu de
        emergency stop, watchdog caiu) ou recusa de comando de
        movimento -- NUNCA movimento rotineiro bem-sucedido. Quem filtra
        isso é robot_tools._detectar_transicao_seguranca/_descrever_recusa,
        não aqui -- este método só empacota o que já chegou filtrado."""
        self.adicionar(criar_impulso("corporal", descricao))

    def evento_jogo(self, descricao: str, forca: float | None = None) -> None:
        """Algo que aconteceu no Minecraft e vale comentar: jogador falou
        no chat do jogo, dano relevante, morte, entrar/sair do servidor.
        Quem filtra é minecraft_tools (_ao_chat/_ao_evento) -- este método
        só empacota o que já chegou filtrado, mesmo contrato de
        evento_corporal. `forca=None` usa o padrão do tipo."""
        self.adicionar(criar_impulso("jogo", descricao, forca=forca))

    def sugestao_assunto(self, texto: str) -> None:
        """Adiciona assunto real sugerido a partir da memória da pessoa."""
        self.adicionar(criar_impulso("iniciativa", texto))

    def pesquisa_pronta(self, resumo: str) -> None:
        self.adicionar(criar_impulso("pesquisa", resumo))

    def _fio_repetido(self, novo: Fio) -> bool:
        alvo = set(novo.assunto.lower().split())
        for f in self.fios:
            outro = set(f.assunto.lower().split())
            if not outro:
                continue
            if len(alvo & outro) / len(alvo | outro) > 0.6:
                return True
        return False

    # ------------------------------------------------------------- tick

    def tick(self, estado, agora: float | None = None) -> Veredito:
        """Avalia se é hora de falar. Devolve o veredito sempre.

        Devolver o veredito mesmo quando não passa é de propósito: é o que
        deixa você olhar o log e entender POR QUE ela ficou quieta, em vez
        de só notar que ficou.
        """
        if agora is None:
            agora = time.time()

        if agora < self._bloqueado_ate:
            return Veredito(False, "aguardando depois de impulso rejeitado")

        self._limpar(agora)
        impulso = self._melhor(agora)

        if impulso is None:
            impulso = self._do_vazio(agora)

        v = self.portao.avaliar(
            impulso,
            silencio=agora - self.ultima_fala_alguem,
            desde_fala_dela=agora - self.ultima_fala_dela,
            falas_sem_resposta=self.falas_sem_resposta,
            ocupada=self.ocupada,
            estado=estado,
            agora=agora,
        )
        self.ultimo_veredito = v

        if v.passou and v.impulso is not None:
            # Segundo portao: passou no "quando", falta o "o que".
            ok, motivo = self.substancia.validar(v.impulso, self._ultima_fala_texto)
            if not ok:
                # Consome mesmo rejeitando: impulso sem substancia nao
                # melhora esperando na fila, e deixa-lo la faria o mesmo
                # lixo ser reavaliado a cada tick ate expirar.
                self._consumir(v.impulso)
                # Trava curta depois de rejeitar. `_do_vazio` FABRICA um
                # impulso novo a cada tick (ele nunca esteve na fila, entao
                # `_consumir` nao tem o que remover) -- sem esta trava, o
                # ciclo "fabrica -> passa no portao -> rejeita" se repetia
                # a cada 5s indefinidamente. Visto no simulador.
                self._bloqueado_ate = agora + self.cfg.consciencia.cooldown_fala
                return Veredito(False, f"sem substancia: {motivo}",
                                v.limiar, v.forca, v.impulso, avaliado=True)
            self._consumir(v.impulso)
        return v

    def _limpar(self, agora: float) -> None:
        self.fila = deque(
            (i for i in self.fila if not i.expirado(agora)), maxlen=self.fila.maxlen
        )
        # Fio velho demais deixa de ser retomada e vira estranheza: perguntar
        # sobre algo de três dias atrás no meio de outra conversa não soa
        # atencioso, soa fora de contexto.
        limite = self.cfg.consciencia.horas_para_fio_azedar
        self.fios = deque(
            (f for f in self.fios if not f.usado and f.idade_horas() < limite),
            maxlen=self.fios.maxlen,
        )

    def _melhor(self, agora: float) -> Impulso | None:
        vivos = [i for i in self.fila if not i.expirado(agora)]
        return max(vivos, key=lambda i: i.forca) if vivos else None

    def _do_vazio(self, agora: float) -> Impulso | None:
        """Sem nada na fila: tenta um fio, senão o impulso vazio.

        O fio mais recente primeiro. Retomar a última coisa que a pessoa
        mencionou é mais natural que desenterrar a de anteontem.
        """
        for fio in reversed(self.fios):
            if not fio.usado:
                nome = self.nomes.get(str(fio.usuario))
                # Sem nome conhecido, a atribuição some em vez de virar id.
                alvo = f"o que {nome} mencionou" if nome else "o que foi dito antes"
                return criar_impulso(
                    "fio", f"retomar {alvo}: {fio.assunto}", origem=fio.usuario,
                )
        # forca_vazio (config) SOBRESCREVE FORCA_PADRAO["vazio"]. Isso já
        # custou uma calibragem: mexer só na tabela não muda nada aqui, e
        # o simulador mostrou "silêncio absoluto -> 0 falas" mesmo com a
        # tabela em 0.62. Quem manda é o config -- então o default dele
        # subiu junto (config.py, EVA_INICIATIVA_VAZIO_FORCA), e a tabela
        # fica como piso pra quem não define a variável.
        return criar_impulso(
            "vazio", "puxar assunto",
            forca=max(self.cfg.consciencia.forca_vazio,
                      FORCA_PADRAO["vazio"]),
        )

    def _consumir(self, impulso: Impulso) -> None:
        if impulso in self.fila:
            self.fila.remove(impulso)
        if impulso.tipo == "fio":
            for f in self.fios:
                if f.assunto in impulso.conteudo:
                    f.usado = True
                    break

    # ------------------------------------------------------------- info

    def situacao(self, agora: float | None = None) -> dict:
        if agora is None:
            agora = time.time()
        return {
            "canal": self.canal,
            "silencio": round(agora - self.ultima_fala_alguem, 1),
            "desde_fala_dela": round(agora - self.ultima_fala_dela, 1),
            "falas_sem_resposta": self.falas_sem_resposta,
            "impulsos": [str(i) for i in self.fila],
            "fios": [f.assunto for f in self.fios if not f.usado],
            "ultimo_falante": self.ultimo_falante,
            "ocupada": self.ocupada,
            # Motivo real do último tick, com os números -- é o que
            # permite responder "por que ela não puxou assunto" olhando o
            # dashboard em vez de caçar no console (ver Veredito.__str__).
            "ultimo_veredito": str(self.ultimo_veredito) if self.ultimo_veredito else None,
        }