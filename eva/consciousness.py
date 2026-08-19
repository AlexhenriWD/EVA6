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

import re
import time
from collections import deque
from dataclasses import dataclass, field

# Força de cada tipo de impulso. Não é ajuste fino: reflete o quanto o
# conteúdo é específico. Retomar algo que a pessoa disse vale mais que
# comentar que a tela mudou, que vale mais que falar por falar.
FORCA_PADRAO = {
    "fio": 0.70,
    "visual": 0.65,
    "corporal": 0.65,  # mesmo peso de visual -- notar o próprio corpo é
                       # tão específico quanto notar a tela. Só emitido
                       # em transição de segurança/recusa, nunca por
                       # movimento rotineiro -- ver robot_tools.py.
    "pesquisa": 0.60,
    "vazio": 0.35,
}

# Quanto tempo cada tipo continua valendo depois de criado.
TTL_PADRAO = {
    "fio": 900.0,      # 15 min -- fio é durável, o assunto não azeda rápido
    "visual": 60.0,    # a tela já mudou de novo
    "corporal": 90.0,  # estado físico muda rápido -- comentar "acabei de
                       # tomar um susto" 3min depois já ficou esquisito
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
        self.falas_sem_resposta = 0
        self.ocupada = False

        self.ultimo_veredito: Veredito | None = None

    # ---------------------------------------------------------- eventos

    def registrar_nome(self, usuario: str, nome: str | None) -> None:
        if nome:
            self.nomes[str(usuario)] = nome

    def alguem_falou(self, usuario: str, mensagem: str = "", plano=None,
                     nome: str | None = None) -> None:
        """Registra fala humana. Zera a escalada e colhe fios."""
        self.ultima_fala_alguem = time.time()
        self.ultimo_falante = usuario
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
        """Gancho da fase 3. A visão chama aqui quando algo muda de verdade."""
        self.adicionar(criar_impulso("visual", descricao))

    def evento_corporal(self, descricao: str) -> None:
        """Transição de segurança do corpo físico (entrou/saiu de
        emergency stop, watchdog caiu) ou recusa de comando de
        movimento -- NUNCA movimento rotineiro bem-sucedido. Quem filtra
        isso é robot_tools._detectar_transicao_seguranca/_descrever_recusa,
        não aqui -- este método só empacota o que já chegou filtrado."""
        self.adicionar(criar_impulso("corporal", descricao))

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
        return criar_impulso(
            "vazio", "puxar assunto",
            forca=self.cfg.consciencia.forca_vazio,
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