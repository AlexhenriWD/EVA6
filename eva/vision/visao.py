"""
Orquestração da visão -- liga captura, detector e MiniCPM-V, e produz duas
coisas de natureza diferente: CENA e EVENTO.

A DISTINÇÃO QUE EVITA O NARRADOR
-----------------------------------
O modo de falha mais comum em sistema de reação visual (visto no próprio
código antigo do projeto, no V4) é: rodar o VLM num timer e comentar toda
descrição nova. Vira narrador de si mesmo -- "agora você abriu o
inventário", "agora está no menu" -- insuportável em noventa segundos.

A cura é separar dois produtos da mesma análise:

  CENA     lenta, persistente. "Usuário jogando Hades." É o "onde eu
           estou". Entra em TODO prompt via contexto_visual (ver
           context.py / EVA.responder), sempre a versão mais recente. Só
           muda quando a cena muda de verdade -- não gera fala sozinha,
           só informa a próxima resposta da EVA sobre o que está na tela.

  EVENTO   rápido, transitório. "O personagem morreu para o chefe." NÃO
           entra no prompt permanente -- vira um Impulso na Consciencia
           (evento_visual(), já existente) e só é comentado se o Portão
           de fala aprovar (silêncio suficiente, força do impulso acima
           do limiar, etc). Pode nunca ser dito, e está certo que seja
           assim: nem todo evento merece comentário.

Confundir os dois é o que produz o narrador. A cena responde "o que estou
vendo"; o evento responde "aconteceu algo que valha a pena mencionar".

QUANDO UMA ANÁLISE VIRA OS DOIS, UM OU NENHUM
------------------------------------------------
Todo disparo do detector (MACRO) gera uma análise via MiniCPM-V. O
resultado sempre atualiza a cena candidata. Só vira EVENTO (e só
atualiza a cena de fato) se a descrição nova for suficientemente
diferente da cena anterior -- comparação por sobreposição de palavras,
como no código de referência do V4 (_detect_visual_change), limiar 0.4.
Abaixo disso, o detector de diferença pegou uma mudança visual real (por
isso disparou MACRO) mas que não muda o que vale contar -- ex: HUD
piscando, animação de fundo -- e nada é atualizado nem disparado.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .captura import CapturaTela, DetectorDiferenca, Mudanca
from .minicpm import ClienteVisao, ErroVisao, PROMPT_CENA


@dataclass
class RegistroCena:
    descricao: str
    atualizada_em: float

    def idade_segundos(self, agora: float | None = None) -> float:
        return (agora or time.time()) - self.atualizada_em


def _similaridade(a: str, b: str) -> float:
    """Sobreposição de palavras entre duas descrições, 0.0-1.0.

    Mesma lógica do código de referência (V4): comparação simples, rápida,
    sem chamada de modelo. Não precisa ser sofisticada -- só precisa
    distinguir "mesma cena, palavras meio diferentes" de "cena
    genuinamente diferente", e sobreposição de palavras já resolve isso
    bem o suficiente para descrições curtas (a maioria abaixo de 20
    palavras, por causa do PROMPT_CENA).
    """
    palavras_a = set(a.lower().split())
    palavras_b = set(b.lower().split())
    if not palavras_a or not palavras_b:
        return 0.0
    intersecao = palavras_a & palavras_b
    uniao = palavras_a | palavras_b
    return len(intersecao) / len(uniao) if uniao else 0.0


class SistemaVisual:
    """Uma instância por processo -- há uma tela só (a do seu PC), não uma
    por guild/canal de voz, diferente da Consciencia (que é por canal
    porque cada call tem seu próprio silêncio e seus próprios impulsos).

    Não roda laço próprio -- `tick()` é chamado pelo integrador (ver
    bridge_client.py) no ritmo que fizer sentido. Igual à Consciencia,
    isso deixa o comportamento testável sem asyncio nem GPU: dá pra chamar
    `tick(frame_falso)` num teste e verificar o resultado.
    """

    def __init__(self, cfg):
        self.cfg = cfg.visao
        self.captura = CapturaTela(
            monitor=self.cfg.monitor, largura=self.cfg.largura_captura)
        self.detector = DetectorDiferenca(
            desvios=self.cfg.limiar_desvios,
            minimo_absoluto=self.cfg.limiar_minimo_absoluto,
        )
        self.cliente = ClienteVisao(
            base_url=self.cfg.base_url, modelo=self.cfg.modelo,
            api_key=self.cfg.api_key, timeout=self.cfg.timeout,
        )
        self.cena: RegistroCena | None = None
        self._rajada_pendente: list = []
        self.ativo = False  # controlado por quem integra (ligado durante call)

    # ------------------------------------------------------------- tick

    def tick(self, agora: float | None = None) -> str | None:
        """Um ciclo: captura, avalia diferença, e SE for MACRO, analisa.

        Devolve o texto do EVENTO se um foi gerado (para o chamador
        empurrar em Consciencia.evento_visual()), ou None na grande
        maioria das chamadas -- a maior parte dos ticks só atualiza o
        detector de diferença e não faz nada além disso, que é
        justamente o ponto: barato por padrão, caro só quando compensa.

        Erro de captura ou de análise NUNCA propaga daqui -- visão é
        funcionalidade auxiliar, uma falha nela não pode derrubar a
        conversa. Best-effort, mesmo padrão de busca/embeddings.
        """
        if not self.ativo:
            return None

        try:
            frame = self.captura.capturar()
        except Exception as e:
            if self.cfg.debug:
                print(f"[visao] erro na captura: {e}")
            return None

        veredito = self.detector.avaliar(frame)
        if self.cfg.debug and veredito.mudanca != Mudanca.NENHUMA:
            print(f"[visao] {veredito}")

        if veredito.mudanca != Mudanca.MACRO:
            return None

        return self._analisar_mudanca(agora)

    def _analisar_mudanca(self, agora: float | None) -> str | None:
        """Rajada de quadros + MiniCPM-V. Só chega aqui quando o detector
        já aprovou -- é a parte cara, e só roda quando compensa.
        """
        try:
            quadros = self._capturar_rajada()
            descricao = self.cliente.analisar(quadros, PROMPT_CENA)
        except ErroVisao as e:
            if self.cfg.debug:
                print(f"[visao] erro na análise: {e}")
            return None
        except Exception as e:
            if self.cfg.debug:
                print(f"[visao] erro inesperado na análise: {e}")
            return None

        if not descricao:
            return None

        agora = agora or time.time()

        if self.cena is None:
            self.cena = RegistroCena(descricao, agora)
            return descricao  # primeira cena da sessão também é evento

        similaridade = _similaridade(descricao, self.cena.descricao)
        if similaridade >= self.cfg.limiar_mudanca_cena:
            # o detector de diferença disparou (mudou visualmente) mas a
            # DESCRIÇÃO continua parecida -- HUD, animação de fundo, etc.
            # Não é mudança que vale contar. Nem cena nem evento mudam.
            return None

        self.cena = RegistroCena(descricao, agora)
        return descricao

    def _capturar_rajada(self) -> list[bytes]:
        """N quadros JPEG espaçados no tempo, para o MiniCPM-V ler como
        sequência curta em vez de foto isolada -- dá entendimento de
        movimento/direção que uma imagem única não dá.
        """
        quadros = []
        for i in range(self.cfg.rajada_quadros):
            quadros.append(self.captura.capturar_jpeg())
            if i < self.cfg.rajada_quadros - 1:
                time.sleep(self.cfg.rajada_intervalo)
        return quadros

    # -------------------------------------------------------- contexto

    def contexto_atual(self, ttl_segundos: float | None = None) -> str | None:
        """A cena para injetar via contexto_visual em EVA.responder().

        `ttl_segundos` descarta cena velha demais -- sem isso, se a
        captura for desligada (saiu da call) mas o processo continuar
        vivo, uma cena de horas atrás poderia entrar num prompt sem
        relação nenhuma com o que está acontecendo agora. Default vem de
        VisaoConfig.cena_ttl.
        """
        if self.cena is None:
            return None
        ttl = ttl_segundos if ttl_segundos is not None else self.cfg.cena_ttl
        if self.cena.idade_segundos() > ttl:
            return None
        return self.cena.descricao

    def analisar_agora(self) -> str | None:
        """Captura e analisa AGORA, ignorando o detector de diferença --
        usado quando a pessoa pede explicitamente pra olhar a tela
        (visao_relevante() deu True). Vale pagar o custo de mais uma
        chamada ao MiniCPM-V mesmo que nada tenha mudado o suficiente
        pra disparar o tick normal: contexto_atual() sozinho depende do
        último tick de fundo ter rodado E aprovado como MACRO, o que não
        tem relação nenhuma com o instante em que a pergunta foi feita.

        Diferente de _analisar_mudanca(), NÃO aplica o filtro de
        similaridade -- aqui a pessoa quer saber o que está na tela
        agora, não se mudou o bastante pra virar evento.
        """
        if not self.ativo:
            return None
        try:
            quadros = self._capturar_rajada()
            descricao = self.cliente.analisar(quadros, PROMPT_CENA)
        except ErroVisao as e:
            if self.cfg.debug:
                print(f"[visao] erro na análise sob demanda: {e}")
            return None
        except Exception as e:
            if self.cfg.debug:
                print(f"[visao] erro inesperado na análise sob demanda: {e}")
            return None
        if not descricao:
            return None
        self.cena = RegistroCena(descricao, time.time())
        return descricao

    # ---------------------------------------------------------- ciclo

    def ligar(self) -> None:
        self.ativo = True
        self.detector.reiniciar()

    def desligar(self) -> None:
        self.ativo = False

    def fechar(self) -> None:
        self.captura.fechar()
