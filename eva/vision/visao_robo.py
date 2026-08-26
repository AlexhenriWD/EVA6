"""
Orquestração da visão do ROBÔ -- mesmo padrão de visao.py (SistemaVisual):
CENA lenta/persistente entra em todo prompt via contexto_visual; EVENTO
rápido/transitório vira Impulso na Consciencia, só comentado se o Portão
de fala aprovar. Ver visao.py pela explicação completa de por que separar
os dois evita o "narrador de si mesmo" -- é o mesmo raciocínio aqui, não
repetido.

POR QUE MÓDULO SEPARADO EM VEZ DE PARAMETRIZAR SistemaVisual: a fonte de
captura muda (câmera do robô via rede, não mss local), e sobretudo o
ciclo de vida é diferente -- a visão de tela sempre tem uma fonte (a
tela existe assim que o processo sobe); a do robô pode estar "ligada"
(ativo=True, a call está de pé) sem ter quadro nenhum ainda, porque a
conexão com o robô é sob demanda (ver eva_command_server.py e
integrations/robot_client.py) e pode subir DEPOIS da call já ter
começado. tick() trata isso como qualquer outra falha de captura
best-effort -- não é erro, é "ainda não". DetectorDiferenca é reusado
direto (puro numpy, nada específico de tela).
"""

from __future__ import annotations

import time

from .captura import DetectorDiferenca, Mudanca
from .minicpm import ClienteVisao, ErroVisao, PROMPT_CENA_ROBO
from .visao import RegistroCena, _similaridade


class CapturaRobo:
    """Adaptador fino, mesmo formato de captura.CapturaTela
    (capturar()/capturar_jpeg()) -- pra caber no mesmo desenho de
    SistemaVisual sem duplicar DetectorDiferenca nem a lógica de rajada.

    NÃO GUARDA REFERÊNCIA ao cliente de vídeo -- pede o quadro atual via
    tools.robot_tools.obter_quadro_camera() toda vez. Resolve um
    problema de ordem real: esta captura pode ser criada na entrada da
    call, ANTES do robô ter conectado de verdade (conexão é sob
    demanda) -- guardar uma referência guardaria None pra sempre, mesmo
    depois do robô conectar.
    """

    def capturar(self):
        import io
        import numpy as np
        from PIL import Image

        jpeg = self.capturar_jpeg()
        if jpeg is None:
            raise RuntimeError(
                "sem quadro da câmera do robô ainda (desconectado, ou "
                "conexão de vídeo não chegou a tempo)"
            )
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        return np.asarray(img)

    def capturar_jpeg(self, qualidade: int = 85) -> bytes | None:
        # `qualidade` ignorada de propósito -- o quadro já vem
        # comprimido pela própria câmera do robô (ver camera_manager.py,
        # lado Pi); recomprimir aqui só perderia qualidade à toa.
        from ..tools import robot_tools
        return robot_tools.obter_quadro_camera()


class SistemaVisualRobo:
    """Câmera do robô -- mesmo padrão de SistemaVisual: tick() barato por
    padrão, só paga o modelo caro quando o DetectorDiferenca aprova
    MACRO. Não roda laço próprio -- tick() é chamado pelo integrador
    (bridge_client._laco_visao_robo) no ritmo configurado.
    """

    def __init__(self, cfg):
        self.cfg = cfg.visao  # mesma config de servidor/modelo da visão de tela
        self.captura = CapturaRobo()
        self.detector = DetectorDiferenca(
            desvios=self.cfg.limiar_desvios,
            minimo_absoluto=self.cfg.limiar_minimo_absoluto,
        )
        self.cliente = ClienteVisao(
            base_url=self.cfg.base_url, modelo=self.cfg.modelo,
            api_key=self.cfg.api_key, timeout=self.cfg.timeout,
        )
        self.cena: RegistroCena | None = None
        self.ativo = False  # controlado por quem integra (ligado durante call)

    # ------------------------------------------------------------- tick

    def tick(self, agora: float | None = None) -> str | None:
        """Ver SistemaVisual.tick() -- mesmo contrato exato. Erro de
        captura (inclusive "robô ainda não conectou") NUNCA propaga
        daqui, mesmo tratamento best-effort."""
        if not self.ativo:
            return None

        try:
            frame = self.captura.capturar()
        except Exception as e:
            if self.cfg.debug:
                print(f"[visao-robo] sem quadro ainda: {e}")
            return None

        veredito = self.detector.avaliar(frame)
        if self.cfg.debug and veredito.mudanca != Mudanca.NENHUMA:
            print(f"[visao-robo] {veredito}")

        if veredito.mudanca != Mudanca.MACRO:
            return None

        return self._analisar_mudanca(agora)

    def _analisar_mudanca(self, agora: float | None) -> str | None:
        try:
            quadros = self._capturar_rajada()
            if not quadros:
                return None
            descricao = self.cliente.analisar(quadros, PROMPT_CENA_ROBO)
        except ErroVisao as e:
            if self.cfg.debug:
                print(f"[visao-robo] erro na análise: {e}")
            return None
        except Exception as e:
            if self.cfg.debug:
                print(f"[visao-robo] erro inesperado na análise: {e}")
            return None

        if not descricao:
            return None

        agora = agora or time.time()

        if self.cena is None:
            self.cena = RegistroCena(descricao, agora)
            return descricao  # primeira cena da sessão também é evento

        similaridade = _similaridade(descricao, self.cena.descricao)
        if similaridade >= self.cfg.limiar_mudanca_cena:
            return None

        self.cena = RegistroCena(descricao, agora)
        return descricao

    def _capturar_rajada(self) -> list[bytes]:
        """N quadros JPEG espaçados no tempo -- mesmo raciocínio de
        SistemaVisual: sequência curta dá noção de movimento que uma
        foto isolada não dá, e no robô isso importa tanto quanto (ou
        mais) que na tela. Descarta silenciosamente qualquer captura
        que vier None no meio (robô desconectou no meio da rajada)."""
        quadros = []
        for i in range(self.cfg.rajada_quadros):
            quadro = self.captura.capturar_jpeg()
            if quadro is not None:
                quadros.append(quadro)
            if i < self.cfg.rajada_quadros - 1:
                time.sleep(self.cfg.rajada_intervalo)
        return quadros

    # -------------------------------------------------------- contexto

    def contexto_atual(self, ttl_segundos: float | None = None) -> str | None:
        """A cena para injetar via contexto_visual em EVA.responder()
        quando a pergunta é sobre o robô. Mesmo TTL de SistemaVisual."""
        if self.cena is None:
            return None
        ttl = ttl_segundos if ttl_segundos is not None else self.cfg.cena_ttl
        if self.cena.idade_segundos() > ttl:
            return None
        return self.cena.descricao

    def analisar_agora(self) -> str | None:
        """Captura e analisa AGORA, ignorando o detector de diferença e
        o filtro de similaridade -- usado quando a pessoa pergunta
        diretamente sobre o robô (ver bridge_client._contexto_visual_robo_para)
        ou depois de robo_olhar mover a cabeça. Vale pagar mais uma
        chamada ao modelo de visão mesmo que nada tenha "mudado o
        bastante" pra disparar o tick de fundo."""
        if not self.ativo:
            return None
        try:
            quadros = self._capturar_rajada()
            if not quadros:
                return None
            descricao = self.cliente.analisar(quadros, PROMPT_CENA_ROBO)
        except ErroVisao as e:
            if self.cfg.debug:
                print(f"[visao-robo] erro na análise sob demanda: {e}")
            return None
        except Exception as e:
            if self.cfg.debug:
                print(f"[visao-robo] erro inesperado na análise sob demanda: {e}")
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
