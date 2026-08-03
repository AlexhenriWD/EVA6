"""
Captura de tela e detecção de mudança visual.

CAPTURA: tela local do PC (mss), não a call do Discord -- a API de voz do
Discord não expõe vídeo de entrada para um bot, então "ver o que o usuário
está mostrando" só é possível capturando a tela de quem roda a EVA
(compartilhar tela no Discord é irrelevante aqui; a EVA olha o monitor
direto, sempre).

POR QUE NÃO 1 IMAGEM POR SEGUNDO
---------------------------------
Uma imagem em resolução normal custa 4-5s de análise no MiniCPM-V (mais de
mil tokens de visão em prefill, num modelo 8B rodando junto com o
conversacional na mesma GPU). Rodar a cada segundo faria a visão brigar
constantemente pela GPU -- você falaria e ela demoraria porque "estava
olhando a tela" naquele instante.

A saída não é reduzir a frequência da CAPTURA (que é barata, é só um
screenshot), é filtrar ANTES de chamar o modelo caro. É essa a função do
DetectorDiferenca abaixo.

O DETECTOR -- por que não YOLO
--------------------------------
YOLO/RT-DETR foram cogitados e descartados: treinados em COCO (pessoa,
carro, cachorro), não detectam nada útil numa tela de jogo, IDE ou
documento -- "trocou de aba", "morreu pro chefe", "abriu um menu" não são
classes do COCO.

O que funciona aqui é mais burro e mais barato: diferença de quadros em
baixa resolução (32x32, escala de cinza -- ~1ms de CPU, zero VRAM). A
parte que importa é o limiar ser ADAPTATIVO, não fixo: num jogo rápido ou
vídeo rodando, quadro a quadro muda o tempo todo (movimento normal,
"micro"), e um limiar fixo dispararia a cada captura. Mantendo média móvel
e desvio padrão da diferença recente, só dispara quando o quadro foge
muito da própria linha de base -- que é exatamente um corte de cena: menu
→ partida, troca de janela, tela de morte, documento novo. Em jogo rápido
a linha de base já é alta, então só um corte real dispara; em tela parada
a linha de base é ~0, então qualquer mudança já dispara. O mesmo código se
calibra sozinho pros dois casos.
"""

from __future__ import annotations

import io
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image


class Mudanca(Enum):
    NENHUMA = "nenhuma"
    MICRO = "micro"    # movimento normal -- ignorar, não vale chamar o VLM
    MACRO = "macro"    # corte de cena -- vale chamar o VLM


@dataclass
class Veredito:
    mudanca: Mudanca
    diferenca: float
    limiar: float

    def __str__(self) -> str:
        return f"{self.mudanca.value} (diff={self.diferenca:.2f}, limiar={self.limiar:.2f})"


class DetectorDiferenca:
    """Puro em numpy, sem I/O -- testável com frames sintéticos, sem tela
    nem GPU nenhuma. Ver o teste em vision/captura.py::calibrar (rodável
    como script) ou os testes automatizados.
    """

    def __init__(self, tamanho: tuple[int, int] = (32, 32),
                 janela: int = 20, desvios: float = 2.5,
                 minimo_absoluto: float = 3.0):
        self.tamanho = tamanho
        self.desvios = desvios
        # Diferença absoluta mínima para sequer considerar "mudança" -- sem
        # isso, uma tela perfeitamente estática (desvio~0 no histórico
        # ainda curto) dispararia em ruído de captura mínimo.
        self.minimo_absoluto = minimo_absoluto
        self._historico: deque[float] = deque(maxlen=janela)
        self._anterior: np.ndarray | None = None

    def _reduzir(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame).convert("L").resize(
            self.tamanho, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)

    def avaliar(self, frame: np.ndarray) -> Veredito:
        pequeno = self._reduzir(frame)

        if self._anterior is None:
            self._anterior = pequeno
            return Veredito(Mudanca.NENHUMA, 0.0, 0.0)

        diferenca = float(np.mean(np.abs(pequeno - self._anterior)))
        self._anterior = pequeno

        # Janela ainda não aqueceu -- evita falso positivo no início, antes
        # de haver linha de base suficiente pra saber o que é "normal".
        minimo_amostras = max(5, (self._historico.maxlen or 20) // 4)
        if len(self._historico) < minimo_amostras:
            self._historico.append(diferenca)
            return Veredito(Mudanca.NENHUMA, diferenca, 0.0)

        media = float(np.mean(self._historico))
        desvio = float(np.std(self._historico))
        limiar = max(media + self.desvios * desvio, self.minimo_absoluto)

        self._historico.append(diferenca)

        if diferenca > limiar:
            return Veredito(Mudanca.MACRO, diferenca, limiar)
        if diferenca > media * 1.3 and diferenca > self.minimo_absoluto * 0.5:
            return Veredito(Mudanca.MICRO, diferenca, limiar)
        return Veredito(Mudanca.NENHUMA, diferenca, limiar)

    def reiniciar(self) -> None:
        """Zera a linha de base. Use ao trocar de fonte de captura (ex:
        trocar de monitor) -- sem isso, o primeiro quadro da fonte nova
        seria comparado contra o último da fonte antiga."""
        self._historico.clear()
        self._anterior = None


class CapturaTela:
    """Adaptador fino sobre mss -- a única parte que de fato toca hardware.

    Requer: pip install mss pillow numpy

    THREAD-LOCAL DE PROPÓSITO: mss não pode ser compartilhado entre
    threads (é limitação documentada da própria lib, mais forte ainda no
    backend do Windows). Como a captura roda via asyncio.to_thread e o
    pool de threads do asyncio pode reusar threads diferentes a cada
    chamada, cachear uma única instância em `self` arriscaria um mss sendo
    chamado da thread errada -- e o sintoma seria uma falha esporádica,
    difícil de reproduzir. Uma instância por thread (via threading.local)
    resolve isso de vez.
    """

    def __init__(self, monitor: int = 1, largura: int = 672):
        self.monitor = monitor
        self.largura = largura
        self._local = threading.local()

    def _sct(self):
        if not hasattr(self._local, "sct"):
            import mss
            self._local.sct = mss.mss()
        return self._local.sct

    def capturar(self) -> np.ndarray:
        """RGB, redimensionado para `largura` (mantendo proporção).

        672px de lado maior corta os tokens de visão do MiniCPM-V por 3-4x
        frente a 1280x720 -- é o que tira a análise de ~5s para ~2s. Serve
        bem para "o que estou vendo"; não serve para ler texto miúdo numa
        tela (isso pediria resolução maior e análise mais lenta -- se um
        dia precisar, capture em dois passes: baixa resolução para o
        detector de diferença, alta resolução só quando for de fato
        analisar).
        """
        sct = self._sct()
        monitor = sct.monitors[self.monitor]
        shot = sct.grab(monitor)
        # .rgb já vem convertido de BGRA -> RGB pela própria mss; usar o
        # atributo pronto evita reimplementar a troca de canais na mão.
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        razao = self.largura / img.width
        altura = max(1, int(img.height * razao))
        img = img.resize((self.largura, altura), Image.BILINEAR)
        return np.asarray(img)

    def capturar_jpeg(self, qualidade: int = 85) -> bytes:
        arr = self.capturar()
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG", quality=qualidade)
        return buf.getvalue()

    def monitores_disponiveis(self) -> list[dict]:
        """Lista monitores para quem quiser trocar de EVA_VISAO_MONITOR.
        Índice 0 é sempre "todos os monitores combinados" na mss -- não
        costuma ser o que se quer; 1 é tipicamente o principal."""
        return list(self._sct().monitors)

    def fechar(self) -> None:
        if hasattr(self._local, "sct"):
            self._local.sct.close()
            del self._local.sct
