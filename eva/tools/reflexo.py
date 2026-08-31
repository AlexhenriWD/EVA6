"""
reflexo.py -- a camada de movimento que NÃO passa pelo modelo.

POR QUE EXISTIR, dado que _ciclo_iniciativa já existe: são autonomias
diferentes, e o sistema só tinha a pior das duas.

    _ciclo_iniciativa   acorda a cada 300s, monta um prompt, pergunta ao
                        LLM "quer se mexer?". Uma inferência por impulso,
                        cinco minutos entre um e outro. Isso não é um
                        corpo que reage -- é um agendador que às vezes
                        lembra que tem um corpo.

    _ciclo_reflexo      roda a cada 0.4s, decide em código, custo zero.

Nada que se pareça com uma criatura DECIDE, conscientemente, virar a
cabeça quando algo se mexe no canto do olho. Isso acontece antes de
qualquer pensamento, e o pensamento chega depois, já com a cabeça
virada. Um sistema que só tem a camada deliberativa vai sempre parecer
teleoperado -- mesmo quando o operador é ele mesmo -- porque toda ação
passa por um "eu decidi".

EIXOS: usa cabeça (canal 3, horizontal) e pitch (canal 1, vertical).
NÃO usa yaw: girar a base torce o flat CSI da PiCam e leva quase um
segundo por movimento suave, o que é lento demais pra reflexo. Cotovelo
também fica de fora -- ele muda a ALTURA do ponto de vista, que é
decisão de postura, não reação a movimento.

HONESTIDADE: o reflexo escreve em _eventos_corpo dizendo explicitamente
que foi reflexo, não decisão. Isso importa mais do que parece -- este
sistema já produziu, em uso real, "vou dar uma olhada em volta" sem ter
olhado, uma estante com livros de ficção científica que não existia, e
um cheiro de café num robô sem nariz. Movimento automático narrado como
escolha seria a quarta confabulação, e a pior: a única em que o mundo
físico concorda com a mentira.
"""

from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

# ---------------------------------------------------------------- ajustes

# Período do laço. 0.4s dá ~2.5 correções por segundo: rápido o bastante
# pra parecer atenção, lento o bastante pra não brigar com os 15fps do
# stream nem saturar a fila de comandos do Pi.
INTERVALO_REFLEXO_S = 0.4

# Resolução de trabalho. Diferença de quadro não precisa de detalhe --
# precisa de ONDE, e 64x48 responde isso em microssegundos. Subir isto
# não melhora o seguimento, só gasta CPU.
LARGURA_TRABALHO = 64
ALTURA_TRABALHO = 48

# Quanto de diferença conta como "algo se mexeu". Em unidades de nível de
# cinza médio por pixel na região mais ativa. Precisa ser alto o
# suficiente pra ignorar ruído de sensor com pouca luz -- a picam em
# ambiente escuro produz granulado que, sem limiar, faz a cabeça
# perseguir estática a noite toda.
LIMIAR_MOVIMENTO = 12.0

# Fração do quadro que o alvo pode estar fora do centro sem correção.
# Sem zona morta o servo nunca para: sempre há um pixel de erro, e o
# resultado lê como tique nervoso, não como atenção.
ZONA_MORTA = 0.12

# Ganho da correção, em graus por unidade de desvio normalizado. Baixo de
# propósito: a cabeça PERSEGUE devagar em vez de saltar. Servo saltando
# entre posições é a diferença entre "acompanhando algo" e "com defeito".
#
# GANHO_VERTICAL é NEGATIVO de propósito. dy vem negativo quando o alvo
# está na parte de CIMA da imagem, e pitch MAIOR levanta a mira -- o
# contrário do que arm_controller.look_up() sempre assumiu (ele faz
# current - graus). Confirmado por foto na calibração: cotovelo 120 com
# pitch 110 aponta pra frente; com pitch 70 aponta pra trás e pra baixo.
# Com ganho positivo a cabeça FUGIRIA do alvo em vez de segui-lo.
GANHO_HORIZONTAL = 14.0
GANHO_VERTICAL = -8.0

# Passo máximo por ciclo. Trava dura contra um flash de luz ou alguém
# passando colado na câmera mandarem a cabeça pro batente de uma vez.
PASSO_MAX_GRAUS = 8

# Curso dos eixos que o reflexo usa. Espelha _CURSO_EIXO de robot_tools;
# safety.py no Pi continua sendo quem valida de verdade. Isto existe pra
# o reflexo não gastar ida e volta de rede pedindo ângulo que vai ser
# recusado -- e, pior, receber um dict de erro que ninguém lê.
CURSO_CABECA = (0, 117)
# Só a faixa em que o pitch REALMENTE muda a mira. O curso físico é
# 40..110, mas abaixo de ~95 ele para de inclinar a câmera e passa a só
# recuar o braço -- medido na calibração (cotovelo 140: pitch
# 100→90→80→70 manteve o mesmo ângulo de câmera, mudando só a posição do
# cotovelo). Deixar o reflexo usar a faixa inteira gastaria comando em
# movimento que não corrige nada e, pior, mudaria a POSTURA sem ninguém
# ter pedido. Sobram ~15° de autoridade vertical contra 117 na
# horizontal: o seguimento lateral é o que funciona de verdade aqui.
CURSO_PITCH = (95, 110)

# Intervalo mínimo entre dois avisos em _eventos_corpo. Um evento por
# movimento viraria spam e a Consciência comentaria cada tremida de
# cortina.
INTERVALO_EVENTO_S = 20.0

# Quantos ciclos seguidos com movimento na MESMA região antes de tratar
# como "tem alguém aí" em vez de "alguma coisa piscou". 6 ciclos a 0.4s
# = ~2.4s de presença contínua. Cortina balançando e sombra passando não
# sustentam isso; pessoa parada mexendo, sim.
CICLOS_PARA_PRESENCA = 6
# Quanto o alvo pode andar entre um ciclo e outro e ainda contar como o
# mesmo. Generoso: pessoa se mexe.
TOLERANCIA_MESMA_REGIAO = 0.45


def _cinza_reduzido(jpeg: bytes) -> np.ndarray | None:
    try:
        img = Image.open(io.BytesIO(jpeg)).convert("L")
        img = img.resize((LARGURA_TRABALHO, ALTURA_TRABALHO))
        return np.asarray(img, dtype=np.int16)
    except Exception:
        # Quadro truncado/corrompido acontece -- o stream já foi cortado
        # no meio antes. Reflexo não é lugar de tratar isso: devolve None
        # e o próximo ciclo tenta de novo, 0.4s depois.
        return None


def centroide_de_movimento(anterior: bytes, atual: bytes) -> tuple[float, float] | None:
    """Onde algo se mexeu, em coordenadas normalizadas.

    Devolve (dx, dy) com -1.0 = borda esquerda/topo, +1.0 =
    direita/baixo, 0.0 = centro. None quando nada passou do limiar.

    Diferença de quadro em escala de cinza, com o centroide ponderado
    pela intensidade da mudança. É burro de propósito: sem modelo, sem
    detecção de pessoa, sem rede, sem GPU. Roda em microssegundos e por
    isso PODE rodar a cada 0.4s sem competir com o llama-server -- que é
    o requisito real. Um reflexo que espera inferência não é reflexo.

    O preço de ser burro: ele segue qualquer movimento, não só pessoas.
    Cortina, sombra, tela de monitor mudando. Na prática, numa sala com
    uma pessoa, a pessoa é de longe a maior fonte de mudança -- e quando
    não é, o pior caso é a câmera olhar pra coisa errada por alguns
    segundos, o que é bem menos grave que perder o seguimento."""
    a = _cinza_reduzido(anterior)
    b = _cinza_reduzido(atual)
    if a is None or b is None:
        return None

    dif = np.abs(b - a)
    if float(dif.max()) < LIMIAR_MOVIMENTO:
        return None

    # Zera o que está abaixo do limiar antes de pesar o centroide. Sem
    # isso o ruído de fundo, espalhado pelo quadro inteiro, puxa o
    # centroide pro centro geométrico e o alvo real some na média.
    dif = np.where(dif >= LIMIAR_MOVIMENTO, dif, 0)
    total = float(dif.sum())
    if total <= 0:
        return None

    colunas = dif.sum(axis=0)
    linhas = dif.sum(axis=1)
    cx = float(np.dot(np.arange(len(colunas)), colunas) / total)
    cy = float(np.dot(np.arange(len(linhas)), linhas) / total)

    dx = (cx / (len(colunas) - 1)) * 2.0 - 1.0
    dy = (cy / (len(linhas) - 1)) * 2.0 - 1.0
    return dx, dy


def _passo(desvio: float, ganho: float) -> int:
    graus = desvio * ganho
    graus = max(-PASSO_MAX_GRAUS, min(PASSO_MAX_GRAUS, graus))
    return int(round(graus))


def alvo_do_reflexo(dx: float, dy: float, cabeca: int, pitch: int) -> dict:
    """Ângulos que aproximam a mira do que se mexeu.

    Devolve só os eixos que realmente mudam -- mandar um eixo pro mesmo
    ângulo em que já está faz o ArmController responder "já estava nessa
    posição" e gasta uma ida ao Pi à toa, 2.5x por segundo.

    O SINAL de cada eixo é hipótese, não medição. Horizontal: assume que
    aumentar o canal 3 move a mira pra direita da imagem. Vertical:
    assume que DIMINUIR o pitch levanta (mesma suposição de
    arm_controller.look_up, que faz current - graus e nunca foi conferida
    contra imagem). Se o seguimento fugir do alvo em vez de persegui-lo,
    é sinal trocado -- inverta o ganho correspondente, não o código."""
    alvo: dict = {}

    if abs(dx) > ZONA_MORTA:
        lo, hi = CURSO_CABECA
        novo = max(lo, min(hi, cabeca + _passo(dx, GANHO_HORIZONTAL)))
        if abs(novo - cabeca) >= 2:
            alvo["cabeca"] = novo

    if abs(dy) > ZONA_MORTA:
        lo, hi = CURSO_PITCH
        novo = max(lo, min(hi, pitch + _passo(dy, GANHO_VERTICAL)))
        if abs(novo - pitch) >= 2:
            alvo["pitch"] = novo

    return alvo


def descrever_presenca(dx: float) -> str:
    """Frase pra quando o movimento sustenta -- o gatilho de "tem alguém"."""
    lado = "à minha esquerda" if dx < 0 else "à minha direita"
    return (f"tem alguma coisa se mexendo {lado} há alguns segundos, "
            f"do jeito que gente parada mexendo se mexe -- minha câmera "
            f"está acompanhando sozinha")


def descrever_evento(dx: float, dy: float) -> str:
    """Frase pra _eventos_corpo. Sempre diz que foi reflexo.

    Ela descobre o próprio movimento pelo mesmo canal do emergency stop:
    alguém contou. Não afirma ter escolhido, afirma ter notado -- que é
    a única das duas que é verdade, e a que mantém a regra de não
    fabricar presença física."""
    lado = "à esquerda" if dx < 0 else "à direita"
    if abs(dy) > abs(dx) * 1.5:
        lado = "acima" if dy < 0 else "abaixo"
    return (f"alguma coisa se mexeu {lado} e minha câmera foi atrás sozinha "
            f"-- reflexo, não decisão minha")


class EstadoReflexo:
    """Memória curta do laço: último quadro, último aviso, supressão.

    Fica numa classe em vez de globais soltas porque robot_tools já tem
    módulo-global demais, e porque a supressão precisa ser setada de
    FORA (por robo_olhar, robo_gesto, robo_postura) -- reflexo brigando
    com comando explícito produz movimento que ninguém pediu e ninguém
    consegue explicar, inclusive ela."""

    def __init__(self) -> None:
        self.ativo = False
        self.quadro_anterior: bytes | None = None
        self.ultimo_evento_ts = 0.0
        self.suprimir_ate = 0.0
        # Ângulos comandados por último. Mantidos localmente em vez de
        # perguntar ao robô a cada ciclo: estado() é uma ida e volta de
        # rede, e 2.5 delas por segundo entupiriam a fila de comandos do
        # Pi -- que processa uma mensagem de cada vez. Enquanto o reflexo
        # está ativo ele é o único mexendo nestes dois eixos (a supressão
        # cobre todo o resto), então o valor local é a verdade. Ao sair
        # de uma supressão, ressemeia do robô.
        self.cabeca = 90
        self.pitch = 90
        self.precisa_ressemear = True
        # Presença sustentada -- ver CICLOS_PARA_PRESENCA.
        self._ciclos_seguidos = 0
        self._ultimo_dx = 0.0
        self.ultima_presenca_ts = 0.0

    def suprimir(self, segundos: float = 3.0) -> None:
        """Cala o reflexo por um tempo -- chamar no início de qualquer
        ferramenta que mova o corpo de propósito.

        Reflexo brigando com comando explícito produz movimento que
        ninguém pediu e ninguém consegue explicar, inclusive ela. E
        depois da ferramenta o corpo está em outro lugar, então tanto o
        quadro de referência quanto os ângulos locais são jogados fora."""
        self.suprimir_ate = time.monotonic() + segundos
        self.quadro_anterior = None
        self.precisa_ressemear = True
        self._ciclos_seguidos = 0

    def registrar_movimento(self, dx: float) -> bool:
        """Conta ciclos seguidos na mesma região. True quando vira presença.

        Isto é o que separa "algo piscou" de "tem gente aí" -- e é o
        gatilho que faz sentido pra ela querer falar. Muito melhor que o
        despertador de 5 minutos do _ciclo_iniciativa: acontece quando há
        motivo, não quando o relógio bate."""
        if abs(dx - self._ultimo_dx) <= TOLERANCIA_MESMA_REGIAO:
            self._ciclos_seguidos += 1
        else:
            self._ciclos_seguidos = 1
        self._ultimo_dx = dx

        if self._ciclos_seguidos < CICLOS_PARA_PRESENCA:
            return False
        agora = time.monotonic()
        if agora - self.ultima_presenca_ts < INTERVALO_EVENTO_S * 3:
            return False
        self.ultima_presenca_ts = agora
        self._ciclos_seguidos = 0
        return True

    def perdeu_referencia(self) -> None:
        self.quadro_anterior = None
        self._ciclos_seguidos = 0

    def pode_agir(self) -> bool:
        return self.ativo and time.monotonic() >= self.suprimir_ate

    def deve_avisar(self) -> bool:
        agora = time.monotonic()
        if agora - self.ultimo_evento_ts < INTERVALO_EVENTO_S:
            return False
        self.ultimo_evento_ts = agora
        return True