"""
Ferramentas de robô -- o corpo físico dela, exposto pro mesmo mecanismo
de ferramenta que já existe (registro em registry.py).

MESMO ESPÍRITO de minecraft_tools.py: conexão PERSISTENTE numa thread
dedicada com seu próprio event loop asyncio; cada ferramenta (função
síncrona, mesmo contrato de sempre -- recebe argumento nomeado, devolve
dict) faz a ponte pra essa thread e espera o resultado.

DIFERENÇA CENTRAL: o corpo aqui é físico. Errar não reseta como um
respawn. Por isso:

- Nenhuma ferramenta aqui assume que o modelo "vai ser razoável" --
  quem valida de verdade é o SafetyController do lado do servidor
  (eva_command_server.py → eva_robot.py → safety.py). Se o servidor
  recusar, a ferramenta devolve o motivo, não insiste.
- Existe um heartbeat CONTÍNUO (a cada HEARTBEAT_INTERVALO_S) enquanto a
  conexão estiver de pé -- independente de estar se movendo ou não. Sem
  isso, o watchdog do robô (ver safety.py) desligaria o robô por
  "abandono" só porque a EVA estava pensando em outra coisa.
- A iniciativa (_ciclo_iniciativa) tem portões mais rígidos que os do
  Minecraft: bateria, watchdog, emergency_stop, e um teto de duração por
  movimento de iniciativa antes de parar e reler o estado real -- nunca
  deixa o modelo escolher "andar por muito tempo" numa tacada só.

A conexão só sobe na PRIMEIRA vez que alguma ferramenta de robô for
chamada de verdade. Sem EVA_ROBOT_HOST/eva_command_server.py rodando,
as ferramentas falham com {"erro": ...} como qualquer outra ferramenta
deste projeto quando algo dá errado (nunca levantam exceção).

IMPORTANTE: assim como minecraft_tools, pra estas ferramentas serem
alcançáveis de verdade o decisor por LLM precisa estar ligado
(EVA_DECISION_LLM=1) -- decidir "quando ela deve andar/olhar/explorar"
não é padrão de texto escrito à mão, é julgamento.

ONDE ISSO DEVE FICAR: ao lado de minecraft_tools.py (mesmo pacote
`tools`) -- ajuste os imports relativos se o layout real for outro.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
from typing import Callable

from ..integrations.robot_client import ClienteRobo
from ..integrations.robot_video_client import ClienteVideoRobo
from .registry import registro

_cliente: ClienteRobo | None = None
_cliente_video: ClienteVideoRobo | None = None
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_pronto = threading.Event()
# Protege só a CRIAÇÃO da thread (não o resto de _iniciar_thread_robo) --
# ver docstring da função pra o bug real que isso fecha.
_lock_inicio = threading.Lock()

# ===========================================================================
# PONTE PRA CONSCIÊNCIA -- ver bridge_client.py (_ligar_visao_se_precisar/
# _desligar_visao_se_precisar chamam definir_em_call; _laco_corpo drena
# drenar_eventos_corpo). Este módulo não importa bridge_client de volta --
# a direção é sempre daqui pra lá, thread-safe:
#
#   _em_call        threading.Event, só liga/desliga -- seguro ler/escrever
#                    de qualquer thread sem lock.
#   _eventos_corpo   queue.Queue (thread-safe por natureza) -- este módulo
#                    só PRODUZ (via _emitir_evento_corpo), bridge_client
#                    só CONSOME (via drenar_eventos_corpo). Nunca a
#                    recíproca -- evita qualquer disputa.
# ===========================================================================

_em_call = threading.Event()
_eventos_corpo: "queue.Queue[str]" = queue.Queue()
_consciencia_callback: Callable[[str], None] | None = None

# Última leitura de safety vista por _detectar_transicao_seguranca -- só
# emite impulso na TRANSIÇÃO (ex: virou emergency_stop AGORA), não a cada
# tick que ela já está em emergency_stop -- senão vira ruído repetitivo.
_ultimo_estado_seguranca: dict | None = None

HEARTBEAT_INTERVALO_S = float(os.environ.get("EVA_ROBOT_HEARTBEAT_S", "0.5"))
BATERIA_MINIMA_SEGURA_V = float(os.environ.get("EVA_ROBOT_BATERIA_MINIMA", "6.6"))
# A cada quantos ticks de heartbeat o estado completo é buscado só pra
# checar transição de segurança (ver _ciclo_heartbeat). Com o default de
# HEARTBEAT_INTERVALO_S, isso fica em poucos segundos -- bem abaixo dos
# 300s de antes (quando essa checagem só vivia em _ciclo_iniciativa),
# sem virar um estado() a cada 0.5-1s.
CHECAGEM_SEGURANCA_A_CADA_N_TICKS = int(os.environ.get("EVA_ROBOT_CHECAGEM_SEGURANCA_TICKS", "4"))

# Idade máxima de um quadro de câmera pra ele ainda valer como "o que ela
# está vendo agora" -- ver obter_quadro_camera().
IDADE_MAXIMA_QUADRO_S = float(os.environ.get("EVA_ROBOT_QUADRO_MAX_IDADE_S", "5.0"))

# Ângulo pro qual robo_destravar_braco recolhe o cotovelo. Valor FIXO no
# código, nunca parâmetro de ferramenta -- ver a docstring de lá.
COTOVELO_NEUTRO = int(os.environ.get("EVA_ROBOT_COTOVELO_NEUTRO", "120"))

# Sinal do delta de pitch que LEVANTA a mira. arm_controller.look_up()
# assume que ângulo menor é pra cima (faz current - graus), mas isso
# nunca foi conferido contra imagem. Compare 20-pitch-040.jpg com
# 22-pitch-110.jpg da calibração: se a 040 mostra mais teto, este valor
# está certo; se mostra mais chão, inverta pra 20 (e look_up/look_down
# estão trocados desde sempre).
PITCH_DELTA_CIMA = int(os.environ.get("EVA_ROBOT_PITCH_DELTA_CIMA", "-20"))

# ===========================================================================
# EIXOS DO CORPO -- confirmados por imagem (ferramentas/testar_cabeca.py),
# não deduzidos do desenho mecânico.
#
#   0 yaw     0..90    gira a BASE          -- horizontal
#   1 pitch  40..110   levanta/abaixa       -- VERTICAL, é o único
#   3 cabeça  0..117   gira só a CÂMERA     -- horizontal (pan, não tilt)
#
# Duas consequências que mudam o desenho de tudo aqui embaixo:
#
# 1) Há DOIS eixos horizontais e UM vertical. Yaw + cabeça somam ~207° de
#    pan, não os 90° do yaw sozinho -- "olhar em volta" é bem menos
#    limitado do que o curso da base sugere.
# 2) Não existe roll. Nada inclina a imagem de lado, então não existe
#    gesto de "inclinar a cabeça em dúvida" -- inventar um alias que faz
#    outra coisa seria mentir pra ela sobre o próprio corpo.
#
# O yaw parar em 90 (e não 180, que o servo alcança) é o flat CSI da
# PiCam: ele sobe pelo braço até o Pi e não tem folga pra torcer além
# disso. Aumentar o limite rasga o cabo.
#
# Este dicionário NÃO valida nada -- quem valida é safety.py, no Pi,
# sempre. Ele existe pra os gestos serem PLANEJADOS dentro do curso que
# existe, em vez de mandarem ângulo que vai ser clampado em silêncio e
# virar "gesto" sem movimento nenhum.
# ===========================================================================

_CURSO_EIXO: dict[int, tuple[int, int]] = {0: (0, 90), 1: (40, 110), 3: (0, 117)}
_NOME_CANAL = {0: "yaw", 1: "pitch", 3: "cabeca"}

# Yaw que aponta pra frente do carro. robo_olhar_em_volta sempre volta
# pra cá no fim: deixar a base parada num extremo mantém o flat CSI
# torcido por tempo indeterminado.
YAW_FRENTE = int(os.environ.get("EVA_ROBOT_YAW_FRENTE", "90"))


def _iniciar_thread_robo() -> None:
    """Sobe a thread com o event loop dedicado, uma vez só, preguiçoso --
    só na primeira chamada de ferramenta de robô de verdade.

    BUG REAL corrigido aqui: com duas ferramentas robo_* do mesmo plano
    rodando em paralelo (ver orchestrator._executar_ferramentas), a
    segunda chamada via `if _thread is not None: return` sempre vencia a
    corrida ANTES de `_pronto.wait()` -- ela via a thread já criada pela
    primeira (só a criação, não a conexão) e retornava na hora, sem
    esperar nada. `_chamar()` então via `_cliente`/`_loop` ainda None e
    falhava com "thread de conexão não iniciou a tempo", mesmo com a
    outra ferramenta do mesmo turno conectando com sucesso um instante
    depois. Agora o lock protege só a CRIAÇÃO (uma vez só, como antes) e
    `_pronto.wait()` roda pra QUALQUER chamador, criador ou não -- quem
    chega depois espera a mesma inicialização em vez de desistir cedo.
    """
    global _thread
    with _lock_inicio:
        if _thread is None:
            def rodar_loop():
                global _cliente, _cliente_video, _loop
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                host = os.environ.get("EVA_ROBOT_HOST", "127.0.0.1")
                porta = int(os.environ.get("EVA_ROBOT_PORT", "5000"))
                porta_video = int(os.environ.get("EVA_ROBOT_VIDEO_PORT", "8000"))
                _cliente = ClienteRobo(host=host, port=porta, fonte="eva")
                _cliente_video = ClienteVideoRobo(host=host, port=porta_video)
                _pronto.set()
                # Sempre agendados, independente do valor inicial de
                # _iniciativa_ativa -- controle de liga/desliga é checado a
                # cada ciclo (ver definir_iniciativa), não só na
                # inicialização.
                _criar_tarefa_permanente("heartbeat", _ciclo_heartbeat)
                _criar_tarefa_permanente("iniciativa", _ciclo_iniciativa)
                _criar_tarefa_permanente("video", _cliente_video.rodar)
                _loop.run_forever()

            _thread = threading.Thread(target=rodar_loop, daemon=True, name="eva-robot")
            _thread.start()

    # Fora do lock de propósito: se ficasse dentro, um segundo chamador
    # ficaria preso esperando o lock (que só solta depois do rodar_loop
    # arrancar) em vez de simplesmente esperar _pronto -- funcionalmente
    # parecido, mas prende o lock por mais tempo que o necessário à toa.
    _pronto.wait(timeout=5)


_tarefas_fundo: dict[str, asyncio.Task] = {}


def _criar_tarefa_permanente(nome: str, fabrica) -> None:
    """Cria uma task de fundo GUARDANDO a referência, e a recria se ela
    morrer.

    BUG REAL fechado aqui: asyncio guarda só referência FRACA às tasks
    (Task.__init__ registra em _all_tasks, que é um WeakSet). Uma task
    criada com create_task() e cujo retorno é descartado pode ser
    coletada pelo GC no meio da execução. Foi exatamente o que aconteceu
    com a task de vídeo, 42 segundos depois de subir:

        Task was destroyed but it is pending!
        task: <Task pending coro=<ClienteVideoRobo.rodar() running at
              robot_video_client.py:64> ...>

    Depois disso não houve mais NENHUMA linha [robo-video] no log
    inteiro: a conexão de vídeo morreu em silêncio e `ultimo_frame`
    congelou no último quadro recebido. robo_ver/robo_olhar continuaram
    respondendo normalmente, descrevendo uma cena de minutos antes, sem
    erro nenhum -- o pior tipo de falha, a que parece sucesso.

    A referência forte em _tarefas_fundo fecha a coleta pelo GC. O
    done_callback cobre o resto: se o laço terminar por exceção não
    tratada, ele volta, e a linha no terminal registra que voltou --
    nenhum destes laços pode sumir sem deixar rastro."""
    def _refazer(tarefa: asyncio.Task) -> None:
        if tarefa.cancelled():
            return
        print(f"[robo] laço de fundo '{nome}' terminou sozinho "
              f"({tarefa.exception()!r}) -- recriando")
        _criar_tarefa_permanente(nome, fabrica)

    tarefa = _loop.create_task(fabrica(), name=f"eva-robo-{nome}")
    tarefa.add_done_callback(_refazer)
    _tarefas_fundo[nome] = tarefa


def _chamar(coro_fn, *args, timeout: float = 10.0, **kwargs) -> dict:
    """Ponte síncrona->assíncrona, mesmo padrão de minecraft_tools._chamar:
    roda a corrotina no loop dedicado (rodando numa OUTRA thread) a
    partir de QUALQUER thread chamadora, espera o resultado com timeout.
    Nunca levanta exceção."""
    _iniciar_thread_robo()
    if _cliente is None or _loop is None:
        return {"erro": "robo_indisponivel", "detalhe": "thread de conexão não iniciou a tempo"}
    try:
        futuro = asyncio.run_coroutine_threadsafe(coro_fn(_cliente, *args, **kwargs), _loop)
        return futuro.result(timeout=timeout + 2)
    except Exception as e:
        return {"erro": "robo_falha", "detalhe": str(e)[:200]}


def _desembrulhar(resultado_bruto: dict) -> dict:
    """A resposta crua do servidor vem como {"ok":..., "cmd":...,
    "estado"/"detalhe":...}. Ferramenta devolve algo mais direto pro
    modelo: em sucesso de get_state, só o "estado"; em falha, sempre
    {"erro": ...} (convenção do projeto, ver registry.py)."""
    if not isinstance(resultado_bruto, dict):
        return {"erro": "resposta_invalida"}
    if resultado_bruto.get("erro") and "ok" not in resultado_bruto:
        return resultado_bruto  # já veio no formato de erro de _chamar
    if not resultado_bruto.get("ok", False):
        return {"erro": resultado_bruto.get("erro", "falha_desconhecida"),
                "detalhe": resultado_bruto.get("detalhe")}
    if "estado" in resultado_bruto:
        return resultado_bruto["estado"]
    return {k: v for k, v in resultado_bruto.items() if k not in ("ok", "cmd", "seq")}


# ===========================================================================
# EVENTOS CORPORAIS -- o que entra na fila pra Consciência, ver bridge_client.
# Critério: só o que é sobre SEGURANÇA/ARBITRAGEM do corpo -- nunca
# movimento rotineiro bem-sucedido (isso ela pode esquecer, é o combinado).
# ===========================================================================

def _emitir_evento_corpo(descricao: str) -> None:
    try:
        _eventos_corpo.put_nowait(descricao)
    except Exception:
        pass  # fila cheia é cenário absurdo aqui (consumida a cada poucos
              # segundos); melhor perder um evento que travar quem chamou.


def _descrever_recusa(erro: str, detalhe: str | None) -> str | None:
    """Traduz o código/motivo cru do servidor pra uma frase que vale virar
    impulso -- só quando é sobre o CORPO em si (segurança física,
    arbitragem de controle). Falha de rede/protocolo (robo_desconectado,
    comando_expirado, json_invalido...) fica de fora de propósito: não é
    algo que ela "sentiu", é infraestrutura, e virar impulso disso seria
    ruído sem nenhum valor de auto-consciência.

    Motivos vêm crus de safety.py (ver eva_command_server.py):
    "EMERGENCY STOP ativo", "Obstáculo muito/próximo (Xcm)",
    "Bateria crítica/baixa (XV)" -- mais o código "manual_ativo" da
    arbitragem em si (eva_command_server._processar_mensagem).
    """
    texto = f"{erro} {detalhe or ''}"
    if "EMERGENCY STOP" in texto:
        return f"tentei me mover e fui recusada: emergency stop está ativo ({detalhe or erro})"
    if "Obstáculo" in texto:
        return f"tentei me mover e fui recusada por causa de um obstáculo: {detalhe or erro}"
    if "Bateria" in texto:
        return f"tentei me mover e fui recusada por causa da bateria: {detalhe or erro}"
    if erro == "manual_ativo":
        return "tentei me mover, mas alguém está no controle manual agora"
    return None


def _talvez_emitir_recusa(resultado: dict) -> dict:
    """Passa o resultado já desembrulhado adiante sem alterar nada --
    só observa de passagem se vale emitir evento. Usado em robo_mover e
    robo_olhar (as duas ferramentas que de fato tentam mexer o corpo);
    robo_estado é só leitura, robo_parar não é recusado por segurança
    (é o próprio botão de segurança)."""
    if isinstance(resultado, dict) and resultado.get("erro"):
        descricao = _descrever_recusa(resultado["erro"], resultado.get("detalhe"))
        if descricao:
            _emitir_evento_corpo(descricao)
    return resultado


# ------------------------------------------------------------- ferramentas


@registro.adicionar(
    "robo_estado",
    "Estado atual do corpo físico do robô: modo, câmera ativa, ângulos do "
    "braço/cabeça, e segurança (obstáculo, bateria, emergency stop). "
    "Consulte antes de mover -- sem isso você está decidindo às cegas, e "
    "aqui errar bate em coisa de verdade, não é um jogo.",
)
def robo_estado() -> dict:
    async def _fn(cliente: ClienteRobo):
        return await cliente.estado()
    return _desembrulhar(_chamar(_fn))


@registro.adicionar(
    "robo_mover",
    "Move o robô na direção dada. Anda por cerca de 1 segundo e PARA "
    "SOZINHA -- se quiser continuar andando, chame de novo. "
    "vx=frente(+)/trás(-), vy=lateral direita(+)/esquerda(-), "
    "vz=giro horário(+)/anti-horário(-), todos de -1.0 a 1.0 -- use "
    "valores pequenos (0.2-0.4) a menos que tenha certeza do caminho. O "
    "resultado já vem com distancia_obstaculo_cm (leitura atual do "
    "sensor) -- confira antes de chamar de novo, principalmente se o "
    "valor estiver baixo. O servidor recusa o comando se detectar "
    "obstáculo, bateria crítica ou emergency stop ativo -- se vier erro, "
    "NÃO insista no mesmo comando, chame robo_estado pra entender o "
    "motivo antes de tentar de novo.",
    {"vx": "-1.0 a 1.0 (frente/trás)", "vy": "-1.0 a 1.0 (lateral)",
     "vz": "-1.0 a 1.0 (giro)"},
)
def robo_mover(vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> dict:
    async def _fn(cliente: ClienteRobo):
        return await cliente.mover(vx=vx, vy=vy, vz=vz)
    return _talvez_emitir_recusa(_desembrulhar(_chamar(_fn)))


def _ver_agora(prompt: str | None = None) -> dict:
    """Captura o quadro mais recente da câmera do robô e pede pro modelo
    de visão descrever. Usado por robo_ver e por robo_olhar (depois de
    mover a cabeça). Nunca levanta exceção -- erro de captura ou de
    visão vira {"erro": ...}, mesma convenção do resto do projeto."""
    from ..vision.minicpm import ErroVisao, PROMPT_CENA_ROBO
    _iniciar_thread_robo()
    quadro = obter_quadro_camera()
    if quadro is None:
        # Distinguir "nunca chegou quadro" de "chegou, mas é velho" --
        # são causas diferentes e levam a ações diferentes (esperar a
        # conexão subir vs. investigar por que a conexão caiu). Antes as
        # duas caíam no mesmo "sem_video", e o segundo caso nem chegava
        # aqui: o quadro velho era descrito como se fosse atual.
        if _cliente_video is not None and _cliente_video.ultimo_frame is not None:
            idade = time.monotonic() - _cliente_video.ultimo_frame_ts
            return {"erro": "video_parado",
                    "detalhe": f"o último quadro tem {idade:.0f}s -- a conexão "
                               f"de vídeo com o robô caiu"}
        return {"erro": "sem_video",
                "detalhe": "ainda sem quadro da câmera -- conexão de vídeo pode não ter estabelecido ainda"}
    try:
        descricao = _cliente_visao_robo().analisar([quadro], prompt or PROMPT_CENA_ROBO)
        return {"descricao_cena": descricao}
    except ErroVisao as e:
        return {"erro": "visao_falhou", "detalhe": str(e)[:200]}
    except Exception as e:
        return {"erro": "visao_falhou", "detalhe": str(e)[:200]}


@registro.adicionar(
    "robo_ver",
    "Olha pela câmera do robô AGORA e descreve o que vê -- inclui "
    "pessoas, se houver. Use quando quiser saber o que tem à frente sem "
    "mover a cabeça nem o corpo. Se quiser olhar pra outra direção "
    "primeiro, use robo_olhar (que já descreve sozinho depois de mover, "
    "não precisa chamar os dois).",
)
def robo_ver() -> dict:
    return _ver_agora()


def _aguardar_quadro_novo(timeout_s: float = 4.0) -> bool:
    """Espera chegar um quadro que seja POSTERIOR ao movimento/troca que
    acabou de acontecer.

    Os dois lados guardam o quadro velho de propósito: no Pi,
    camera_manager.get_frame() devolve last_good_frame quando ainda não
    há quadro novo; aqui, `ultimo_frame` só é substituído quando um
    quadro novo chega de verdade. Isso é o comportamento certo pra
    soluço momentâneo, e o errado logo depois de uma troca de câmera
    (onde o quadro velho é literalmente de OUTRA câmera) ou de um
    movimento suave (que leva quase um segundo, tempo em que o stream de
    15fps continua entregando quadros da posição anterior).

    Roda na thread que chamou a ferramenta, não no loop do robô -- é só
    poll de um float, nada que precise do event loop."""
    if _cliente_video is None:
        return False
    marco = _cliente_video.ultimo_frame_ts
    limite = time.monotonic() + timeout_s
    while time.monotonic() < limite:
        if _cliente_video.ultimo_frame_ts > marco:
            return True
        time.sleep(0.05)
    return False


@registro.adicionar(
    "robo_olhar",
    "Aponta a câmera da cabeça do robô e já descreve o que ela vê na "
    "posição nova -- inclui pessoas, se houver. yaw: gira a base, muda a "
    "direção horizontal (0-90). cabeca: gira só a câmera, TAMBÉM "
    "horizontal (0-117) -- é mais rápido e menos forçado que girar a base, "
    "prefira este pra ajustes pequenos de lado. pitch: levanta e abaixa a "
    "mira, é o único eixo vertical (40-110). Nada inclina a imagem de "
    "lado, isso o corpo não faz. Se a câmera ativa for a 'usb' nada disso "
    "muda o que você vê -- ela é fixa no corpo; troque pra 'picam' antes "
    "com robo_trocar_camera. Demora alguns segundos (o movimento é "
    "rápido, descrever a cena não) -- é esperado.",
    {"yaw": "graus, 0-90, opcional -- direção horizontal, gira a base",
     "cabeca": "graus, 0-117, opcional -- direção horizontal, gira só a câmera",
     "pitch": "graus, 40-110, opcional -- direção vertical"},
)
def robo_olhar(yaw: int | None = None, pitch: int | None = None,
               cabeca: int | None = None) -> dict:
    async def _fn(cliente: ClienteRobo):
        return await cliente.olhar(yaw=yaw, pitch=pitch, cabeca=cabeca)
    resultado = _talvez_emitir_recusa(_desembrulhar(_chamar(_fn, timeout=12.0)))
    if isinstance(resultado, dict) and not resultado.get("erro"):
        # Espera o quadro NOVO antes de descrever -- sem isso, com
        # smooth=True, a descrição sai da posição ANTERIOR e ela conclui
        # que a cabeça não se moveu.
        _aguardar_quadro_novo(timeout_s=2.0)
        # Erro de visão aqui não vira erro do robo_olhar -- o movimento
        # em si já teve sucesso; câmera/modelo de visão fora do ar não
        # deveria fazer parecer que a cabeça não se moveu.
        visao = _ver_agora()
        if visao.get("descricao_cena"):
            resultado["descricao_cena"] = visao["descricao_cena"]
    return resultado


@registro.adicionar(
    "robo_trocar_camera",
    "Troca qual câmera do robô está ativa -- ele tem DUAS, e só uma "
    "funciona por vez. 'usb': fixa no corpo, aponta pra frente, é a de "
    "navegação. 'picam': montada na ponta do braço, acompanha o "
    "movimento dos servos -- é a única que você consegue apontar, então é "
    "a que serve pra olhar pra alguém ou pra algo específico. Depois de "
    "trocar, robo_ver e robo_olhar passam a descrever o que a câmera nova "
    "vê.",
    {"tipo": "'usb', 'picam', ou omita pra alternar"},
)
def robo_trocar_camera(tipo: str | None = None) -> dict:
    async def _fn(cliente: ClienteRobo):
        return await cliente.trocar_camera(tipo)
    resultado = _desembrulhar(_chamar(_fn, timeout=18.0))
    if isinstance(resultado, dict) and not resultado.get("erro"):
        if not _aguardar_quadro_novo():
            resultado["aviso"] = ("a câmera trocou, mas nenhum quadro novo "
                                  "chegou ainda -- espere antes de descrever a cena")
    return resultado


@registro.adicionar(
    "robo_olhar_em_volta",
    "Varre o entorno parando em quatro posições e descreve o que a câmera "
    "vê em cada uma, depois volta a olhar pra frente. Usa os dois eixos "
    "horizontais (base e câmera) pra cobrir o máximo que o corpo alcança "
    "-- que não é uma volta completa, a base não gira 360. Demora "
    "bastante (quatro descrições de cena): use quando quiser mesmo um "
    "panorama, não pra conferir uma coisa só -- pra isso use robo_olhar "
    "ou robo_ver.",
)
def robo_olhar_em_volta() -> dict:
    # A base fica nos extremos e a câmera varre dentro de cada um: mover
    # a base é o caro (flat CSI torcendo), mover só a câmera é barato.
    paradas = [
        {"yaw": 90, "cabeca": 117},
        {"yaw": 90, "cabeca": 30},
        {"yaw": 0, "cabeca": 90},
        {"yaw": 0, "cabeca": 20},
    ]
    vistas = []
    for parada in paradas:
        async def _fn(cliente: ClienteRobo, alvo=parada):
            # alvo como default de propósito: sem isso o closure leria
            # `parada` no momento da execução, e todas as chamadas usariam
            # a última posição da lista.
            return await cliente.olhar(smooth=True, **alvo)
        r = _desembrulhar(_chamar(_fn, timeout=12.0))
        if r.get("erro"):
            _talvez_emitir_recusa(r)
            break
        _aguardar_quadro_novo(timeout_s=2.0)
        visao = _ver_agora()
        if visao.get("descricao_cena"):
            vistas.append({**parada, "descricao_cena": visao["descricao_cena"]})

    # Volta pra frente aconteça o que acontecer -- inclusive se a
    # varredura foi interrompida no meio por recusa.
    async def _voltar(cliente: ClienteRobo):
        return await cliente.olhar(yaw=YAW_FRENTE, smooth=True)
    _chamar(_voltar, timeout=12.0)

    if not vistas:
        return {"erro": "varredura_falhou",
                "detalhe": "nenhuma posição rendeu descrição"}
    return {"vistas": vistas}


# --------------------------------------------------------------- gestos
#
# nome: (tipo, canal, amplitude_graus, repeticoes)
#
# "sim" é pitch e "nao" é a cabeça, não o contrário: o canal 3 é PAN
# (gira a câmera pros lados) e o pitch é o único eixo vertical. Trocar os
# dois produziria a câmera subindo e descendo enquanto encara o mesmo
# ponto -- um elevador, não um aceno.
#
# Nada de "inclinar a cabeça em dúvida": não existe eixo de roll neste
# corpo, e um gesto que promete uma coisa e faz outra é pior que gesto
# nenhum. "espiar" usa o pitch e é honesto sobre o que é.
_GESTOS: dict[str, tuple[str, int, int, int]] = {
    "sim": ("oscilar", 1, 12, 2),
    "nao": ("oscilar", 3, 15, 2),
    "espiar": ("segurar", 1, PITCH_DELTA_CIMA, 1),
}


def _centro_viavel(canal: int, base: int, amplitude: int) -> int | None:
    """Onde a oscilação cabe inteira dentro do curso do eixo.

    Sem isso, um gesto perto de um batente teria metade dos passos
    clampados em silêncio pelo ArmController e sairia como: parado,
    vira, parado, vira. O centro é deslocado só o necessário; o gesto
    termina voltando pro ponto de onde saiu de qualquer forma."""
    lo, hi = _CURSO_EIXO[canal]
    a = abs(amplitude)
    if hi - lo < 2 * a:
        return None
    return max(lo + a, min(hi - a, base))


async def _executar_gesto(cliente: ClienteRobo, tipo: str, canal: int,
                          amplitude: int, repeticoes: int) -> dict:
    nome_param = _NOME_CANAL[canal]
    lo, hi = _CURSO_EIXO[canal]

    r = await cliente.estado()
    if not r.get("ok"):
        return r
    angulos = ((r.get("estado") or {}).get("arm") or {}).get("angles") or {}
    # ArmController.get_status() usa chaves int; o JSON de volta as
    # transforma em string. Aceita as duas formas.
    base = int(angulos.get(str(canal), angulos.get(canal, 90)))

    if tipo == "oscilar":
        centro = _centro_viavel(canal, base, amplitude)
        if centro is None:
            return {"ok": False, "erro": "sem_curso",
                    "detalhe": f"{nome_param} não tem {2 * abs(amplitude)}° livres"}
        passos = [centro]
        for _ in range(repeticoes):
            passos += [centro - abs(amplitude), centro + abs(amplitude)]
        passos.append(base)
    else:  # "segurar" -- vai até o alvo, fica, e volta
        alvo = max(lo, min(hi, base + amplitude))
        if alvo == base:
            alvo = max(lo, min(hi, base - amplitude))
        if alvo == base:
            return {"ok": False, "erro": "sem_curso",
                    "detalhe": f"{nome_param} já está no batente ({base}°)"}
        passos = [alvo] * 4 + [base]  # repetir o alvo é o que segura a pose

    recusa = None
    for angulo in passos:
        # smooth=False de propósito: cada passo vira UMA escrita de PWM e
        # devolve na hora. Com smooth=True o _command_loop do Pi trava
        # ~0.9s por passo e a sequência inteira engasgaria o heartbeat.
        passo = await cliente.olhar(**{nome_param: int(angulo)}, smooth=False)
        if not passo.get("ok"):
            recusa = passo.get("detalhe") or passo.get("erro")
            break
        await asyncio.sleep(0.18)

    if recusa:
        await cliente.olhar(**{nome_param: base}, smooth=False)
        return {"ok": False, "erro": "gesto_recusado", "detalhe": recusa}
    return {"ok": True, "cmd": "gesto"}


@registro.adicionar(
    "robo_gesto",
    "Faz um gesto físico com o corpo do robô e volta pra posição "
    "anterior. 'sim': aceno afirmativo. 'nao': nega com a câmera. "
    "'espiar': levanta a mira e segura, como quem tenta ver melhor. Não "
    "descreve cena nenhuma -- é só o gesto, pra quando reagir com o corpo "
    "diz mais que reagir com palavra. Só funciona se alguém estiver vendo "
    "o robô de verdade.",
    {"gesto": "sim, nao ou espiar"},
)
def robo_gesto(gesto: str) -> dict:
    entrada = _GESTOS.get((gesto or "").strip().lower())
    if entrada is None:
        return {"erro": "gesto_desconhecido",
                "detalhe": f"conhecidos: {', '.join(_GESTOS)}"}
    tipo, canal, amplitude, repeticoes = entrada

    async def _fn(cliente: ClienteRobo):
        return await _executar_gesto(cliente, tipo, canal, amplitude, repeticoes)

    return _talvez_emitir_recusa(_desembrulhar(_chamar(_fn, timeout=15.0)))


@registro.adicionar(
    "robo_destravar_braco",
    "Recolhe o cotovelo pra posição neutra. Use SÓ quando robo_olhar ou "
    "robo_gesto forem recusados com 'cotovelo em posição crítica' -- acima "
    "de 160 graus o cotovelo trava todos os outros eixos por segurança, e "
    "este é o único comando que sai desse estado.",
)
def robo_destravar_braco() -> dict:
    # Ângulo FIXO, nunca parâmetro. O cotovelo é o único eixo capaz de
    # travar todos os outros (safety.validate_servo_command, regra 1), e
    # ele está de fora do robo_olhar justamente por isso. Sem ESTA saída,
    # porém, um cotovelo deixado em 165 pelo gamepad deixaria ela cega e
    # paralisada pra sempre: toda chamada voltaria "outros eixos
    # travados" e nenhuma ferramenta dela alcançaria o canal 2 pra
    # desfazer. Uma saída, com destino fixo, resolve sem devolver o
    # controle do eixo perigoso.
    async def _fn(cliente: ClienteRobo):
        return await cliente.olhar(cotovelo=COTOVELO_NEUTRO, smooth=True)
    return _desembrulhar(_chamar(_fn, timeout=15.0))


@registro.adicionar(
    "robo_parar",
    "Para o robô imediatamente. Sempre seguro chamar, mesmo sem saber o "
    "estado atual -- use se algo parecer errado ou incerto.",
)
def robo_parar() -> dict:
    async def _fn(cliente: ClienteRobo):
        return await cliente.parar()
    return _desembrulhar(_chamar(_fn))


# ===========================================================================
# HEARTBEAT -- alimenta o watchdog do robô enquanto a conexão está de pé,
# independente de estar se movendo ou não. Ver safety.py: watchdog sem
# heartbeat regular aciona emergency stop sozinho (por bom motivo -- se
# ninguém está de fato supervisionando, parar é o padrão seguro).
# ===========================================================================

# Depois de tantas falhas seguidas, para de logar cada tick. O robô
# desligado é o caso NORMAL (EVA roda sem ele o tempo todo), e com
# HEARTBEAT_INTERVALO_S de 0.5-1s isso rendia uma linha por segundo pra
# sempre -- 349 linhas numa call de teste, afogando [visao], [eva] e
# [consciencia] no terminal. Antes de silenciar, ainda imprime as
# primeiras: essas SIM importam, é quando a queda acabou de acontecer.
FALHAS_ATE_SILENCIAR = int(os.environ.get("EVA_ROBOT_HEARTBEAT_FALHAS_LOG", "5"))

# Já silenciado, ainda solta uma linha esporádica -- sem isso não dá pra
# distinguir "robô fora do ar" de "o ciclo de heartbeat morreu", e essa
# distinção é justamente o que o DIAGNÓSTICO acima precisava enxergar.
FALHAS_LEMBRETE_A_CADA = int(os.environ.get("EVA_ROBOT_HEARTBEAT_LEMBRETE", "300"))


def _deve_logar_falha(n: int) -> bool:
    """Primeiras N falhas sempre; depois só de FALHAS_LEMBRETE_A_CADA em
    FALHAS_LEMBRETE_A_CADA. A contagem consecutiva continua subindo e
    aparecendo na linha, então o diagnóstico não perde informação -- só
    para de repetir a mesma linha centenas de vezes."""
    if n <= FALHAS_ATE_SILENCIAR:
        return True
    return n % FALHAS_LEMBRETE_A_CADA == 0


async def _ciclo_heartbeat() -> None:
    """Roda pra sempre em segundo plano, independente do que estiver
    acontecendo no resto do sistema (inclusive enquanto ninguém está
    chamando ferramenta nenhuma -- é exatamente pra isso que existe).

    CORRIGIDO (achado em uso real): antes só tentava heartbeat
    `if _cliente.conectado`. Se qualquer soluço de rede derrubasse a
    conexão momentaneamente (ClienteRobo._enviar marca conectado=False
    em qualquer falha), esse `if` passava a bloquear PRA SEMPRE -- o
    ciclo continuava rodando, mas nunca mais tentava mandar nada, porque
    nada aqui reconectava sozinho. Só um comando de ferramenta chamado
    por você (_chamar -> _enviar -> vê not conectado -> conectar())
    reconectava. Resultado: um soluço de rede qualquer, mesmo de menos
    de 1 segundo, podia calar o heartbeat de vez, e o watchdog do robô
    estourava minutos depois sem nenhum aviso -- exatamente o log real
    que motivou essa correção (heartbeat "sumiu" entre duas chamadas de
    ferramenta feitas à mão, sem nenhum Ctrl+C nem erro visível).

    Agora tenta SEMPRE, a cada ciclo, goste ou não do que
    `_cliente.conectado` diz -- é `cliente.heartbeat()` (via `_enviar`)
    quem decide se precisa reconectar primeiro, não este loop.

    CORRIGIDO (segundo achado, ligado à revisão de segurança do
    safety.py): _detectar_transicao_seguranca() antes só rodava dentro
    de _ciclo_iniciativa, no cadence de INICIATIVA_INTERVALO_S (padrão
    300s) -- e só quando havia call ativa E iniciativa ligada. Com o
    fix em safety.py que agora aciona emergency_stop de verdade a partir
    do monitoramento contínuo (não só no instante de um comando `drive`),
    uma transição de segurança pode acontecer a qualquer momento, mesmo
    sem ela ter tentado se mover -- e o comentário sobre isso ("acabei de
    entrar em emergency stop") podia demorar até 5min pra sair, porque
    dependia do próximo tick lento de iniciativa. Este loop já roda a
    cada HEARTBEAT_INTERVALO_S (0.5-1s) incondicionalmente, então é o
    lugar certo pra essa checagem -- não a cada heartbeat (isso trocaria
    "sinal de vida leve" por "polling pesado" de estado completo), mas a
    cada CHECAGEM_SEGURANCA_A_CADA_N_TICKS ticks, o que já derruba o
    atraso de 300s pra poucos segundos sem sobrecarregar a conexão.

    DIAGNÓSTICO (achado em uso real, ainda não totalmente explicado):
    apareceu pelo menos uma vez watchdog timeout de 16.2s durante uma
    pausa de confirmação do testar_robo.py, mesmo com este ciclo
    testado e confirmado rodando a cada ~1s sob o mesmo padrão de
    threading/asyncio (em Linux -- 7 heartbeats em 8s de pausa
    simulada, gap máximo 1.01s, sem estagnar). Não reproduzi a causa
    exata -- pode ter sido só o tempo real de leitura/confirmação, ou
    algo específico do Windows/ProactorEventLoop que não aparece em
    teste automatizado neste ambiente. Por isso: (1) intervalo default
    caiu de 1.0s pra 0.5s (mais margem dentro do WATCHDOG_TIMEOUT de
    5s -- 10x em vez de 5x); (2) toda falha de heartbeat agora aparece
    no terminal, com contagem de falhas consecutivas -- se isso
    disparar de novo, o terminal do PC vai mostrar ONDE parou, em vez
    de só o resultado final (o timeout no Pi)."""
    falhas_consecutivas = 0
    ticks = 0
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVALO_S)
        ticks += 1
        if _cliente is not None:
            try:
                r = await _cliente.heartbeat()
                if r.get("ok"):
                    if falhas_consecutivas >= FALHAS_ATE_SILENCIAR:
                        # Só avisa da volta se ele tinha sumido de fato --
                        # senão um soluço de rede de 1 tick viraria duas
                        # linhas de log em vez de zero.
                        print(f"[heartbeat-robo] robô respondeu de novo "
                              f"(depois de {falhas_consecutivas} falhas)")
                    falhas_consecutivas = 0
                else:
                    falhas_consecutivas += 1
                    if _deve_logar_falha(falhas_consecutivas):
                        print(f"[heartbeat-robo] falhou "
                              f"(consecutiva #{falhas_consecutivas}): {r}")
            except Exception as e:
                falhas_consecutivas += 1
                if _deve_logar_falha(falhas_consecutivas):
                    print(f"[heartbeat-robo] exceção "
                          f"(consecutiva #{falhas_consecutivas}): {e}")

            # Checagem de segurança em separado do heartbeat em si --
            # heartbeat continua leve e roda todo tick; isto aqui só
            # busca o estado completo a cada N ticks (ver constante),
            # exatamente pra notar transição de segurança rápido sem
            # virar polling pesado a cada 0.5-1s.
            if ticks % CHECAGEM_SEGURANCA_A_CADA_N_TICKS == 0:
                try:
                    resposta_estado = await _cliente.estado(background=True)
                    if resposta_estado.get("ok"):
                        _detectar_transicao_seguranca(
                            resposta_estado.get("estado", {}).get("safety") or {}
                        )
                except Exception as e:
                    # Mesmo critério do heartbeat: com o robô fora do ar
                    # esta checagem falha junto, e logar as duas por tick
                    # dobrava o ruído.
                    if _deve_logar_falha(falhas_consecutivas or 1):
                        print(f"[heartbeat-robo] checagem de segurança falhou: {e}")


# ===========================================================================
# INICIATIVA -- ela decide sozinha se quer se mexer, sem ninguém pedir.
# ===========================================================================
#
# Mesmo espírito de minecraft_tools._ciclo_iniciativa: periodicamente, sem
# ninguém pedir nada, ela pode decidir se mexer -- ou decidir que não
# quer, que também é resposta válida.
#
# DESLIGADO POR PADRÃO -- mover um robô físico sozinha é mudança de
# comportamento grande o bastante pra merecer opt-in explícito
# (EVA_ROBOT_INICIATIVA=1), não vir ativo sem avisar.
#
# Portões mais rígidos que os do Minecraft, porque aqui o corpo é físico:
# bateria, watchdog, emergency_stop, e sobretudo um teto de DURAÇÃO por
# movimento (DURACAO_MAX_MOVIMENTO_S) -- o modelo nunca decide "por
# quanto tempo" andar, só a direção; quem decide a duração é este código,
# de propósito.

_iniciativa_ativa = os.environ.get("EVA_ROBOT_INICIATIVA", "0") == "1"
# 5 min padrão -- generoso de propósito, mesmo valor do Minecraft.
INICIATIVA_INTERVALO_S = int(os.environ.get("EVA_ROBOT_INICIATIVA_INTERVALO", "300"))
DURACAO_MAX_MOVIMENTO_S = float(os.environ.get("EVA_ROBOT_INICIATIVA_DURACAO_MAX", "1.5"))
VELOCIDADE_MAX_INICIATIVA = float(os.environ.get("EVA_ROBOT_INICIATIVA_VELOCIDADE_MAX", "0.4"))

PROMPT_INICIATIVA = """Você está com um corpo físico agora -- um robô com rodas, câmera e cabeça móvel -- sozinha, sem ninguém pedindo nada. Você pode escolher se mexer um pouco (olhar em volta ou andar um pouco), comentar algo sobre o que percebeu, as duas coisas, ou nenhuma.

Estado atual do robô (real, agora):
{estado}

Decida: quer se mexer, por conta própria? Se sim, o quê -- olhar (ação "olhar", yaw/pitch em graus) ou andar um pouco (ação "mover", vx/vy/vz pequenos, tipo 0.3)? Independente disso, tem algo que valeria comentar em voz alta agora (bateria, algo notado, o próprio movimento)? Se não tiver nada que valha a pena, "comentario" fica null -- não precisa preencher.

Responda APENAS um JSON, sem mais nada:
{{"agir": true, "acao": "olhar", "yaw": 60, "pitch": 100, "comentario": null}}
ou
{{"agir": true, "acao": "mover", "vx": 0.3, "vy": 0.0, "vz": 0.0, "comentario": "vou dar uma olhada aí"}}
ou
{{"agir": false, "comentario": "a bateria está ficando baixa, vale avisar"}}
ou
{{"agir": false, "comentario": null}}"""


def definir_consciencia_callback(callback: Callable[[str], None] | None) -> None:
    """Registra o destino dos comentários do ciclo de iniciativa.

    Sem uma call ativa o callback fica nulo e o comentário é descartado.
    """
    global _consciencia_callback
    _consciencia_callback = callback


def definir_iniciativa(ativo: bool) -> None:
    """Liga/desliga o ciclo de iniciativa em tempo real -- ver
    _ciclo_iniciativa, que checa essa flag a cada volta em vez de só na
    inicialização. Chamado pelo dashboard (ver dashboard.py)."""
    global _iniciativa_ativa
    _iniciativa_ativa = ativo


def definir_em_call(ativo: bool) -> None:
    """Liga/desliga o sinal de 'tem gente numa call agora' -- chamado por
    bridge_client.py (_ligar_visao_se_precisar/_desligar_visao_se_precisar,
    mesmo _guilds_com_call que a visão já usa) toda vez que uma guild
    entra/sai de call.

    GATE FÍSICO: _ciclo_iniciativa só considera agir (olhar/mover por
    conta própria) quando isso está ligado -- não tem por que o corpo se
    mexer sozinho se não tem ninguém que possa ver ou ouvir o resultado.
    Combinado explicitamente com Alex: iniciativa exige EVA_ROBOT_INICIATIVA=1
    *e* call ativa, as duas condições, não uma ou outra.

    threading.Event é seguro chamar/checar de qualquer thread sem lock
    (é exatamente pra isso que a classe existe)."""
    if ativo:
        _em_call.set()
    else:
        _em_call.clear()


def drenar_eventos_corpo() -> list[str]:
    """Esvazia a fila de eventos corporais dignos de virar impulso de
    Consciência (transição de segurança, recusa de comando -- ver
    _detectar_transicao_seguranca/_descrever_recusa). Não bloqueia;
    devolve [] se não tem nada novo. Chamado periodicamente por
    bridge_client.py (_laco_corpo), que empurra cada item pra
    Consciencia.evento_corporal de toda guild com call ativa -- mesmo
    padrão de _laco_visao."""
    eventos: list[str] = []
    while True:
        try:
            eventos.append(_eventos_corpo.get_nowait())
        except queue.Empty:
            break
    return eventos


def conectar_dashboard() -> dict:
    """Força a conexão com o robô a partir do painel -- botão 'conectar
    agora', pra não depender só do warm-up automático ao entrar numa call
    (bridge_client._ligar_robo_consciencia_se_precisar) nem da primeira
    ferramenta robo_* ser chamada de verdade em conversa.

    Diferente de parar_dashboard/estop_dashboard/reset_estop_dashboard
    (que usam _chamar, e portanto já supõem uma thread capaz de
    responder), este é o único caso que chama _iniciar_thread_robo()
    diretamente -- é a própria ação de subir a conexão, não uma ação que
    depende dela já estar de pé. _iniciar_thread_robo() já é idempotente
    (só cria a thread uma vez, protegida por _lock_inicio -- ver
    docstring dela) e espera até 5s por _pronto antes de devolver, então
    clicar de novo com o robô já conectado é seguro e rápido (retorna na
    hora, sem reconectar à toa)."""
    _iniciar_thread_robo()
    conectado = _cliente is not None and _cliente.conectado
    if not conectado:
        return {"ok": False, "erro": "robo_desconectado",
                "detalhe": "thread subiu mas conexão não se estabeleceu a tempo"}
    return {"ok": True, "conectado": True}


def status_dashboard() -> dict:
    """Estado do robô pro painel de controle.

    Diferente de minecraft_tools.status_dashboard() (que é só leitura de
    atributo, sem I/O): aqui, se já existe uma thread de conexão (alguma
    ferramenta de robô já foi usada nesta sessão), busca o estado real do
    robô -- inclusive identificação de câmera (qual está ativa, device
    id de cada uma) e segurança (obstáculo, bateria, emergency stop) --
    com TIMEOUT CURTO (2s), porque isto é chamado a cada poll do painel:
    se o robô estiver desligado/inacessível, o painel não pode travar
    esperando.

    Se NENHUMA ferramenta de robô foi usada ainda, não força conexão só
    pra mostrar status (mesmo espírito de minecraft_tools) -- devolve
    conectado=False sem tentar nada.
    """
    conectado = _cliente is not None and _cliente.conectado
    resultado = {
        "conectado": conectado,
        "iniciativa_ativa": _iniciativa_ativa,
        "estado": None,
        "erro_estado": None,
    }

    if _thread is None:
        return resultado

    async def _fn(cliente: ClienteRobo):
        return await cliente.estado()

    bruto = _chamar(_fn, timeout=2.0)
    if bruto.get("erro"):
        resultado["erro_estado"] = bruto.get("erro")
    elif bruto.get("ok"):
        resultado["estado"] = bruto.get("estado")
    else:
        resultado["erro_estado"] = bruto.get("erro", "falha_desconhecida")
    return resultado


def parar_dashboard() -> dict:
    """Para o robô a partir do painel -- mesma ação de robo_parar, só que
    chamável sem passar pelo registro de ferramentas (o painel não é o
    modelo decidindo, é você clicando)."""
    async def _fn(cliente: ClienteRobo):
        return await cliente.parar()
    return _desembrulhar(_chamar(_fn, timeout=3.0))


def estop_dashboard(motivo: str = "acionado pelo painel") -> dict:
    """Emergency stop a partir do painel. DE PROPÓSITO não é uma
    ferramenta registrada (@registro.adicionar) -- estop é uma ação
    drástica demais pra deixar o modelo decidir sozinho como se fosse
    'mais um botão'; robo_parar (parada normal) já cobre o caso de 'algo
    parece errado, prefiro não continuar'. Isto aqui é o botão vermelho
    de verdade, só acionável por você, pelo painel."""
    async def _fn(cliente: ClienteRobo):
        return await cliente.estop(motivo)
    return _desembrulhar(_chamar(_fn, timeout=3.0))


def reset_estop_dashboard() -> dict:
    """Libera o emergency stop a partir do painel. Mesma lógica de
    estop_dashboard() pra NÃO ser ferramenta da EVA -- decidir 'já é
    seguro seguir andando' depois de uma parada de emergência é
    julgamento humano, não do modelo. O servidor ainda faz a checagem
    real (safety._check_safe_to_reset -- bateria/obstáculo) antes de
    aceitar; isto aqui só manda o pedido."""
    async def _fn(cliente: ClienteRobo):
        return await cliente.reset_estop()
    return _desembrulhar(_chamar(_fn, timeout=3.0))


def obter_quadro_camera(idade_maxima_s: float | None = None) -> bytes | None:
    """Último quadro JPEG da câmera do robô, se for RECENTE -- usado por
    vision/visao_robo.py (SistemaVisualRobo), que não precisa conhecer
    ClienteVideoRobo por dentro, só pedir o quadro atual.

    Devolve None quando a conexão de vídeo ainda não subiu, quando não
    chegou quadro nenhum, OU quando o último quadro é velho demais.

    A checagem de idade é o ponto: sem ela, uma conexão de vídeo que caiu
    deixa `ultimo_frame` congelado pra sempre, e quem consome isto recebe
    bytes JPEG perfeitamente válidos de uma cena que não existe mais.
    Aconteceu em uso real -- a task de vídeo foi coletada pelo GC (ver
    _criar_tarefa_permanente) e ela descreveu, com confiança, um corredor
    que tinha visto minutos antes. Recusar o quadro velho transforma isso
    num erro visível em vez de numa alucinação silenciosa."""
    if _cliente_video is None or _cliente_video.ultimo_frame is None:
        return None
    limite = IDADE_MAXIMA_QUADRO_S if idade_maxima_s is None else idade_maxima_s
    if limite > 0 and (time.monotonic() - _cliente_video.ultimo_frame_ts) > limite:
        return None
    return _cliente_video.ultimo_frame


def _cliente_visao_robo():
    """Cliente de visão pro robô -- reusa a MESMA config de cfg.visao
    (mesmo servidor/modelo que já atende a visão de tela, ver visao.py);
    só o prompt muda (PROMPT_CENA_ROBO em vez de PROMPT_CENA). Resolvido
    a partir da config carregada do zero, mesmo motivo de
    _clientes_llm_decisao(): este módulo não tem instância viva de EVA
    por perto."""
    from ..config import carregar_config
    from ..vision.minicpm import ClienteVisao
    cfg = carregar_config().visao
    return ClienteVisao(base_url=cfg.base_url, modelo=cfg.modelo,
                         api_key=cfg.api_key, timeout=cfg.timeout)


def _clientes_llm_decisao():
    """Monta o par (cliente principal, cliente reserva) pra decisão
    pequena e estruturada -- mesma fábrica que orchestrator.py usa via
    decision.clientes_decisao, só que resolvida aqui a partir da config
    carregada do zero, porque este módulo não tem uma instância viva de
    EVA por perto (mesmo motivo pelo qual minecraft_tools também não usa
    self.cfg -- é uma thread própria, desacoplada do ciclo principal)."""
    from ..config import carregar_config
    from ..decision import clientes_decisao
    cfg = carregar_config()
    return clientes_decisao(cfg.decisao)


def _seguro_para_iniciativa(estado: dict) -> tuple[bool, str]:
    seg = (estado or {}).get("safety") or {}
    if seg.get("emergency_stop"):
        return False, "emergency_stop ativo"
    if not seg.get("watchdog_ok", True):
        return False, "watchdog não ok"
    bateria = (seg.get("last_sensor_data") or {}).get("battery_v")
    if bateria is not None and bateria < BATERIA_MINIMA_SEGURA_V:
        return False, f"bateria baixa ({bateria}V)"
    return True, "ok"


def _detectar_transicao_seguranca(seg: dict) -> None:
    """Compara a leitura de safety deste tick com a do tick anterior e
    emite evento corporal só na TRANSIÇÃO (virou X agora), nunca a cada
    tick que já está em X -- senão "ainda em emergency stop" repetiria a
    cada tick e viraria ruído.

    CORRIGIDO (era v1, de propósito, agora fechado): antes só rodava
    dentro de _ciclo_iniciativa (cadence de INICIATIVA_INTERVALO_S,
    padrão 5min) e só quando havia call ativa -- uma transição de
    segurança sem nenhum comando EVA envolvido podia demorar até 5min
    pra virar impulso. Agora roda também dentro de _ciclo_heartbeat (a
    cada CHECAGEM_SEGURANCA_A_CADA_N_TICKS ticks de heartbeat, tipo
    poucos segundos), incondicional -- não depende mais de iniciativa
    ligada nem de call ativa. Recusa de comando (_talvez_emitir_recusa)
    continua sendo a via mais comum e instantânea de qualquer forma, já
    que quase toda transição de segurança real acontece EM RESPOSTA a
    uma tentativa de mover -- isto aqui cobre o resto: transição
    espontânea (bateria caindo sozinha, obstáculo aparecendo) sem
    nenhuma tentativa de mover envolvida."""
    global _ultimo_estado_seguranca

    bateria = (seg.get("last_sensor_data") or {}).get("battery_v")
    atual = {
        "emergency_stop": bool(seg.get("emergency_stop", False)),
        "watchdog_ok": bool(seg.get("watchdog_ok", True)),
        "bateria_baixa": bateria is not None and bateria < BATERIA_MINIMA_SEGURA_V,
        "bateria_v": bateria,
    }
    anterior = _ultimo_estado_seguranca
    _ultimo_estado_seguranca = atual

    if anterior is None:
        return  # primeira leitura da sessão -- sem "antes" pra comparar

    if atual["emergency_stop"] and not anterior["emergency_stop"]:
        _emitir_evento_corpo("acabei de entrar em emergency stop -- parei de me mover")
    elif not atual["emergency_stop"] and anterior["emergency_stop"]:
        _emitir_evento_corpo("o emergency stop foi liberado -- posso me mover de novo")

    if not atual["watchdog_ok"] and anterior["watchdog_ok"]:
        _emitir_evento_corpo("o watchdog do meu corpo perdeu contato por um instante")

    if atual["bateria_baixa"] and not anterior["bateria_baixa"]:
        _emitir_evento_corpo(f"minha bateria caiu abaixo do seguro ({atual['bateria_v']}V)")


async def _ciclo_iniciativa() -> None:
    """Roda pra sempre em segundo plano (só age de verdade se
    EVA_ROBOT_INICIATIVA=1 *e* houver call ativa -- ver definir_em_call),
    checando a cada INICIATIVA_INTERVALO_S segundos se ela quer se mexer
    por conta própria. Nunca age com bateria baixa, watchdog ruim ou
    emergency stop ativo -- mesmas travas de segurança do resto do
    sistema, checadas de novo aqui porque o servidor pode recusar o
    comando de qualquer forma, mas é melhor nem tentar do que gerar um
    comando que sabemos que vai falhar.

    COMBINADO COM ALEX: ela só se move por iniciativa própria quando tem
    alguém numa call pra ver/ouvir o resultado -- gate de _em_call vem
    ANTES de qualquer coisa (nem consulta o robô), pra não gastar
    ciclo/rede/LLM à toa quando não tem ninguém por perto de qualquer jeito.
    """
    while True:
        await asyncio.sleep(INICIATIVA_INTERVALO_S)
        if not _iniciativa_ativa:
            continue  # desligada agora -- só espera o próximo ciclo, não decide nada
        if not _em_call.is_set():
            continue  # sem call ativa -- não tem por que se mexer nem checar nada
        try:
            if _cliente is None or not _cliente.conectado:
                continue  # sem conexão real, não tem base pra decidir nada

            resposta_estado = await _cliente.estado()
            if not resposta_estado.get("ok"):
                continue
            estado = resposta_estado.get("estado", {})

            # Nota a transição de segurança MESMO que o tick decida não
            # agir logo depois -- ver docstring de _detectar_transicao_seguranca.
            # Chamada aqui TAMBÉM (além de agora rodar dentro de
            # _ciclo_heartbeat, bem mais frequente) não duplica comentário:
            # a função compara contra o último estado global visto de
            # QUALQUER lugar, então na prática o heartbeat já vai ter
            # pego a transição antes deste tick lento chegar -- isto aqui
            # só continua existindo como rede de segurança caso o
            # heartbeat esteja falhando por algum motivo nesse momento.
            _detectar_transicao_seguranca(estado.get("safety") or {})

            ok, motivo = _seguro_para_iniciativa(estado)
            if not ok:
                print(f"[iniciativa-robo] pulando ciclo: {motivo}")
                continue

            prompt = PROMPT_INICIATIVA.format(estado=json.dumps(estado, ensure_ascii=False))
            from ..decision import completar_com_reserva
            principal, reserva = _clientes_llm_decisao()
            bruto = completar_com_reserva(principal, reserva, prompt)
            m = re.search(r"\{.*\}", bruto, re.S)
            if not m:
                continue
            decisao = json.loads(m.group(0))
            if decisao.get("agir"):
                acao = decisao.get("acao")
                if acao == "olhar":
                    yaw = decisao.get("yaw")
                    pitch = decisao.get("pitch")
                    resultado = await _cliente.olhar(yaw=yaw, pitch=pitch)
                    if resultado.get("ok"):
                        print(f"[iniciativa-robo] olhou por conta própria: yaw={yaw} pitch={pitch}")
                    else:
                        print(f"[iniciativa-robo] olhar recusado: {resultado.get('erro')}")

                elif acao == "mover":
                    vx = max(-VELOCIDADE_MAX_INICIATIVA, min(VELOCIDADE_MAX_INICIATIVA,
                              float(decisao.get("vx", 0.0))))
                    vy = max(-VELOCIDADE_MAX_INICIATIVA, min(VELOCIDADE_MAX_INICIATIVA,
                              float(decisao.get("vy", 0.0))))
                    vz = max(-VELOCIDADE_MAX_INICIATIVA, min(VELOCIDADE_MAX_INICIATIVA,
                              float(decisao.get("vz", 0.0))))
                    resultado = await _cliente.mover(
                        vx=vx, vy=vy, vz=vz,
                        ttl_ms=int(DURACAO_MAX_MOVIMENTO_S * 1000),
                    )
                    if resultado.get("ok"):
                        print(f"[iniciativa-robo] moveu por conta própria: vx={vx} vy={vy} vz={vz}")
                        # Duração decidida AQUI, não pelo modelo -- ver
                        # docstring da seção. Depois de parar, o próximo
                        # ciclo relê o estado real antes de decidir de novo.
                        await asyncio.sleep(DURACAO_MAX_MOVIMENTO_S)
                        await _cliente.parar()
                    else:
                        print(f"[iniciativa-robo] movimento recusado pelo servidor: {resultado.get('erro')}")

            comentario = decisao.get("comentario")
            if comentario and _consciencia_callback:
                try:
                    _consciencia_callback(str(comentario))
                except Exception as e:
                    print(f"[iniciativa-robo] callback de comentário falhou: {e}")
            # ação desconhecida ou ausente: ignora silenciosamente, mesmo
            # espírito de "agir=false" não gerar ruído.
        except Exception as e:
            print(f"[iniciativa-robo] ciclo falhou, tenta de novo no próximo intervalo: {e}")