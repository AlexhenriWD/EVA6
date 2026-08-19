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

from ..integrations.robot_client import ClienteRobo
from .registry import registro

_cliente: ClienteRobo | None = None
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_pronto = threading.Event()

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

# Última leitura de safety vista por _detectar_transicao_seguranca -- só
# emite impulso na TRANSIÇÃO (ex: virou emergency_stop AGORA), não a cada
# tick que ela já está em emergency_stop -- senão vira ruído repetitivo.
_ultimo_estado_seguranca: dict | None = None

HEARTBEAT_INTERVALO_S = float(os.environ.get("EVA_ROBOT_HEARTBEAT_S", "0.5"))
BATERIA_MINIMA_SEGURA_V = float(os.environ.get("EVA_ROBOT_BATERIA_MINIMA", "6.6"))


def _iniciar_thread_robo() -> None:
    """Sobe a thread com o event loop dedicado, uma vez só, preguiçoso --
    só na primeira chamada de ferramenta de robô de verdade."""
    global _thread
    if _thread is not None:
        return

    def rodar_loop():
        global _cliente, _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        host = os.environ.get("EVA_ROBOT_HOST", "127.0.0.1")
        porta = int(os.environ.get("EVA_ROBOT_PORT", "5000"))
        _cliente = ClienteRobo(host=host, port=porta, fonte="eva")
        _pronto.set()
        # Sempre agendados, independente do valor inicial de
        # _iniciativa_ativa -- controle de liga/desliga é checado a cada
        # ciclo (ver definir_iniciativa), não só na inicialização.
        _loop.create_task(_ciclo_heartbeat())
        _loop.create_task(_ciclo_iniciativa())
        _loop.run_forever()

    _thread = threading.Thread(target=rodar_loop, daemon=True, name="eva-robot")
    _thread.start()
    _pronto.wait(timeout=5)


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
    "Move o robô continuamente na direção dada por um curto período (o "
    "servidor tem um teto de tempo por comando, não fica andando pra "
    "sempre). vx=frente(+)/trás(-), vy=lateral direita(+)/esquerda(-), "
    "vz=giro horário(+)/anti-horário(-), todos de -1.0 a 1.0 -- use "
    "valores pequenos (0.2-0.4) a menos que tenha certeza do caminho. O "
    "servidor recusa o comando se detectar obstáculo, bateria crítica ou "
    "emergency stop ativo -- se vier erro, NÃO insista no mesmo comando, "
    "chame robo_estado pra entender o motivo antes de tentar de novo.",
    {"vx": "-1.0 a 1.0 (frente/trás)", "vy": "-1.0 a 1.0 (lateral)",
     "vz": "-1.0 a 1.0 (giro)"},
)
def robo_mover(vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> dict:
    async def _fn(cliente: ClienteRobo):
        return await cliente.mover(vx=vx, vy=vy, vz=vz)
    return _talvez_emitir_recusa(_desembrulhar(_chamar(_fn)))


@registro.adicionar(
    "robo_olhar",
    "Move a cabeça do robô (base/yaw e ombro/pitch), em graus. Use pra "
    "olhar em volta sem deslocar o corpo -- prefira isso a robo_mover "
    "quando o objetivo é só ver algo, não chegar em algum lugar.",
    {"yaw": "graus, 0-180, opcional", "pitch": "graus, 0-180, opcional"},
)
def robo_olhar(yaw: int | None = None, pitch: int | None = None) -> dict:
    async def _fn(cliente: ClienteRobo):
        return await cliente.olhar(yaw=yaw, pitch=pitch)
    return _talvez_emitir_recusa(_desembrulhar(_chamar(_fn)))


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
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVALO_S)
        if _cliente is not None:
            try:
                r = await _cliente.heartbeat()
                if r.get("ok"):
                    falhas_consecutivas = 0
                else:
                    falhas_consecutivas += 1
                    print(f"[heartbeat-robo] falhou (consecutiva #{falhas_consecutivas}): {r}")
            except Exception as e:
                falhas_consecutivas += 1
                print(f"[heartbeat-robo] exceção (consecutiva #{falhas_consecutivas}): {e}")


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

PROMPT_INICIATIVA = """Você está com um corpo físico agora -- um robô com rodas, câmera e cabeça móvel -- sozinha, sem ninguém pedindo nada. Você pode escolher se mexer um pouco (olhar em volta ou andar um pouco), ou simplesmente não fazer nada -- as duas são respostas válidas, não precisa preencher o tempo com ação sem motivo real.

Estado atual do robô (real, agora):
{estado}

Decida: você quer se mexer agora, por conta própria? Se sim, o quê -- olhar em volta (ação "olhar", com yaw/pitch em graus) ou andar um pouco numa direção (ação "mover", com vx/vy/vz pequenos, tipo 0.3)? Se não, tudo bem também -- responda que não.

Responda APENAS um JSON, sem mais nada:
{{"agir": true, "acao": "olhar", "yaw": 60, "pitch": 100}}
ou
{{"agir": true, "acao": "mover", "vx": 0.3, "vy": 0.0, "vz": 0.0}}
ou
{{"agir": false}}"""


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
    cada INICIATIVA_INTERVALO_S e viraria ruído.

    LIMITAÇÃO CONHECIDA (v1, de propósito -- ver conversa com Alex):
    isso só roda dentro de _ciclo_iniciativa, ou seja, no cadence de
    INICIATIVA_INTERVALO_S (padrão 5min) e só quando há call ativa. Uma
    transição de segurança que aconteça SEM nenhum comando EVA envolvido
    (ex: bateria caindo sozinha enquanto ela não tenta nada) pode demorar
    até 5min pra virar impulso. Recusa de comando (_talvez_emitir_recusa)
    já é instantânea porque não depende deste polling -- é a via mais
    comum de qualquer forma, já que quase toda transição de segurança
    real acontece EM RESPOSTA a uma tentativa de mover."""
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
            if not decisao.get("agir"):
                continue  # "decidiu não fazer nada" não gera log de propósito -- é o resultado esperado na maior parte do tempo

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
            # ação desconhecida ou ausente: ignora silenciosamente, mesmo
            # espírito de "agir=false" não gerar ruído.
        except Exception as e:
            print(f"[iniciativa-robo] ciclo falhou, tenta de novo no próximo intervalo: {e}")