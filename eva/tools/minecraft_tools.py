"""
Ferramentas de Minecraft -- o corpo dela no jogo, exposto pro mesmo
mecanismo de ferramenta que já existe (registro em registry.py).

DIFERENTE de builtin.py: toda ferramenta ali é sem estado, abre e fecha
conexão própria a cada chamada (clima, buscar). Minecraft não pode
funcionar assim -- reconectar a cada ação perderia o bot do lugar onde
estava e seria lento demais pra qualquer coisa em tempo real. Por isso
aqui existe uma conexão PERSISTENTE, mantida numa thread dedicada com seu
próprio event loop asyncio, e cada ferramenta (função síncrona, mesmo
contrato de sempre -- recebe argumento nomeado, devolve dict) faz a ponte
pra essa thread e espera o resultado.

A conexão só sobe na PRIMEIRA vez que alguma ferramenta de Minecraft for
chamada de verdade -- sem Minecraft configurado ou o bridge não rodando,
essas ferramentas nunca tentam conectar sozinhas, e falham com
{"erro": ...} como qualquer outra ferramenta deste projeto quando algo dá
errado (nunca levantam exceção, mesma regra de registry.py).

PERCEPÇÃO (novo): antes este módulo era só saída -- a EVA agia no jogo mas
nada do que acontecia lá chegava até ela. Chat de jogador, dano e morte
chegavam do bridge e morriam num print() dentro de minecraft_client. Agora
existe fila de eventos + drenar_eventos_jogo(), exatamente o mesmo padrão
de robot_tools (_emitir_evento_corpo / drenar_eventos_corpo): o cliente
entrega por callback, aqui vira uma frase em primeira pessoa, e
bridge_client empurra pra Consciencia.evento_jogo de toda guild com call.

IMPORTANTE: pra essas ferramentas serem alcançáveis de verdade, o decisor
por LLM precisa estar ligado (EVA_DECISION_LLM=1) -- o decisor por regras
(padrão hoje) não tem como saber quando chamar "minecraft_minerar" sem
regex escrita à mão pra cada caso, e escrever regex pra "quando ela deve
minerar vs atacar vs craftar" não é uma tarefa de padrão de texto, é
julgamento. O decisor por LLM já lê a descrição de toda ferramenta
registrada (via registro.descrever()) e decide dinamicamente -- é
literalmente pra isso que ele existe.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field

from ..config import DecisionConfig
from ..integrations.minecraft_client import ClienteMinecraft
from .registry import registro

_cliente: ClienteMinecraft | None = None
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_pronto = threading.Event()


# ===========================================================================
# PERCEPÇÃO -- o que acontece no jogo vira impulso de Consciência
# ===========================================================================
#
# Contrato de thread (mesmo de robot_tools):
#   _eventos_jogo   queue.Queue, thread-safe por construção. Este módulo só
#                   PRODUZ (de dentro do event loop dedicado, via callback
#                   do cliente); bridge_client só CONSOME (drenar_eventos_
#                   jogo, chamado do event loop principal). Nunca o
#                   contrário -- é o que mantém os dois loops desacoplados.
#
# Fila LIMITADA de propósito: se ninguém estiver drenando (EVA rodando sem
# bridge de Discord, ou nenhuma call ativa), evento de jogo continua
# chegando a cada poucos segundos e a fila cresceria pra sempre. Com teto,
# o pior caso é perder evento velho -- que é o certo, já que comentar algo
# que aconteceu há muito tempo no jogo já não vale nada mesmo.
MAX_EVENTOS_NA_FILA = 50
_eventos_jogo: queue.Queue = queue.Queue(maxsize=MAX_EVENTOS_NA_FILA)

# Dano pequeno não vira evento. Cair meio coração de queda, tomar 1 de
# fome, raspar em cacto -- isso acontece o tempo todo e virar impulso a
# cada vez seria ruído constante, exatamente o problema que
# _detectar_transicao_seguranca resolve do lado do robô. Só dano que
# alguém comentaria em voz alta passa.
DANO_MINIMO_PRA_COMENTAR = 3.0

# Limite de caracteres por mensagem no chat do Minecraft. O servidor
# DESCARTA (ou o cliente trunca) qualquer coisa acima disso -- resposta
# longa dela simplesmente não apareceria, sem erro nenhum de volta. 250
# em vez de 256 pra deixar folga pro prefixo que o servidor coloca no
# nome. Ver _partir_para_chat.
MAX_CHARS_CHAT_JOGO = 250
# Teto de mensagens por resposta. Sem isso, uma resposta longa vira seis
# linhas seguidas de spam no chat de todo mundo no servidor.
MAX_MENSAGENS_POR_RESPOSTA = 3


def _emitir_evento_jogo(descricao: str, forca: float | None = None) -> None:
    """Enfileira algo que já veio pronto pra virar impulso de Consciência
    (dano, morte, conexão, desfecho de tarefa). Fala de jogador NÃO passa
    por aqui -- ver _ao_chat."""
    _enfileirar({"tipo": "impulso", "descricao": descricao, "forca": forca})


def _enfileirar(item: dict) -> None:
    """Nunca bloqueia, nunca levanta -- chamado de dentro do event loop
    dedicado, onde qualquer bloqueio trava a recepção inteira do
    WebSocket."""
    try:
        _eventos_jogo.put_nowait(item)
    except queue.Full:
        # Fila cheia significa que ninguém está drenando. Descarta o mais
        # velho pra abrir espaço: evento recente vale mais que evento
        # antigo neste domínio.
        try:
            _eventos_jogo.get_nowait()
            _eventos_jogo.put_nowait(item)
        except Exception:
            pass


def drenar_eventos_jogo() -> list[dict]:
    """Esvazia a fila de coisas que aconteceram no jogo. Não bloqueia;
    devolve [] se não tem nada novo. Chamado periodicamente por
    bridge_client.py (_laco_jogo) -- mesmo padrão de
    drenar_eventos_corpo/_laco_corpo.

    Dois formatos de item, e a distinção importa:

      {"tipo": "chat", "jogador": ..., "texto": ...}
          Fala CRUA de jogador. Deliberadamente NÃO é impulso ainda --
          quem decide se ela responde no chat do jogo ou só comenta na
          call é o bridge_client, porque é a mesma classe de decisão que
          já vive lá pro Discord (responder só quando mencionada, ver
          _ao_receber_mensagem). Se isto virasse impulso aqui, ela
          responderia no chat E comentaria a mesma coisa em voz alta.

      {"tipo": "impulso", "descricao": ..., "forca": ...}
          Já pronto pra Consciencia.evento_jogo. `forca=None` significa
          "usa o padrão do tipo".
    """
    eventos: list[dict] = []
    while True:
        try:
            eventos.append(_eventos_jogo.get_nowait())
        except queue.Empty:
            break
    return eventos


def nome_no_jogo() -> str:
    """Nome do bot no servidor -- é assim que alguém chama por ela no
    chat. Mesma variável que o bridge em Node usa pra entrar."""
    return os.environ.get("EVA_MC_USERNAME", "EVA")


def _ao_chat(jogador: str, texto: str) -> None:
    """Alguém falou no chat do jogo.

    O eco da própria fala dela já foi filtrado do lado do bridge
    (`username === bot.username` em minecraft_bridge.js), então tudo que
    chega aqui é de outra pessoa. Vai cru pra fila -- ver
    drenar_eventos_jogo pro porquê de não virar impulso aqui.
    """
    _enfileirar({"tipo": "chat", "jogador": jogador, "texto": texto.strip()})


def _partir_para_chat(texto: str) -> list[str]:
    """Quebra uma resposta em mensagens que cabem no chat do jogo.

    Corta em fronteira de PALAVRA, não no meio. Prefere quebrar depois de
    fim de frase quando dá, pra cada mensagem sair inteira em vez de
    cortada na metade de uma ideia.
    """
    texto = " ".join(texto.split())
    if not texto:
        return []
    if len(texto) <= MAX_CHARS_CHAT_JOGO:
        return [texto]

    partes: list[str] = []
    resto = texto
    while resto and len(partes) < MAX_MENSAGENS_POR_RESPOSTA:
        if len(resto) <= MAX_CHARS_CHAT_JOGO:
            partes.append(resto)
            break
        janela = resto[:MAX_CHARS_CHAT_JOGO]
        corte = max(janela.rfind(". "), janela.rfind("! "), janela.rfind("? "))
        if corte < MAX_CHARS_CHAT_JOGO // 2:
            corte = janela.rfind(" ")
        if corte <= 0:
            corte = MAX_CHARS_CHAT_JOGO - 1
        else:
            corte += 1
        partes.append(resto[:corte].strip())
        resto = resto[corte:].strip()
    return [p for p in partes if p]


def _ao_evento(evento: str, dados: dict) -> None:
    """Dano e morte vindos do bridge. O bridge já filtra dano pra só
    disparar em QUEDA de vida (regenerar não dispara nada); aqui filtra
    de novo por magnitude, pra dano trivial não virar impulso."""
    if evento == "dano":
        perdeu = float(dados.get("perdeu") or 0)
        vida = dados.get("vida")
        if perdeu < DANO_MINIMO_PRA_COMENTAR and (vida is None or vida > VIDA_MINIMA_SEGURA):
            return
        _emitir_evento_jogo(
            f"levei dano no Minecraft: perdi {perdeu:.0f} de vida, tô com {vida}/20")
    elif evento == "morte":
        pos = (dados or {}).get("posicao") or {}
        onde = (f" em ({pos.get('x')}, {pos.get('y')}, {pos.get('z')})"
                if pos else "")
        _emitir_evento_jogo(f"eu morri no Minecraft{onde} -- perdi o que tava carregando")
    elif evento == "item_quebrado":
        _emitir_evento_jogo(
            f"quebrou um item que eu tava usando no Minecraft: {dados.get('item', 'não sei qual')}")


def _ao_conexao(conectado: bool, motivo: str) -> None:
    if conectado:
        _emitir_evento_jogo(f"acabei de entrar no Minecraft -- {motivo}")
    else:
        _emitir_evento_jogo(f"caí do Minecraft: {motivo}")


# ===========================================================================
# CONEXÃO
# ===========================================================================


def _iniciar_thread_minecraft() -> None:
    """Sobe a thread com o event loop dedicado, uma vez só, preguiçoso --
    só na primeira chamada de ferramenta de Minecraft de verdade."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    if _thread is not None:
        # Thread morta (aconteceu de verdade antes do fix de exceção em
        # ClienteMinecraft.rodar: ConnectionClosedError matava o loop e
        # nada nunca reiniciava). Deixa recriar em vez de ficar preso.
        print("[minecraft] thread de conexão tinha morrido -- recriando")
        _pronto.clear()
        _thread = None

    def rodar_loop():
        global _cliente, _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        url = os.environ.get("EVA_MC_BRIDGE_URL", "ws://localhost:8766")
        _cliente = ClienteMinecraft(url, verboso=False)
        # verboso=False: sem terminal interativo aqui. Os prints do modo
        # solto (snapshot a cada 4s) inundariam o console do processo
        # principal, misturados com log de voz e de conversa.
        _cliente.mostrar_snapshot_automatico = False
        # É AQUI que a percepção passa a existir. Sem estes três, este
        # módulo é só saída -- ela age no jogo e não sabe de nada.
        _cliente.ao_chat = _ao_chat
        _cliente.ao_evento = _ao_evento
        _cliente.ao_conexao = _ao_conexao
        _pronto.set()
        # Sempre agendado, independente do valor inicial -- controle de
        # liga/desliga agora é por dentro (_iniciativa_ativa, checado a
        # cada ciclo), não por fora. Isso é o que permite o dashboard
        # alternar em tempo real, sem precisar reiniciar (diferente da
        # visão, que só decide na inicialização).
        _loop.create_task(_ciclo_iniciativa())
        try:
            _loop.run_until_complete(_cliente.rodar())
        except Exception as e:
            print(f"[minecraft] laço de conexão encerrou: {type(e).__name__}: {e}")

    _thread = threading.Thread(target=rodar_loop, daemon=True, name="eva-minecraft")
    _thread.start()
    _pronto.wait(timeout=5)


def _esperar_conexao(timeout: float = 10.0) -> bool:
    """Espera a conexão de verdade ficar pronta, não só o objeto existir.

    Bug real confirmado em teste: _pronto sinalizava assim que o objeto
    ClienteMinecraft era criado, mas o handshake de verdade (WebSocket
    conectar + esperar a mensagem "conectado" do bridge) leva um tempinho
    real depois disso -- o log mostrava ela conectando no console e, no
    MESMO turno, a ferramenta já reportando desconectado, porque checou
    `conectado` cedo demais, antes do handshake terminar.
    """
    fim = time.time() + timeout
    while time.time() < fim:
        if _cliente is not None and _cliente.conectado:
            return True
        time.sleep(0.2)
    return False


def _esperar_snapshot(timeout: float = 8.0) -> bool:
    """Espera o PRIMEIRO snapshot chegar -- não é a mesma coisa que estar
    conectado. Confirmado em teste real: `conectado=True` não significa
    que já tem snapshot, porque o periódico só chega a cada
    SNAPSHOT_INTERVALO_MS (até 4s de espera natural desde a conexão).
    Checar `ultimo_snapshot` uma vez só, sem esperar, fazia a tarefa
    desistir na largada ("perdi o estado do jogo") mesmo com tudo
    funcionando -- só corrida entre "conectou" e "primeiro snapshot".
    Versão síncrona (bloqueante) -- use em contexto de thread comum, não
    dentro do event loop dedicado (aí usa _esperar_snapshot_async).
    """
    fim = time.time() + timeout
    while time.time() < fim:
        if _cliente is not None and _cliente.ultimo_snapshot is not None:
            return True
        time.sleep(0.3)
    return False


async def _esperar_snapshot_async(timeout: float = 8.0) -> bool:
    """Mesma espera de _esperar_snapshot, mas sem bloquear o event loop --
    usa asyncio.sleep em vez de time.sleep, porque isso roda DENTRO do
    loop dedicado (_executar_tarefa), e um sleep bloqueante ali travaria
    até o recebimento de mensagem do bridge, não só a tarefa."""
    fim = time.time() + timeout
    while time.time() < fim:
        if _cliente is not None and _cliente.ultimo_snapshot is not None:
            return True
        await asyncio.sleep(0.3)
    return False


def _chamar(tipo: str, timeout: float = 30.0, **campos) -> dict:
    """Ponte síncrona->assíncrona: roda a corrotina no loop dedicado
    (rodando numa OUTRA thread) a partir de QUALQUER thread chamadora, e
    espera o resultado com timeout -- run_coroutine_threadsafe existe
    exatamente pra isso, é a forma segura de fazer isso sem race
    condition. Nunca levanta exceção: erro de conexão vira {"erro": ...},
    mesma regra de toda ferramenta deste projeto (ver registry.py).

    NUNCA CHAME ISTO DE DENTRO DO LOOP DEDICADO. Era exatamente esse o
    bug: _executar_tarefa é corrotina rodando NO _loop, e chamava
    _chamar("falar_no_jogo", ...) direto nos seis caminhos de
    encerramento. run_coroutine_threadsafe agenda no _loop e
    futuro.result() bloqueia a thread chamadora -- que ali é a própria
    thread do loop, que precisaria estar livre pra executar o que acabou
    de ser agendado. Resultado: 32s de loop congelado (sem receber
    snapshot, sem resolver ação nenhuma), timeout, e a mensagem nunca
    saindo. Dentro do loop, use `await _chamar_async(...)`.
    """
    _iniciar_thread_minecraft()
    if _cliente is None or _loop is None:
        return {"erro": "minecraft_indisponivel", "detalhe": "thread de conexão não iniciou a tempo"}
    if _loop.is_closed():
        return {"erro": "minecraft_indisponivel", "detalhe": "loop de conexão encerrado"}
    if not _cliente.conectado and not _esperar_conexao():
        return {"erro": "minecraft_desconectado",
                "detalhe": "bridge não conectado -- node minecraft_bridge.js está rodando?"}
    try:
        futuro = asyncio.run_coroutine_threadsafe(
            _cliente.enviar_acao(tipo, timeout=timeout, **campos), _loop)
        return futuro.result(timeout=timeout + 2)
    except Exception as e:
        return {"erro": "minecraft_falha", "detalhe": str(e)[:200]}


async def _chamar_async(tipo: str, timeout: float = 30.0, **campos) -> dict:
    """Versão pra uso DENTRO do loop dedicado (tarefa, ciclo de
    iniciativa). Mesmo contrato de retorno de _chamar, sem a ponte entre
    threads -- que é justamente o que causava o congelamento."""
    if _cliente is None:
        return {"erro": "minecraft_indisponivel", "detalhe": "sem cliente"}
    if not _cliente.conectado:
        return {"erro": "minecraft_desconectado", "detalhe": "bridge não conectado"}
    try:
        return await _cliente.enviar_acao(tipo, timeout=timeout, **campos)
    except Exception as e:
        return {"erro": "minecraft_falha", "detalhe": str(e)[:200]}


async def _falar_async(texto: str) -> None:
    """Anúncio da tarefa no chat do jogo, de dentro do loop. Falha aqui
    não muda o desfecho da tarefa -- só não avisou.

    Parte igual a minecraft_falar: o objetivo vem em linguagem natural de
    quem pediu (ou do próprio ciclo de iniciativa), então "Não consegui
    -- {objetivo}. {motivo}" pode facilmente passar do teto do chat."""
    for parte in _partir_para_chat(texto):
        await _chamar_async("falar_no_jogo", timeout=10, texto=parte)


# ------------------------------------------------------------- ferramentas


@registro.adicionar(
    "minecraft_estado",
    "Estado atual do corpo no Minecraft: posição, vida, fome, inventário, "
    "blocos e entidades por perto. Consulte antes de decidir qualquer "
    "próximo passo -- sem isso você está decidindo às cegas.",
)
def minecraft_estado() -> dict:
    if _cliente is None or not _cliente.conectado:
        _iniciar_thread_minecraft()
        _esperar_conexao()
    if _cliente is not None and _cliente.ultimo_snapshot is None:
        _esperar_snapshot()
    if _cliente is None or _cliente.ultimo_snapshot is None:
        return {"erro": "minecraft_sem_snapshot",
                "detalhe": "ainda não recebeu nenhum snapshot -- bot conectou de verdade?"}
    # Cópia rasa de propósito: registry.executar carimba "_ms" no dict que
    # a ferramenta devolve. Sem a cópia, isso ia direto pro
    # `ultimo_snapshot` compartilhado, poluindo o estado que a tarefa e o
    # ciclo de iniciativa leem.
    return dict(_cliente.ultimo_snapshot)


def snapshot_atual() -> dict | None:
    """Devolve o último snapshot sem conectar nem esperar pelo jogo."""
    if _cliente is None or _cliente.ultimo_snapshot is None:
        return None
    return dict(_cliente.ultimo_snapshot)


@registro.adicionar(
    "minecraft_mover",
    "Move o corpo no Minecraft até uma coordenada específica que você já "
    "sabe de fonte real (ex: veio de minecraft_estado). NÃO chame antes "
    "de minecraft_minerar ou minecraft_atacar -- as duas já andam até o "
    "alvo sozinhas, chamar mover antes é redundante e arriscado (manda "
    "pra coordenada às cegas, sem checar o que tem no caminho). Use só "
    "quando o destino não é sobre minerar/atacar algo específico -- ex: "
    "ir a um ponto que o jogador apontou por coordenada.",
    {"x": "coordenada X", "y": "coordenada Y", "z": "coordenada Z"},
)
def minecraft_mover(x: float, y: float, z: float) -> dict:
    # 60s: thinkTimeout do pathfinder é 15s (só a busca), mas andar de
    # verdade depois de achar caminho pode passar disso em terreno de
    # montanha -- com o padrão de 30s aqui, o Python desistia de esperar
    # antes do Node terminar de andar, e a PRÓXIMA ferramenta chamada
    # (ex: minerar) cortava esse goto() ainda em andamento sem querer
    # (setGoal(null) de uma ação nova rejeita a anterior com "goal was
    # changed"). Confirmado em teste real.
    return _chamar("mover_para", timeout=60, x=float(x), y=float(y), z=float(z))


@registro.adicionar(
    "minecraft_minerar",
    "Minera um tipo de bloco por perto (nome exato do jogo, ex: oak_log, "
    "iron_ore, stone). Anda até o mais próximo do tipo pedido e quebra.",
    {"bloco": "nome do bloco, ex: oak_log", "quantidade": "quantos minerar (padrão 1)"},
)
def minecraft_minerar(bloco: str, quantidade: int = 1) -> dict:
    return _chamar("minerar", timeout=60, bloco=str(bloco), quantidade=int(quantidade))


@registro.adicionar(
    "minecraft_craftar",
    "Fabrica um item a partir do que tem no inventário (nome exato do "
    "jogo, ex: wooden_pickaxe, oak_planks, crafting_table). Usa mesa de "
    "trabalho por perto se a receita precisar e tiver uma por perto.",
    {"item": "nome do item a craftar", "quantidade": "quantos (padrão 1)"},
)
def minecraft_craftar(item: str, quantidade: int = 1) -> dict:
    return _chamar("craftar", item=str(item), quantidade=int(quantidade))


@registro.adicionar(
    "minecraft_equipar",
    "Equipa um item do inventário na mão (ex: uma picareta antes de "
    "minerar, uma espada antes de atacar).",
    {"item": "nome do item"},
)
def minecraft_equipar(item: str) -> dict:
    return _chamar("equipar", item=str(item))


@registro.adicionar(
    "minecraft_atacar",
    "Ataca a entidade viva mais próxima que bater o nome pedido (mob como "
    "chicken/zombie, ou nome de jogador). Se aproxima antes de golpear.",
    {"alvo": "nome da entidade ou jogador"},
)
def minecraft_atacar(alvo: str) -> dict:
    # Mesmo motivo do minecraft_mover: também se aproxima via goto() antes
    # de golpear, sujeito ao mesmo timeout curto demais.
    return _chamar("atacar", timeout=60, alvo=str(alvo))


@registro.adicionar(
    "minecraft_seguir",
    "Passa a seguir um jogador pelo nome, continuamente, até receber outra "
    "ordem (minecraft_parar ou outra ação).",
    {"jogador": "nome do jogador"},
)
def minecraft_seguir(jogador: str) -> dict:
    return _chamar("seguir", jogador=str(jogador))


@registro.adicionar(
    "minecraft_parar",
    "Para de se mover -- cancela qualquer movimento ou perseguição em "
    "andamento (ex: parar de seguir alguém).",
)
def minecraft_parar() -> dict:
    return _chamar("parar")


@registro.adicionar(
    "minecraft_falar",
    "Fala uma mensagem no chat do Minecraft, visível pra todo mundo no "
    "servidor -- use pra responder quem falou com você no jogo, avisar o "
    "que vai fazer, o que fez, ou o que precisa.",
    {"texto": "o que falar no chat do jogo"},
)
def minecraft_falar(texto: str) -> dict:
    """Parte a mensagem se precisar. O chat do Minecraft tem teto de
    caracteres e o que passa dele é DESCARTADO em silêncio -- sem erro
    de volta, sem nada aparecendo no jogo. Como a resposta dela vem de
    um LLM e pode ter qualquer tamanho, a divisão fica aqui, no único
    ponto por onde toda fala no jogo passa (ferramenta, tarefa e resposta
    de chat), em vez de ser responsabilidade de cada chamador lembrar."""
    partes = _partir_para_chat(str(texto))
    if not partes:
        return {"sucesso": False, "detalhe": "nada pra falar"}
    enviadas = 0
    ultimo: dict = {}
    for parte in partes:
        ultimo = _chamar("falar_no_jogo", timeout=15, texto=parte)
        if ultimo.get("erro") or not ultimo.get("sucesso"):
            break
        enviadas += 1
    if enviadas == 0:
        return ultimo or {"erro": "minecraft_falha", "detalhe": "não enviou nada"}
    return {"sucesso": True,
            "detalhe": f"falou no chat do jogo ({enviadas} mensagem(ns))"}


# ===========================================================================
# TAREFA -- loop de execução autônoma, separado do turno de conversa
# ===========================================================================
#
# Tudo acima é ação ATÔMICA: um pedido, uma execução, um resultado, dentro
# do mesmo turno de resposta. Isso é suficiente pra "minera essa tora
# específica", mas não pra "vai lá e consegue madeira" -- objetivo aberto,
# vários passos, tempo real de jogo entre eles.
#
# Por que isso não pode ser só "chamar várias ferramentas na mesma decisão":
# confirmado em teste real, repetidas vezes -- o decisor escolhe TODAS as
# ferramentas de uma vez, ANTES de ver o resultado da primeira. Isso produz
# coordenada chutada, ferramenta redundante, ação sem fundamento no estado
# real. A tarefa aqui resolve isso invertendo a ordem: olha o estado de
# verdade, decide UM passo, executa, olha de novo -- nunca decide o passo 2
# sem já saber o resultado do passo 1.
#
# Roda como coroutine solta no MESMO loop dedicado da conexão -- não
# bloqueia o turno de conversa que a disparou. Comunicação de progresso é
# direto no chat do jogo, não pela resposta conversacional -- a tarefa
# pode levar minutos, muito além de qualquer turno de resposta.


@dataclass
class PassoTarefa:
    ferramenta: str
    args: dict
    resultado: dict
    ts: float = field(default_factory=time.time)


@dataclass
class TarefaMinecraft:
    objetivo: str
    status: str = "ativa"  # ativa | concluida | falhou | cancelada | esgotou_passos
    passos: list = field(default_factory=list)
    criada_em: float = field(default_factory=time.time)


MAX_PASSOS_TAREFA = 15
MAX_FALHAS_SEGUIDAS = 3
VIDA_MINIMA_SEGURA = 8  # de 20 -- abaixo disso, aborta e avisa, não arrisca mais

_tarefa_atual: TarefaMinecraft | None = None
_clientes_llm_tarefa = None  # (principal, reserva) -- criado preguiçoso


# Só ação atômica que faz sentido como PASSO. Fora de propósito:
#   minecraft_tarefa/status/cancelar -- a tarefa não pode chamar a si
#     mesma nem consultar o próprio status como se fosse um passo.
#   minecraft_estado -- o estado real já é injetado FRESCO no prompt a
#     cada volta do laço (o snapshot periódico chega a cada 4s). Deixar
#     essa ferramenta disponível fazia o decisor gastar um dos 15 passos
#     buscando o que já estava na mão dele, especialmente porque a
#     descrição registrada dela manda explicitamente "consulte antes de
#     decidir qualquer próximo passo" -- correto no turno de conversa,
#     desperdício aqui.
_FERRAMENTAS_TAREFA = {
    "minecraft_mover": minecraft_mover,
    "minecraft_minerar": minecraft_minerar,
    "minecraft_craftar": minecraft_craftar,
    "minecraft_equipar": minecraft_equipar,
    "minecraft_atacar": minecraft_atacar,
    "minecraft_seguir": minecraft_seguir,
    "minecraft_parar": minecraft_parar,
    "minecraft_falar": minecraft_falar,
}


def _descrever_ferramentas_tarefa() -> str:
    """Descreve SÓ as ferramentas que o laço sabe executar.

    Antes o prompt usava registro.descrever(), que lista tudo que está
    registrado no processo -- buscar, clima, robo_*, e as próprias
    minecraft_tarefa/status/cancelar. Mas _FERRAMENTAS_TAREFA só mapeia
    as de ação. Quando o decisor escolhia qualquer outra, o passo voltava
    "ferramenta X não existe" e queimava uma das três falhas seguidas
    permitidas -- a tarefa podia morrer sem nunca ter tentado nada de
    verdade, por culpa do próprio prompt.
    """
    linhas = []
    for nome in _FERRAMENTAS_TAREFA:
        f = registro.get(nome)
        if f is None:
            continue
        params = f" (parâmetros: {', '.join(f.parametros)})" if f.parametros else ""
        linhas.append(f"- {nome}: {f.descricao}{params}")
    return "\n".join(linhas)


def _clientes_para_decidir_passo():
    """Reaproveita a MESMA função que monta o par principal/reserva do
    decisor de ferramentas (decision.clientes_decisao), em vez de montar
    um cliente próprio aqui. Bug real que já aconteceu por causa disso
    NÃO existir antes: esse loop tinha sua própria cópia da lógica,
    desatualizada em relação ao decisor principal -- continuou tentando
    um modelo local que já tinha sido descarregado depois que o Groq
    virou principal só do outro lado. Um lugar só de verdade agora.
    """
    global _clientes_llm_tarefa
    if _clientes_llm_tarefa is None:
        from ..decision import clientes_decisao
        _clientes_llm_tarefa = clientes_decisao(DecisionConfig())
    return _clientes_llm_tarefa


PROMPT_PASSO = """Você está controlando um corpo no Minecraft, executando uma tarefa passo a passo. Nunca invente coordenada, bloco ou item que não esteja no "Estado atual" abaixo -- se precisar de algo que não está lá, o passo certo é se aproximar/procurar, não chutar.

Objetivo da tarefa: {objetivo}

Estado atual do jogo (real, agora -- este bloco é atualizado antes de CADA passo, não precisa pedir de novo):
{estado}

Passos já tentados nesta tarefa, do mais antigo pro mais recente:
{historico}

Ferramentas disponíveis (só estas; qualquer outro nome é passo perdido):
{ferramentas}

Decida o PRÓXIMO passo, um só. Se o objetivo já foi alcançado, responda concluido. Se já tentou o suficiente e não há caminho razoável, responda desistir com um motivo curto.

Responda APENAS um JSON, sem mais nada:
{{"ferramenta": "nome", "args": {{...}}}}
ou {{"concluido": true}}
ou {{"desistir": true, "motivo": "..."}}"""


def _formatar_historico(passos: list) -> str:
    if not passos:
        return "(nenhum ainda)"
    linhas = []
    for p in passos[-6:]:  # só os últimos -- histórico inteiro infla o prompt à toa
        marca = "sucesso" if p.resultado.get("sucesso") else "falhou"
        detalhe = p.resultado.get("detalhe") or p.resultado.get("erro") or ""
        linhas.append(f"- {p.ferramenta}({p.args}) -> {marca}: {detalhe}")
    return "\n".join(linhas)


def _decidir_proximo_passo(tarefa: TarefaMinecraft, estado: dict) -> dict:
    """Uma chamada de LLM, focada, com o estado real na mão. Nunca levanta
    exceção -- erro vira {"desistir": True, ...}, tratado como qualquer
    outra falha de passo pelo loop.

    SÍNCRONA de propósito: quem chama sempre passa por run_in_executor,
    porque completar_com_reserva é uma chamada HTTP bloqueante e rodá-la
    direto no loop dedicado congelaria a recepção do WebSocket enquanto o
    modelo pensa.

    Motivo de erro fica CURTO de propósito -- confirmado em teste real
    que o erro cru (JSON de resposta HTTP inteiro) foi parar direto no
    chat do jogo, ilegível. Corta pra uma linha.
    """
    prompt = PROMPT_PASSO.format(
        objetivo=tarefa.objetivo,
        estado=json.dumps(estado, ensure_ascii=False),
        historico=_formatar_historico(tarefa.passos),
        ferramentas=_descrever_ferramentas_tarefa(),
    )
    try:
        from ..decision import completar_com_reserva
        principal, reserva = _clientes_para_decidir_passo()
        bruto = completar_com_reserva(principal, reserva, prompt)
        m = re.search(r"\{.*\}", bruto, re.S)
        if not m:
            print(f"[tarefa] resposta sem JSON, primeiros 200 chars: {bruto[:200]!r}")
            return {"desistir": True, "motivo": "decisor não devolveu JSON"}
        return json.loads(m.group(0))
    except Exception as e:
        print(f"[tarefa] decisor de passo falhou: {e}")
        motivo = str(e).split("\n")[0][:120]  # primeira linha, curto -- isso vai pro chat do jogo
        return {"desistir": True, "motivo": f"decisor falhou: {motivo}"}


def _executar_ferramenta_passo(nome: str, args: dict) -> dict:
    """Executa uma ferramenta de passo numa thread comum (nunca no loop
    dedicado -- as ferramentas usam _chamar, que é bloqueante).

    O try/except aqui é o que impedia a tarefa de morrer travada: antes o
    laço chamava `funcao(**args)` cru, sem proteção nenhuma. Os args vêm
    de um LLM e são livres -- basta ele escrever `block=` em vez de
    `bloco=` pra dar TypeError, a exceção sair de _executar_tarefa, e
    _tarefa_atual ficar com status "ativa" pra sempre. A partir daí todo
    minecraft_tarefa respondia "tarefa_ja_ativa" até reiniciar o
    processo. registry.executar() já protege contra exatamente isso, mas
    o laço chamava as funções direto, pulando o registro.
    """
    funcao = _FERRAMENTAS_TAREFA.get(nome)
    if funcao is None:
        return {"sucesso": False,
                "detalhe": f"ferramenta '{nome}' não existe -- use só as listadas"}
    if not isinstance(args, dict):
        return {"sucesso": False, "detalhe": f"args inválidos para {nome}: esperava objeto"}
    try:
        resultado = funcao(**args)
    except TypeError as e:
        return {"sucesso": False, "detalhe": f"argumentos errados para {nome}: {str(e)[:120]}"}
    except Exception as e:
        return {"sucesso": False, "detalhe": f"{type(e).__name__}: {str(e)[:120]}"}
    if not isinstance(resultado, dict):
        return {"sucesso": False, "detalhe": f"{nome} devolveu {type(resultado).__name__}"}
    return resultado


async def _executar_tarefa(objetivo: str) -> None:
    """Roda no loop dedicado. TODA saída daqui deixa a tarefa num status
    final -- o try/finally é o que garante isso mesmo se algo inesperado
    escapar, senão a próxima tarefa é recusada pra sempre com
    "tarefa_ja_ativa"."""
    global _tarefa_atual
    tarefa = TarefaMinecraft(objetivo=objetivo)
    _tarefa_atual = tarefa
    loop = asyncio.get_running_loop()

    try:
        falhas_seguidas = 0
        while tarefa.status == "ativa" and len(tarefa.passos) < MAX_PASSOS_TAREFA:
            if _cliente is not None and _cliente.ultimo_snapshot is None:
                await _esperar_snapshot_async()
            estado = _cliente.ultimo_snapshot if _cliente else None
            if estado is None:
                tarefa.status = "falhou"
                await _falar_async(f"Preciso parar -- perdi o estado do jogo. ({objetivo})")
                return

            # trava de segurança: vida baixa aborta na hora, sem exceção --
            # motivada por incidente real (perdeu 12 de vida numa sessão de
            # teste sem nenhum passo de segurança existir ainda).
            if estado.get("vida", 20) <= VIDA_MINIMA_SEGURA:
                tarefa.status = "cancelada"
                await _falar_async(
                    f"Parando -- vida baixa ({estado.get('vida')}/20). "
                    f"Preciso de ajuda ou vou me recuperar antes.")
                _emitir_evento_jogo(
                    f"parei a tarefa '{objetivo}' no Minecraft porque minha vida tá baixa "
                    f"({estado.get('vida')}/20)")
                return

            decisao = await loop.run_in_executor(
                None, _decidir_proximo_passo, tarefa, estado)

            if decisao.get("concluido"):
                tarefa.status = "concluida"
                await _falar_async(f"Pronto -- {objetivo}.")
                _emitir_evento_jogo(f"terminei no Minecraft o que tinha ido fazer: {objetivo}")
                return
            if decisao.get("desistir"):
                tarefa.status = "falhou"
                motivo = str(decisao.get("motivo", "")).strip()
                await _falar_async(f"Não consegui -- {objetivo}. {motivo}".strip())
                _emitir_evento_jogo(
                    f"desisti da tarefa '{objetivo}' no Minecraft"
                    + (f": {motivo}" if motivo else ""))
                return

            nome = decisao.get("ferramenta")
            args = decisao.get("args") or {}
            resultado = await loop.run_in_executor(
                None, _executar_ferramenta_passo, nome, args)

            tarefa.passos.append(PassoTarefa(ferramenta=nome, args=args, resultado=resultado))

            if resultado.get("sucesso"):
                falhas_seguidas = 0
            else:
                falhas_seguidas += 1
                if falhas_seguidas >= MAX_FALHAS_SEGUIDAS:
                    tarefa.status = "falhou"
                    await _falar_async(
                        f"Desistindo -- {objetivo}. {falhas_seguidas} passos seguidos sem sucesso.")
                    _emitir_evento_jogo(
                        f"travei na tarefa '{objetivo}' no Minecraft -- "
                        f"{falhas_seguidas} passos seguidos sem sair do lugar")
                    return

        if tarefa.status == "ativa":
            tarefa.status = "esgotou_passos"
            await _falar_async(
                f"Parando por agora -- {objetivo} ainda não terminou, muitos passos.")
            _emitir_evento_jogo(
                f"parei a tarefa '{objetivo}' no Minecraft, deu muito passo e não terminei")

    except asyncio.CancelledError:
        tarefa.status = "cancelada"
        raise
    except Exception as e:
        # Rede de segurança. Nada aqui deveria levantar depois de
        # _executar_ferramenta_passo, mas se levantar, a tarefa precisa
        # terminar num status final -- senão a próxima é recusada pra
        # sempre.
        tarefa.status = "falhou"
        print(f"[tarefa] erro inesperado no laço: {type(e).__name__}: {e}")
    finally:
        if tarefa.status == "ativa":
            tarefa.status = "falhou"


@registro.adicionar(
    "minecraft_tarefa",
    "Inicia uma tarefa de Minecraft com objetivo aberto que pode levar "
    "vários passos (ex: 'conseguir madeira', 'fazer uma picareta de "
    "ferro', 'explorar essa área'). NÃO espera terminar -- roda sozinha "
    "em segundo plano, olhando o estado real antes de cada passo, e avisa "
    "no chat do jogo quando progride, termina ou desiste. Use isso em vez "
    "de minecraft_mover/minerar/craftar/atacar diretamente quando o "
    "pedido é um objetivo, não uma ação única já bem definida.",
    {"objetivo": "descrição do objetivo, em linguagem natural"},
)
def minecraft_tarefa(objetivo: str) -> dict:
    _iniciar_thread_minecraft()
    # Guardas explícitas: `_cliente` e `_loop` são None se a thread não
    # subiu a tempo. Antes isso virava AttributeError capturado lá longe
    # em registry.executar, devolvendo {"erro": "AttributeError"} -- um
    # erro que não diz nada pra ela sobre o que houve.
    if _cliente is None or _loop is None or _loop.is_closed():
        return {"erro": "minecraft_indisponivel",
                "detalhe": "thread de conexão não iniciou -- node minecraft_bridge.js está rodando?"}
    if not _cliente.conectado and not _esperar_conexao():
        return {"erro": "minecraft_desconectado", "detalhe": "bridge não conectado"}
    if _tarefa_atual is not None and _tarefa_atual.status == "ativa":
        return {"erro": "tarefa_ja_ativa",
                "detalhe": f"já tem uma tarefa rodando: {_tarefa_atual.objetivo} -- "
                           f"use minecraft_cancelar_tarefa antes de iniciar outra"}
    asyncio.run_coroutine_threadsafe(_executar_tarefa(str(objetivo)), _loop)
    return {"iniciado": True, "objetivo": objetivo,
            "nota": "rodando em segundo plano -- avisa no chat do jogo quando progredir"}


@registro.adicionar(
    "minecraft_status_tarefa",
    "Consulta o progresso da tarefa de Minecraft em andamento (ou da "
    "última, se já terminou): objetivo, status, e quantos passos já foram "
    "tentados.",
)
def minecraft_status_tarefa() -> dict:
    if _tarefa_atual is None:
        return {"tarefa": None, "detalhe": "nenhuma tarefa iniciada ainda"}
    return {
        "objetivo": _tarefa_atual.objetivo,
        "status": _tarefa_atual.status,
        "passos_tentados": len(_tarefa_atual.passos),
        "ultimo_passo": (
            f"{_tarefa_atual.passos[-1].ferramenta} -> "
            f"{'sucesso' if _tarefa_atual.passos[-1].resultado.get('sucesso') else 'falhou'}"
        ) if _tarefa_atual.passos else None,
    }


@registro.adicionar(
    "minecraft_cancelar_tarefa",
    "Cancela a tarefa de Minecraft em andamento, se houver uma.",
)
def minecraft_cancelar_tarefa() -> dict:
    if _tarefa_atual is None or _tarefa_atual.status != "ativa":
        return {"sucesso": False, "detalhe": "nenhuma tarefa ativa pra cancelar"}
    _tarefa_atual.status = "cancelada"
    return {"sucesso": True, "detalhe": f"cancelada: {_tarefa_atual.objetivo}"}


# ===========================================================================
# INICIATIVA -- ela decide sozinha se quer fazer algo, sem ninguém pedir
# ===========================================================================
#
# Mesmo espírito da iniciativa que já existe pra conversa (MODO_INICIATIVA
# em context.py): periodicamente, sem ninguém pedir nada, ela pode decidir
# começar uma tarefa por conta própria -- ou decidir que não quer fazer
# nada agora, que também é resposta válida (o próprio card já diz que
# silêncio/inação sem motivo não precisa ser preenchido).
#
# DESLIGADO POR PADRÃO -- agir sozinha no mundo de alguém é mudança de
# comportamento grande o bastante pra merecer opt-in explícito
# (EVA_MC_INICIATIVA=1), não vir ativo sem avisar.

_iniciativa_ativa = os.environ.get("EVA_MC_INICIATIVA", "0") == "1"
# 5 min padrão -- generoso de propósito. Ela não deveria ficar iniciando
# tarefa toda hora só porque pode.
INICIATIVA_INTERVALO_S = int(os.environ.get("EVA_MC_INICIATIVA_INTERVALO", "300"))

PROMPT_INICIATIVA = """Você está com um corpo no Minecraft, sozinha agora, sem ninguém pedindo nada. Você pode escolher fazer alguma coisa por conta própria, ou simplesmente não fazer nada -- as duas são respostas válidas, não precisa preencher o tempo com ação sem motivo real.

Estado atual do jogo (real, agora):
{estado}

Decida: você quer fazer alguma coisa agora, por iniciativa própria? Se sim, qual objetivo (não uma ação isolada, um objetivo que faça sentido perseguir)? Se não, tudo bem também -- responda que não.

Responda APENAS um JSON, sem mais nada:
{{"agir": true, "objetivo": "..."}}
ou
{{"agir": false}}"""


def status_dashboard() -> dict:
    """Estado do Minecraft pro painel de controle -- só leitura, seguro
    de chamar de qualquer thread (só lê atributo simples, sem I/O, sem
    conectar nada sozinho -- se ninguém usou ferramenta de Minecraft
    ainda, `_cliente` é None e isso reflete "nunca conectado", não tenta
    forçar conexão só pra mostrar status).
    """
    conectado = _cliente is not None and _cliente.conectado
    snapshot = _cliente.ultimo_snapshot if _cliente else None
    tarefa = None
    if _tarefa_atual is not None:
        tarefa = {
            "objetivo": _tarefa_atual.objetivo,
            "status": _tarefa_atual.status,
            "passos": len(_tarefa_atual.passos),
        }
    return {
        "conectado": conectado,
        "vida": (snapshot or {}).get("vida"),
        "fome": (snapshot or {}).get("fome"),
        "posicao": (snapshot or {}).get("posicao"),
        "tarefa": tarefa,
        "iniciativa_ativa": _iniciativa_ativa,
        "eventos_na_fila": _eventos_jogo.qsize(),
    }


def definir_iniciativa(ativo: bool) -> None:
    """Liga/desliga o ciclo de iniciativa em tempo real -- ver
    _ciclo_iniciativa, que checa essa flag a cada volta em vez de só na
    inicialização. Chamado pelo dashboard (ver dashboard.py)."""
    global _iniciativa_ativa
    _iniciativa_ativa = ativo


def _decidir_iniciativa(estado: dict) -> dict | None:
    """Síncrona de propósito, mesmo motivo de _decidir_proximo_passo:
    completar_com_reserva é HTTP bloqueante. Antes isso era chamado
    DIRETO dentro da corrotina, congelando a recepção do WebSocket
    (nenhum snapshot entrando, nenhuma ação resolvendo) durante todo o
    tempo que o modelo levasse pra responder."""
    try:
        from ..decision import completar_com_reserva
        prompt = PROMPT_INICIATIVA.format(estado=json.dumps(estado, ensure_ascii=False))
        principal, reserva = _clientes_para_decidir_passo()
        bruto = completar_com_reserva(principal, reserva, prompt)
        m = re.search(r"\{.*\}", bruto, re.S)
        if not m:
            return None
        return json.loads(m.group(0))
    except Exception as e:
        print(f"[iniciativa] decisor falhou: {type(e).__name__}: {e}")
        return None


async def _ciclo_iniciativa() -> None:
    """Roda pra sempre em segundo plano (só se EVA_MC_INICIATIVA=1),
    checando a cada INICIATIVA_INTERVALO_S segundos se ela quer começar
    algo por conta própria. Nunca interrompe tarefa já ativa, nunca age
    com vida baixa -- mesmas travas de segurança do resto do sistema.
    """
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(INICIATIVA_INTERVALO_S)
        if not _iniciativa_ativa:
            continue  # desligada agora -- só espera o próximo ciclo, não decide nada
        try:
            if _tarefa_atual is not None and _tarefa_atual.status == "ativa":
                continue  # já ocupada, não interrompe pra iniciar outra coisa
            if _cliente is None or not _cliente.conectado or _cliente.ultimo_snapshot is None:
                continue  # sem conexão/estado real, não tem base pra decidir nada

            estado = _cliente.ultimo_snapshot
            if estado.get("vida", 20) <= VIDA_MINIMA_SEGURA:
                continue  # não inicia nada por conta própria machucada

            decisao = await loop.run_in_executor(None, _decidir_iniciativa, estado)
            if not decisao:
                continue
            if decisao.get("agir") and decisao.get("objetivo"):
                objetivo = str(decisao["objetivo"])
                print(f"[iniciativa] decidiu agir por conta própria: {objetivo}")
                _emitir_evento_jogo(
                    f"resolvi fazer uma coisa no Minecraft por conta própria: {objetivo}")
                asyncio.create_task(_executar_tarefa(objetivo))
            # agir=false não gera log nenhum de propósito -- "decidiu não
            # fazer nada" a cada 5 minutos seria ruído constante no
            # console, e é o resultado ESPERADO na maior parte do tempo.
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[iniciativa] ciclo falhou, tenta de novo no próximo intervalo: {e}")