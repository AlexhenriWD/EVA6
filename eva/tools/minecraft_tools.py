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
import os
import threading
import time

from ..integrations.minecraft_client import ClienteMinecraft
from .registry import registro

_cliente: ClienteMinecraft | None = None
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_pronto = threading.Event()


def _iniciar_thread_minecraft() -> None:
    """Sobe a thread com o event loop dedicado, uma vez só, preguiçoso --
    só na primeira chamada de ferramenta de Minecraft de verdade."""
    global _thread
    if _thread is not None:
        return

    def rodar_loop():
        global _cliente, _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        url = os.environ.get("EVA_MC_BRIDGE_URL", "ws://localhost:8766")
        _cliente = ClienteMinecraft(url)
        _cliente.mostrar_snapshot_automatico = False  # sem terminal interativo aqui, não tem por quê imprimir
        _pronto.set()
        # Sempre agendado, independente do valor inicial -- controle de
        # liga/desliga agora é por dentro (_iniciativa_ativa, checado a
        # cada ciclo), não por fora. Isso é o que permite o dashboard
        # alternar em tempo real, sem precisar reiniciar (diferente da
        # visão, que só decide na inicialização).
        _loop.create_task(_ciclo_iniciativa())
        _loop.run_until_complete(_cliente.rodar())

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
    exatamente pra isso, é a forma seguro de fazer isso sem race
    condition. Nunca levanta exceção: erro de conexão vira {"erro": ...},
    mesma regra de toda ferramenta deste projeto (ver registry.py).
    """
    _iniciar_thread_minecraft()
    if _cliente is None or _loop is None:
        return {"erro": "minecraft_indisponivel", "detalhe": "thread de conexão não iniciou a tempo"}
    if not _cliente.conectado and not _esperar_conexao():
        return {"erro": "minecraft_desconectado",
                "detalhe": "bridge não conectado -- node minecraft_bridge.js está rodando?"}
    try:
        futuro = asyncio.run_coroutine_threadsafe(
            _cliente.enviar_acao(tipo, timeout=timeout, **campos), _loop)
        return futuro.result(timeout=timeout + 2)
    except Exception as e:
        return {"erro": "minecraft_falha", "detalhe": str(e)[:200]}


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
    return _cliente.ultimo_snapshot


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
    return _chamar("mover_para", timeout=60, x=x, y=y, z=z)


@registro.adicionar(
    "minecraft_minerar",
    "Minera um tipo de bloco por perto (nome exato do jogo, ex: oak_log, "
    "iron_ore, stone). Anda até o mais próximo do tipo pedido e quebra.",
    {"bloco": "nome do bloco, ex: oak_log", "quantidade": "quantos minerar (padrão 1)"},
)
def minecraft_minerar(bloco: str, quantidade: int = 1) -> dict:
    return _chamar("minerar", bloco=bloco, quantidade=int(quantidade), timeout=60)


@registro.adicionar(
    "minecraft_craftar",
    "Fabrica um item a partir do que tem no inventário (nome exato do "
    "jogo, ex: wooden_pickaxe, oak_planks, crafting_table). Usa mesa de "
    "trabalho por perto se a receita precisar e tiver uma por perto.",
    {"item": "nome do item a craftar", "quantidade": "quantos (padrão 1)"},
)
def minecraft_craftar(item: str, quantidade: int = 1) -> dict:
    return _chamar("craftar", item=item, quantidade=int(quantidade))


@registro.adicionar(
    "minecraft_equipar",
    "Equipa um item do inventário na mão (ex: uma picareta antes de "
    "minerar, uma espada antes de atacar).",
    {"item": "nome do item"},
)
def minecraft_equipar(item: str) -> dict:
    return _chamar("equipar", item=item)


@registro.adicionar(
    "minecraft_atacar",
    "Ataca a entidade viva mais próxima que bater o nome pedido (mob como "
    "chicken/zombie, ou nome de jogador). Se aproxima antes de golpear.",
    {"alvo": "nome da entidade ou jogador"},
)
def minecraft_atacar(alvo: str) -> dict:
    # Mesmo motivo do minecraft_mover: também se aproxima via goto() antes
    # de golpear, sujeito ao mesmo timeout curto demais.
    return _chamar("atacar", timeout=60, alvo=alvo)


@registro.adicionar(
    "minecraft_seguir",
    "Passa a seguir um jogador pelo nome, continuamente, até receber outra "
    "ordem (minecraft_parar ou outra ação).",
    {"jogador": "nome do jogador"},
)
def minecraft_seguir(jogador: str) -> dict:
    return _chamar("seguir", jogador=jogador)


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
    "servidor -- use pra avisar o que vai fazer, o que fez, ou o que "
    "precisa, quando fizer sentido comunicar isso pros jogadores.",
    {"texto": "o que falar no chat do jogo"},
)
def minecraft_falar(texto: str) -> dict:
    return _chamar("falar_no_jogo", texto=texto)


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
# Roda como coroutine solta no MESMO loop dedicado da conexão (import
# asyncio.create_task no _loop já ativo) -- não bloqueia o turno de
# conversa que a disparou. Comunicação de progresso é direto no chat do
# jogo (minecraft_falar), não pela resposta conversacional -- a tarefa
# pode levar minutos, muito além de qualquer turno de resposta.

import json
import re
from dataclasses import dataclass, field

from ..config import DecisionConfig


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


PROMPT_PASSO = """Você está controlando um corpo no Minecraft, executando uma tarefa passo a passo. Nunca invente coordenada, bloco ou item que não esteja no "Estado atual" abaixo -- se precisar de algo que não está lá, o passo certo é olhar de novo, não chutar.

Objetivo da tarefa: {objetivo}

Estado atual do jogo (real, agora):
{estado}

Passos já tentados nesta tarefa, do mais antigo pro mais recente:
{historico}

Ferramentas disponíveis:
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
        linhas.append(f"- {p.ferramenta}({p.args}) -> {marca}: {p.resultado.get('detalhe', '')}")
    return "\n".join(linhas)


def _decidir_proximo_passo(tarefa: TarefaMinecraft, estado: dict) -> dict:
    """Uma chamada de LLM, focada, com o estado real na mão. Nunca levanta
    exceção -- erro vira {"desistir": True, ...}, tratado como qualquer
    outra falha de passo pelo loop.

    Motivo de erro fica CURTO de propósito -- confirmado em teste real
    que o erro cru (JSON de resposta HTTP inteiro) foi parar direto no
    chat do jogo via minecraft_falar, ilegível. Corta pra uma linha.
    """
    prompt = PROMPT_PASSO.format(
        objetivo=tarefa.objetivo,
        estado=json.dumps(estado, ensure_ascii=False),
        historico=_formatar_historico(tarefa.passos),
        ferramentas=registro.descrever(),
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


async def _executar_tarefa(objetivo: str) -> None:
    global _tarefa_atual
    tarefa = TarefaMinecraft(objetivo=objetivo)
    _tarefa_atual = tarefa

    falhas_seguidas = 0
    while tarefa.status == "ativa" and len(tarefa.passos) < MAX_PASSOS_TAREFA:
        if _cliente is not None and _cliente.ultimo_snapshot is None:
            await _esperar_snapshot_async()
        estado = _cliente.ultimo_snapshot if _cliente else None
        if estado is None:
            tarefa.status = "falhou"
            _chamar("falar_no_jogo", texto=f"Preciso parar -- perdi o estado do jogo. ({objetivo})")
            break

        # trava de segurança: vida baixa aborta na hora, sem exceção --
        # motivada por incidente real (perdeu 12 de vida numa sessão de
        # teste sem nenhum passo de segurança existir ainda).
        if estado.get("vida", 20) <= VIDA_MINIMA_SEGURA:
            tarefa.status = "cancelada"
            _chamar("falar_no_jogo",
                    texto=f"Parando -- vida baixa ({estado.get('vida')}/20). Preciso de ajuda ou vou me recuperar antes.")
            break

        decisao = await asyncio.get_event_loop().run_in_executor(
            None, _decidir_proximo_passo, tarefa, estado)

        if decisao.get("concluido"):
            tarefa.status = "concluida"
            _chamar("falar_no_jogo", texto=f"Pronto -- {objetivo}.")
            break
        if decisao.get("desistir"):
            tarefa.status = "falhou"
            _chamar("falar_no_jogo", texto=f"Não consegui -- {objetivo}. {decisao.get('motivo', '')}".strip())
            break

        nome = decisao.get("ferramenta")
        args = decisao.get("args") or {}
        funcao = _FERRAMENTAS_TAREFA.get(nome)
        if funcao is None:
            resultado = {"sucesso": False, "detalhe": f"ferramenta '{nome}' não existe"}
        else:
            resultado = await asyncio.get_event_loop().run_in_executor(
                None, lambda: funcao(**args))

        tarefa.passos.append(PassoTarefa(ferramenta=nome, args=args, resultado=resultado))

        if resultado.get("sucesso"):
            falhas_seguidas = 0
        else:
            falhas_seguidas += 1
            if falhas_seguidas >= MAX_FALHAS_SEGUIDAS:
                tarefa.status = "falhou"
                _chamar("falar_no_jogo",
                        texto=f"Desistindo -- {objetivo}. {falhas_seguidas} passos seguidos sem sucesso.")
                break

    if tarefa.status == "ativa":
        tarefa.status = "esgotou_passos"
        _chamar("falar_no_jogo", texto=f"Parando por agora -- {objetivo} ainda não terminou, muitos passos.")


# Só as ferramentas de ação atômica -- minecraft_tarefa/status/cancelar
# ficam de fora de propósito, a tarefa não pode chamar a si mesma nem
# consultar seu próprio status como se fosse um passo de execução.
_FERRAMENTAS_TAREFA = {
    "minecraft_estado": minecraft_estado,
    "minecraft_mover": minecraft_mover,
    "minecraft_minerar": minecraft_minerar,
    "minecraft_craftar": minecraft_craftar,
    "minecraft_equipar": minecraft_equipar,
    "minecraft_atacar": minecraft_atacar,
    "minecraft_seguir": minecraft_seguir,
    "minecraft_parar": minecraft_parar,
    "minecraft_falar": minecraft_falar,
}


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
    if not _cliente.conectado and not _esperar_conexao():
        return {"erro": "minecraft_desconectado", "detalhe": "bridge não conectado"}
    if _tarefa_atual is not None and _tarefa_atual.status == "ativa":
        return {"erro": "tarefa_ja_ativa",
                "detalhe": f"já tem uma tarefa rodando: {_tarefa_atual.objetivo} -- "
                           f"use minecraft_cancelar_tarefa antes de iniciar outra"}
    asyncio.run_coroutine_threadsafe(_executar_tarefa(objetivo), _loop)
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
    }


def definir_iniciativa(ativo: bool) -> None:
    """Liga/desliga o ciclo de iniciativa em tempo real -- ver
    _ciclo_iniciativa, que checa essa flag a cada volta em vez de só na
    inicialização. Chamado pelo dashboard (ver dashboard.py)."""
    global _iniciativa_ativa
    _iniciativa_ativa = ativo


async def _ciclo_iniciativa() -> None:
    """Roda pra sempre em segundo plano (só se EVA_MC_INICIATIVA=1),
    checando a cada INICIATIVA_INTERVALO_S segundos se ela quer começar
    algo por conta própria. Nunca interrompe tarefa já ativa, nunca age
    com vida baixa -- mesmas travas de segurança do resto do sistema.
    """
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

            prompt = PROMPT_INICIATIVA.format(estado=json.dumps(estado, ensure_ascii=False))
            from ..decision import completar_com_reserva
            principal, reserva = _clientes_para_decidir_passo()
            bruto = completar_com_reserva(principal, reserva, prompt)
            m = re.search(r"\{.*\}", bruto, re.S)
            if not m:
                continue
            decisao = json.loads(m.group(0))
            if decisao.get("agir") and decisao.get("objetivo"):
                print(f"[iniciativa] decidiu agir por conta própria: {decisao['objetivo']}")
                asyncio.create_task(_executar_tarefa(decisao["objetivo"]))
            # agir=false não gera log nenhum de propósito -- "decidiu não
            # fazer nada" a cada 5 minutos seria ruído constante no
            # console, e é o resultado ESPERADO na maior parte do tempo.
        except Exception as e:
            print(f"[iniciativa] ciclo falhou, tenta de novo no próximo intervalo: {e}")