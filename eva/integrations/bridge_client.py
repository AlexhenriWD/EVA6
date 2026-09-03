"""
Cliente do EVA Voice Bridge.

O bridge (Node + discord.js) cuida de toda a parte de voz do Discord --
que é onde o discord.py quebra. Este módulo é o lado Python: recebe o
áudio capturado, roda o ciclo cognitivo da EVA e devolve a fala.

PROTOCOLO
---------
WebSocket em ws://localhost:8765. Mensagens JSON, mais quadros binários
para áudio.

Python -> Node:
    {"type":"join",  "guild_id":..., "channel_id":...}
    {"type":"leave", "guild_id":...}
    {"type":"play",  "guild_id":...}   seguido de um quadro BINÁRIO com PCM
    {"type":"stop_play", "guild_id":...}          corta a fala em andamento
    {"type":"fonte_de_eco", "guild_id":..., "user_id":..., "ttl_ms":N}

Node -> Python:
    {"type":"ready", "version":"2.0"}
    {"type":"joined", "guild_id":..., "channel_id":..., "channel_name":...}
    {"type":"left", "guild_id":..., "reason":...}
    {"type":"reconnecting", "guild_id":...}
    {"type":"reconnected", "guild_id":..., "channel_id":...}
    {"type":"error", "message":...}
    {"type":"audio", "guild_id":..., "user_id":..., "bytes":N,
     "durante_fala":bool}
        seguido de um quadro BINÁRIO com N bytes de PCM
    {"type":"play_done", "guild_id":..., "cortada":bool,
     "bytes_reproduzidos":N, "bytes_totais":N}

Áudio nos dois sentidos: PCM 48kHz, estéreo, s16le.

DECISÕES DE DESIGN
------------------
O ciclo da EVA (STT -> memória -> LLM -> TTS) leva segundos e é síncrono.
Rodá-lo direto no laço de eventos travaria o WebSocket, e o bridge ficaria
sem resposta -- inclusive para o keepalive. Por isso tudo que bloqueia vai
para thread via asyncio.to_thread.

FALA DA PESSOA NUNCA É DESCARTADA
---------------------------------
Antes, três guardas jogavam fora o áudio recebido: o bridge.js nem
assinava o microfone durante o playback, e aqui `est.falando` e
`est.processando.locked()` davam `return` seco. O efeito somado é o que
mais atrapalhava na call: a pessoa respondia enquanto a EVA pensava ou
falava, e a resposta dela simplesmente não existia -- sem log, sem aviso,
sem nada. Nenhuma das três guardas existe mais nessa forma. Hoje:

  * áudio que chega enquanto ela GERA (ainda sem som saindo) é transcrito
    e guardado em `est.pendentes`. A geração em curso é invalidada e as
    duas falas viram UM turno só -- ver `_drenar_pendentes`;

  * áudio que chega enquanto ela FALA passa antes pelo teste de eco
    (`_parece_eco`): se for a voz dela voltando pelo microfone de quem
    está em caixa de som, é descartado e aquele usuário é marcado como
    fonte de eco no bridge.js; se for fala de verdade, ela é INTERROMPIDA
    (`stop_play`) e o turno entra na fila de pendentes;

  * o que ela chegou a falar antes do corte substitui, no histórico, a
    resposta inteira que tinha sido gerada (`corrigir_ultimo_turno`) --
    senão o prompt do próximo turno teria como exemplo uma fala que
    nunca aconteceu.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import random
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

from ..config import EVAConfig, carregar_config
from ..orchestrator import EVA
from ..tools import minecraft_tools
from ..voice.audio import (
    ErroAudio,
    alinhar_frames,
    duracao_segundos,
    esta_silencioso,
    para_pcm_discord,
    pcm_para_wav,
)
from ..voice.stt import Transcricao, criar_stt, parece_ruido
from ..voice.tts import ErroTTS, criar_tts


def _carregar_robot_tools():
    """Import condicionado a EVA_ROBOT_ATIVO, mesma flag de
    builtin.py::carregar_ferramentas().

    ACHADO REAL (02/09/2026): `import robot_tools` incondicional aqui no
    topo do módulo (como era antes) bypassava por completo a proteção de
    EVA_ROBOT_ATIVO=0 em builtin.py. O motivo é ordem de import: Python
    executa o corpo de um módulo (inclusive os decoradores
    @registro.adicionar de cada robo_* -- ver robot_tools.py) na
    PRIMEIRA vez que ele é importado, e fica em cache em sys.modules daí
    pra frente. Como este arquivo importava `from ..tools import
    robot_tools` direto no topo -- ANTES de EVA() ser instanciada e
    ANTES de carregar_ferramentas() checar a flag --, o registro global
    de ferramentas (ver registry.py) já vinha com robo_ver/robo_olhar/
    etc. registrados no momento em que builtin.py rodava sua checagem.
    O resultado prático, visto em log real: mesmo com "[ferramentas]
    robot_tools não carregado (EVA_ROBOT_ATIVO=0)" impresso, o decisor
    ainda conseguiu chamar robo_ver() de verdade (erro "sem_video" vindo
    de DENTRO da função -- só possível se ela estava mesmo registrada).

    Este módulo (bridge_client.py) só usa quatro funções de robot_tools:
    robo_conectado(), definir_em_call(), definir_consciencia_callback(),
    drenar_eventos_corpo() -- nenhuma delas registra ferramenta nenhuma,
    são só consulta/config de estado. Com EVA_ROBOT_ATIVO=0, devolve um
    objeto simples com essas quatro como no-op seguro, em vez de
    importar o módulo real e disparar os decoradores.
    """
    if os.environ.get("EVA_ROBOT_ATIVO", "1") == "1":
        from ..tools import robot_tools
        return robot_tools

    class _RobotToolsDesativado:
        @staticmethod
        def robo_conectado() -> bool:
            return False

        @staticmethod
        def definir_em_call(ativo: bool) -> None:
            pass

        @staticmethod
        def definir_consciencia_callback(callback) -> None:
            pass

        @staticmethod
        def drenar_eventos_corpo() -> list:
            return []

    return _RobotToolsDesativado()


robot_tools = _carregar_robot_tools()


PROMPT_PUXAR_ASSUNTO = """Você está numa call. De vez em quando você considera se vale
comentar ou perguntar algo por conta própria, sem esperar ser provocada.

Como a conversa andou nos últimos turnos (pode estar vazio se ninguém falou
ainda, ou pode mostrar um assunto em andamento):
{conversa_recente}

O que você sabe de verdade sobre essa pessoa, de conversas passadas (não
invente nada além disso):
{fatos}

REGRA MAIS IMPORTANTE: se a conversa recente mostra um assunto ainda ativo
(as últimas falas são sobre um tema específico, não genérico), responda
null -- puxar um assunto novo por cima de um assunto vivo é o pior erro
possível aqui, pior que ficar em silêncio. Só sugira algo novo se a conversa
recente estiver vazia, tiver esfriado (última fala foi um fechamento, tipo
"tá bom" ou "valeu"), ou não houver conversa recente nenhuma.

Se for sugerir algo, pode ser: (a) uma pergunta/comentário sobre algo da
memória de longo prazo (fatos acima), OU (b) uma continuação natural do que
já estava sendo dito -- nunca as duas coisas misturadas, e nunca um tema
aleatório desconectado de tudo.

Responda APENAS um JSON:
{{"assunto": "pergunta ou comentário pronto pra dizer, em primeira pessoa"}}
ou
{{"assunto": null}}"""


def _normalizar_para_eco(texto: str) -> list[str]:
    """Texto virando lista de palavras comparável: sem acento, sem
    pontuação, minúsculo. O STT e o TTS quase nunca concordam na
    pontuação, e acento some ou aparece dependendo de como o Whisper
    ouviu -- comparar a string crua daria falso negativo demais."""
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )
    return re.findall(r"[a-z0-9]+", sem_acento)


def _parece_eco(ouvido: str, falado: str) -> bool:
    """A transcrição que chegou é a própria voz da EVA voltando?

    É o teste que torna a interrupção viável com alguém em caixa de som.
    Sem ele, a única alternativa seria manter o microfone fechado durante
    a fala dela -- que é exatamente o comportamento antigo, o que impedia
    interromper.

    Compara sequências de palavras e olha o maior bloco em comum. Eco é
    sempre um TRECHO CONTÍGUO do que ela está dizendo (o microfone pegou
    um pedaço do meio da fala), então bloco contíguo grande = eco;
    palavras espalhadas que coincidem por acaso = fala de verdade.

    Assimetria proposital nos dois erros possíveis:
      * achar que é eco quando era fala real -> a pessoa é ignorada de
        novo, que é o bug que estamos consertando. Caro.
      * achar que é fala real quando era eco -> ela se interrompe à toa
        uma vez, e o usuário vira fonte de eco marcada. Barato e
        auto-corrigível.
    Por isso o limiar é conservador e falas curtas exigem casamento
    exato: "sim", "verdade", "pois é" são coisas que a pessoa fala por
    cima dela o tempo todo, e barrar isso seria pior que o eco.
    """
    palavras_ouvidas = _normalizar_para_eco(ouvido)
    palavras_faladas = _normalizar_para_eco(falado)
    if not palavras_ouvidas or not palavras_faladas:
        return False

    matcher = SequenceMatcher(None, palavras_faladas, palavras_ouvidas,
                              autojunk=False)
    maior = max((b.size for b in matcher.get_matching_blocks()), default=0)

    if len(palavras_ouvidas) < 3:
        # Curto demais pra ter estatística: só conta como eco se TODAS as
        # palavras aparecerem coladas dentro da fala dela.
        return maior == len(palavras_ouvidas)

    return (maior / len(palavras_ouvidas)) >= 0.6


def _cortar_texto_na_fracao(texto: str, fracao: float) -> str:
    """Trecho do texto que corresponde à parte que virou som.

    Corta na fronteira de frase mais próxima do ponto, não no caractere
    exato: gravar meia palavra no histórico seria pior que arredondar. Se
    nenhuma frase inteira coube, devolve a primeira -- ela começou a
    falar, então alguma coisa saiu.
    """
    texto = texto.strip()
    if fracao >= 0.98 or not texto:
        return texto
    if fracao <= 0.02:
        return ""

    limite = max(1, int(len(texto) * fracao))
    frases = re.findall(r"[^.!?…]+[.!?…]*\s*", texto) or [texto]

    acumulado = ""
    for frase in frases:
        if len(acumulado) + len(frase) > limite and acumulado:
            break
        acumulado += frase
    return (acumulado or frases[0]).strip()


@dataclass
class EstadoGuild:
    canal_id: str | None = None
    falando: bool = False
    # Instante em que a fala terminou. Usado para ignorar o eco por um
    # tempinho depois -- o Discord continua entregando pacotes atrasados.
    fim_da_fala: float = 0.0
    # Instante em que o STT terminou de transcrever a fala atual -- ponto
    # de partida pra medir "tempo até o primeiro som" (ver `falar` e
    # `_falar_stream`). Fica no estado da guild, não como parâmetro
    # passado adiante por 3-4 funções, porque só existe um turno de voz
    # em andamento por guild de cada vez (`est.processando` já garante
    # isso).
    t_stt_fim: float = 0.0
    processando: asyncio.Lock = field(default_factory=asyncio.Lock)

    # --- interrupção e turnos pendentes ---
    # O que ela está falando NESTE instante. Só serve para o teste de eco
    # (`_parece_eco`) -- é a única forma de distinguir "a pessoa falou por
    # cima dela" de "a voz dela voltou pelo microfone de quem está em
    # caixa de som", já que os dois chegam aqui como áudio idêntico vindo
    # de um user_id humano.
    texto_em_voz: str = ""
    # Última fala já terminada. O eco chega ATRASADO -- a captura só fecha
    # depois do silêncio, então pacotes de eco aparecem aqui depois do
    # play_done, quando `texto_em_voz` já foi limpo. Sem guardar isso, o
    # primeiro eco pós-fala passaria como fala legítima e ela responderia
    # à própria voz.
    ultimo_texto_falado: str = ""
    # Bytes de PCM entregues ao bridge nesta fala -- com os bytes que o
    # Node reporta como REPRODUZIDOS, dá a fração da fala que virou som,
    # e daí o ponto de corte no texto.
    bytes_em_voz: int = 0
    # Marcado quando um corte foi pedido; lido no play_done pra decidir se
    # corrige o histórico.
    corte_pedido: bool = False
    # Sinaliza pra thread de síntese (streaming) parar de gerar frases
    # novas. threading.Event, não asyncio: quem lê roda em to_thread.
    cortar: threading.Event = field(default_factory=threading.Event)
    # Resposta gerada mas ainda não falada por inteiro, junto do usuário
    # dela -- o par que `_corrigir_historico_apos_corte` precisa.
    usuario_em_voz: str | None = None

    # Falas capturadas enquanto ela estava ocupada, esperando a vez:
    # (user_id, texto, timestamp).
    pendentes: list = field(default_factory=list)
    tarefa_drenagem: asyncio.Task | None = None
    # A geração em curso não vale mais: chegou fala nova antes dela virar
    # som, e as duas viram um turno só.
    geracao_invalidada: bool = False
    # A mensagem que originou a geração em curso -- volta pra fila de
    # pendentes quando ela é invalidada, pra ser reperguntada junto.
    mensagem_em_curso: str | None = None

    # --- multicanal: agregação quando mais de uma pessoa fala por perto ---
    # user_id -> timestamp da última fala dessa pessoa. Usado só para
    # decidir "tem mais gente ativa agora?" -- não é histórico de verdade,
    # memória isso já existe em outro lugar.
    ultima_fala_por: dict = field(default_factory=dict)
    # (user_id, texto, timestamp) acumulado enquanto a janela de
    # agregação está aberta.
    buffer_multicanal: list = field(default_factory=list)
    tarefa_agregacao: asyncio.Task | None = None


class ClienteBridge:
    def __init__(self, config: EVAConfig | None = None, url: str = "ws://localhost:8765"):
        self.cfg = config or carregar_config()
        self.url = url
        self.eva = EVA(self.cfg)
        # Tarefas de fundo (extração de memória) consultam isto antes de
        # gastar GPU -- ver EVA._esperar_ocioso pro número medido. Fica
        # aqui, e não dentro do orchestrator, porque quem sabe se existe
        # turno em andamento é o cliente de voz, não o núcleo cognitivo.
        self.eva.ocupado_agora = self._ocupada_em_alguma_guild

        self.stt, aviso_stt = criar_stt(self.cfg.voz)
        if aviso_stt:
            print(f"[stt] {aviso_stt}")
        self.tts = None
        self.erro_tts: str | None = None
        try:
            self.tts = criar_tts(
                backend=self.cfg.voz.tts_backend or None,
                idioma=self.cfg.voz.tts_idioma,
                quantizar=self.cfg.voz.tts_quantize,
            )
            print(f"[tts] backend ativo: {self.tts.nome} "
                  f"(EVA_TTS_BACKEND={self.cfg.voz.tts_backend!r})")
        except ErroTTS as e:
            self.erro_tts = str(e)
            # ACHADO REAL: antes disto, um ErroTTS aqui (chave da Cartesia
            # faltando, EVA_TTS_CARTESIA_VOICE errado, etc.) ficava só
            # guardado em self.erro_tts, sem UM print sequer -- só
            # aparecia se alguém abrisse o dashboard e olhasse
            # diagnostico(). É o motivo mais provável de "troquei
            # EVA_TTS_BACKEND e não vejo diferença nenhuma": o backend
            # pedido falhou ao criar, self.tts virou None, e a EVA fica
            # muda em silêncio -- sem erro nenhum no console apontando
            # o motivo. Mesmo padrão do EVA_DECISION_GROQ com chave em
            # branco que já apareceu neste projeto.
            print(f"[tts] ERRO ao carregar backend {self.cfg.voz.tts_backend!r}: {e}")
            print(f"[tts] EVA vai ficar SEM VOZ até isso ser corrigido "
                  f"(dashboard mostra o mesmo erro em '/', seção Voz).")

        self.ws = None
        self.guilds: dict[str, EstadoGuild] = {}
        from ..consciousness import Consciencia
        self.consciencias: dict[str, Consciencia] = {}
        self._tarefas_consciencia: dict[str, asyncio.Task] = {}
        self._ultima_curiosidade: dict[str, float] = {}
        self._curiosidades_usadas: dict[str, set[str]] = {}
        self._ultimo_assunto: dict[str, float] = {}

        # Visão: uma instância só (há uma tela, a do PC rodando a EVA --
        # não uma por guild como Consciencia, que é por canal porque cada
        # call tem seu próprio silêncio). Criada mesmo com visao.ativa=False
        # para simplificar o resto do código (sempre existe, só não faz
        # nada) -- só falha e vira None se a dependência mss não estiver
        # instalada, e mesmo aí o resto do sistema continua funcionando
        # sem visão.
        self.visao = None
        self.erro_visao: str | None = None
        if self.cfg.visao.ativa:
            try:
                from ..vision.visao import SistemaVisual
                self.visao = SistemaVisual(self.cfg)
            except ImportError as e:
                self.erro_visao = (
                    f"visão ligada em config mas dependência faltando: {e}. "
                    f"Rode: pip install mss pillow numpy"
                )
        self._tarefa_visao: asyncio.Task | None = None

        # Visão do ROBÔ -- flag separada de propósito (ver
        # VisaoConfig.robo_ativa): call sem robô nenhum não deve rodar
        # este laço nem gerar log nenhum. Não depende de mss (a fonte é
        # a câmera do robô via rede, não o PC local) -- só falha se
        # PIL/numpy faltarem, o que já quebraria a visão de tela também.
        self.visao_robo = None
        self.erro_visao_robo: str | None = None
        if self.cfg.visao.robo_ativa:
            try:
                from ..vision.visao_robo import SistemaVisualRobo
                self.visao_robo = SistemaVisualRobo(self.cfg)
            except ImportError as e:
                self.erro_visao_robo = f"visão do robô ligada em config mas dependência faltando: {e}"
        self._tarefa_visao_robo: asyncio.Task | None = None
        self._tarefa_corpo: asyncio.Task | None = None
        self._tarefa_jogo: asyncio.Task | None = None
        self._dashboard = None  # criado em rodar(), só se cfg.dashboard.ativa
        self._guilds_com_call: set[str] = set()

        # O bridge manda o cabeçalho JSON e logo depois o quadro binário.
        # Guardamos o cabeçalho para saber de quem é o áudio que vem a seguir.
        self._audio_pendente: dict | None = None
        # PROTEGE O SENTIDO CONTRÁRIO: cabeçalho JSON (play/play_start/
        # play_chunk) + quadro binário que o BRIDGE.JS espera receber em
        # sequência imediata (pendingAudioFor, do lado dele, também não é
        # fila -- é UMA variável, sobrescrita pelo cabeçalho mais recente).
        # self.ws é uma conexão só, compartilhada entre falar() (fala
        # espontânea, chamada pelo laço de consciência) e _falar_stream()
        # (resposta em streaming) -- sem lock, um `await` no meio dos dois
        # sends (cabeçalho, then payload) dá brecha pro event loop trocar
        # de tarefa e a OUTRA tarefa mandar o cabeçalho dela no meio,
        # sobrescrevendo pendingAudioFor antes do primeiro payload sair.
        # O binário chega depois pro cabeçalho ERRADO -- Node toca no guild
        # errado, ou o parsing sai torto, dependendo do timing exato. Não é
        # o bug confirmado desta rodada (esse foi a GPU/ROCm), mas é risco
        # real sempre que consciência e resposta normal puderem coincidir
        # no tempo, então corrigido de qualquer forma -- ver falar() e
        # _falar_stream() abaixo, agora sempre dentro deste lock.
        self._envio_audio_lock = asyncio.Lock()
        # Silêncio a ignorar após a EVA falar, em segundos
        self.janela_eco = 0.6
        # Multicanal: se ALGUÉM MAIS falou dentro desse tempo, a fala atual
        # entra em modo agregação em vez de responder na hora -- é o que
        # decide "isso parece uma troca entre várias pessoas" vs "uma
        # pessoa só falando", que é o caso comum e não deve ganhar espera
        # nenhuma.
        self.janela_multiplas_pessoas = 5.0
        # Quanto esperar por MAIS fala (de qualquer pessoa) antes de
        # fechar a janela e mandar tudo agregado -- reinicia a cada fala
        # nova que chega, então é silêncio de todo mundo por esse tempo,
        # não um teto fixo desde a primeira fala.
        self.janela_debounce_multicanal = 1.8
        # Quanto tempo um usuário fica marcado como fonte de eco no
        # bridge.js depois de uma transcrição confirmada como eco. Curto
        # de propósito: se a pessoa colocar o fone, ela volta a poder
        # interromper em menos de um minuto, sem precisar de nenhum
        # comando nem reiniciar nada.
        self.ttl_fonte_de_eco = 45.0

    def estado(self, guild_id: str) -> EstadoGuild:
        return self.guilds.setdefault(str(guild_id), EstadoGuild())

    def consciencia(self, guild_id: str):
        from ..consciousness import Consciencia
        gid = str(guild_id)
        if gid not in self.consciencias:
            self.consciencias[gid] = Consciencia(self.cfg, canal=gid)
        return self.consciencias[gid]

    def _nome(self, usuario: str) -> str | None:
        """Nome da pessoa, se a EVA já souber. A voz do Discord só entrega
        user_id -- o nome vem da tabela `pessoas`, alimentada pelo texto."""
        try:
            return self.eva.memoria.pessoa(str(usuario)).get("nome") or None
        except Exception:
            return None

    async def _transcrever(self, audio: bytes, nome: str = "call.wav") -> Transcricao:
        """Transcreve com UMA retentativa rápida antes de desistir.

        Falha transitória de rede é o caso comum na Groq (ausente no
        whisper.cpp local, que não depende de rede) -- não deveria
        custar o turno de voz inteiro. Levanta a exceção da ÚLTIMA
        tentativa se as duas falharem, para quem chama continuar
        tratando do jeito que já tratava (a assinatura de erro não
        muda, só passa a acontecer depois de 2 tentativas em vez de 1).

        ACHADO REAL: o log do servidor do LM Studio não mostra ISTO --
        Groq e whisper.cpp nunca passam pela porta do LM Studio, então
        aquele log só começa a contar a partir de quando o texto já
        existe. O print abaixo fecha a outra metade: sem ele, "1 a 3
        segundos entre eu falar e a EVA responder" ficava sem forma de
        separar quanto é transcrição e quanto é o resto do pipeline.
        """
        ultimo_erro: Exception | None = None
        _t0 = time.time()
        for tentativa in range(2):
            try:
                t = await asyncio.to_thread(
                    self.stt.transcrever_bytes, audio, nome,
                    self.cfg.voz.stt_vocabulario,
                )
                print(f"[tempo] stt ({self.cfg.voz.stt_backend}): "
                      f"{int((time.time() - _t0) * 1000)}ms")
                return t
            except Exception as e:
                ultimo_erro = e
                if tentativa == 0:
                    print(f"[stt] falhou, tentando de novo: {e}")
                    await asyncio.sleep(0.3)
        raise ultimo_erro

    async def _contexto_visual_para(self, texto: str) -> str | None:
        """A cena para injetar neste turno, ou None. Roteia entre visão
        de TELA e visão do ROBÔ conforme o assunto da mensagem -- as
        duas fontes nunca fazem sentido juntas no mesmo turno (ou a
        pergunta é sobre o que está no monitor, ou é sobre o que o robô
        está vendo, não as duas).
        """
        from ..decision import robo_estado_relevante, robo_olhar_relevante, visao_relevante

        # ROBÔ CONECTADO = a visão dela é a do robô, ponto. Não roteia
        # mais por assunto da mensagem.
        #
        # O roteamento antigo (por palavra na fala) deixava a visão de
        # TELA responder qualquer pergunta que não citasse o robô
        # explicitamente -- e com o robô conectado ela tem um corpo
        # olhando pra um lugar; o que está no monitor não é o que ela
        # está vendo. Em uso real isso produziu descrição de tela
        # apresentada como se fosse a cena à frente do robô.
        #
        # Também é o corte mais barato de latência que existe aqui: as
        # duas visões disputam o mesmo servidor gemma-3-4b, e cada tick
        # de tela custava ~3s de inferência por turno.
        if robot_tools.robo_conectado():
            return await self._contexto_visual_robo_para(texto)

        if robo_olhar_relevante(texto) or robo_estado_relevante(texto):
            return await self._contexto_visual_robo_para(texto)

        if self.visao is None:
            return None

        if visao_relevante(texto):
            fresca = await asyncio.to_thread(self.visao.analisar_agora)
            return fresca or self.visao.contexto_atual()

        cena = self.visao.cena
        if cena and cena.idade_segundos() < self.cfg.visao.janela_relevancia_recente:
            return self.visao.contexto_atual()

        return None

    async def _contexto_visual_robo_para(self, texto: str) -> str | None:
        """Mesmo padrão de _contexto_visual_para, só que na câmera do
        robô: pergunta direta (aqui, QUALQUER pergunta que chegou até
        este ponto já é sobre o robô -- ver o roteamento acima) vale uma
        análise fresca sob demanda, não só confiar no último tick de
        fundo. Se o robô não estiver conectado, analisar_agora() falha
        silenciosamente (best-effort) e contexto_atual() devolve None --
        a pessoa recebe a resposta normal sem contexto visual, não um
        erro.
        """
        if self.visao_robo is None:
            return None
        fresca = await asyncio.to_thread(self.visao_robo.analisar_agora)
        return fresca or self.visao_robo.contexto_atual()

    async def _pesquisar_e_registrar(self, guild_id: str, consulta: str) -> None:
        """Pesquisa em fundo e, se achar algo, alimenta a Consciencia.

        Roda solto via create_task -- exceção aqui nunca deve propagar pro
        chamador, porque quem chamou já seguiu em frente. Falha aqui é
        equivalente a "não achou nada": silenciosa, sem efeito colateral
        visível para o usuário.
        """
        try:
            resumo = await self.eva.pesquisar_lacuna(consulta)
        except Exception as e:
            if self.cfg.debug:
                print(f"[lacuna] erro: {e}")
            return
        if resumo:
            self.consciencia(guild_id).pesquisa_pronta(resumo)
            if self.cfg.debug:
                print(f"[lacuna] impulso registrado: {resumo[:80]}")

    def _tentar_curiosidade(self, guild_id: str, c) -> None:
        """Dispara pesquisa por conta própria quando não há fio nem evento
        pendurado -- é a versão PROATIVA do que hoje só existe REATIVO
        (possivel_lacuna, depois de uma mensagem do usuário). Não decide
        falar agora: só alimenta a fila pro(s) PRÓXIMO(s) tick(s), com força
        de "pesquisa" (0.60) em vez do "vazio" (0.35) de sempre.
        """
        agora = time.time()
        if agora - self._ultima_curiosidade.get(guild_id, 0.0) < self.cfg.consciencia.cooldown_curiosidade:
            return
        if any(not f.usado for f in c.fios):
            return

        usados = self._curiosidades_usadas.setdefault(guild_id, set())
        topico = self._topico_de_curiosidade(
            c.ultimo_falante or self.cfg.usuario, usados)
        if not topico:
            return

        self._ultima_curiosidade[guild_id] = agora
        usados.add(topico)
        asyncio.create_task(self._pesquisar_e_registrar(guild_id, topico))
        if self.cfg.debug:
            print(f"[curiosidade] pesquisando sozinha: {topico[:80]!r}")

    def _tentar_puxar_assunto(self, guild_id: str, c) -> None:
        """Decide se vale puxar assunto -- olhando tanto a memória de longo
        prazo quanto a conversa recente de verdade, pra não sugerir um
        tema desconectado por cima de um assunto que ainda está ativo."""
        agora = time.time()
        if (agora - self._ultimo_assunto.get(guild_id, 0.0)
                < self.cfg.consciencia.cooldown_assunto):
            return
        if any(not f.usado for f in c.fios):
            return

        usuario = c.ultimo_falante or self.cfg.usuario
        try:
            fatos = self.eva.memoria.listar(usuario=usuario, limite=10)
        except Exception:
            fatos = []
        try:
            historico = self.eva.memoria.historico(
                usuario=usuario, limite=self.cfg.memoria.janela_historico)
        except Exception:
            historico = []
        # Sem fatos de longo prazo E sem conversa recente: não há matéria
        # nenhuma pro modelo trabalhar em cima, então nem vale gastar a
        # chamada. Qualquer um dos dois presentes já é o bastante -- é o
        # próprio prompt (PROMPT_PUXAR_ASSUNTO) quem decide se um assunto
        # novo cabe ou se a resposta certa é null.
        if not fatos and not historico:
            return

        self._ultimo_assunto[guild_id] = agora
        asyncio.create_task(self._decidir_assunto(guild_id, usuario, fatos, historico))

    async def _decidir_assunto(self, guild_id: str, usuario: str, fatos, historico) -> None:
        try:
            resumo = "\n".join(f"- {f.conteudo[:150]}" for f in fatos) or "(nada registrado ainda)"
            # Só os últimos turnos importam aqui -- é "a conversa esfriou
            # ou ainda está quente", não um resumo completo da sessão.
            # Mesmo formato role/content que o resto do projeto usa
            # (self.eva.memoria.historico), convertido pra prosa simples
            # porque quem lê isto é um prompt de decisão, não um chat
            # real com roles.
            ultimos = historico[-6:] if historico else []
            if ultimos:
                linhas_conversa = []
                for turno in ultimos:
                    quem = "ela" if turno.get("role") == "assistant" else "a pessoa"
                    linhas_conversa.append(f"- {quem}: {turno.get('content', '')[:150]}")
                conversa_recente = "\n".join(linhas_conversa)
            else:
                conversa_recente = "(nenhuma troca recente -- silêncio real)"
            prompt = PROMPT_PUXAR_ASSUNTO.format(fatos=resumo, conversa_recente=conversa_recente)
            from ..decision import completar_com_reserva, clientes_decisao
            principal, reserva = clientes_decisao(self.cfg.decisao)
            bruto = await asyncio.to_thread(
                completar_com_reserva, principal, reserva, prompt)
            m = re.search(r"\{.*\}", bruto, re.S)
            if not m:
                return
            texto = json.loads(m.group(0)).get("assunto")
            if texto:
                self.consciencia(guild_id).sugestao_assunto(texto)
                if self.cfg.debug:
                    print(f"[assunto] sugestão: {texto[:80]!r}")
        except Exception as e:
            if self.cfg.debug:
                print(f"[assunto] erro: {e}")

    def _topico_de_curiosidade(self, usuario: str, usados: set[str] | None = None) -> str | None:
        """Assunto pra puxar sozinha, tirado de memória episódica recente --
        retomar algo que a pessoa mencionou é mais crível que pesquisar
        tópico genérico do nada (mesma filosofia de 'fio', virando busca).
        """
        try:
            recentes = self.eva.memoria.listar(usuario=usuario, tipo="episodica", limite=8)
        except Exception:
            return None
        recentes = [m for m in recentes if m.conteudo not in (usados or set())]

        # Filtra memória vaga ANTES de virar busca. O extrator produz
        # resumo sem sujeito ("o usuário indica que está se preparando
        # para lidar com desafios") e prefixar "novidades sobre:" nisso dá
        # uma consulta sem alvo -- o SearXNG devolve texto motivacional
        # genérico, que o PortaoSubstancia depois barra. Barrar no fim
        # funciona, mas a busca já foi feita à toa; aqui a gente não gasta
        # a busca. Mesmo teste de concretude do portão, na origem.
        # Prefixo neutro na frente: `_tem_concreto` ignora a primeira
        # palavra de cada frase de propósito (maiúscula de início não é
        # nome próprio). Aqui isso barraria "Alex terminou de montar o
        # robô", que é justamente o tipo de memória que DÁ uma boa busca.
        # Empurrar a frase uma posição resolve sem afrouxar a regra.
        from ..consciousness import _tem_concreto
        concretas = [m for m in recentes
                     if _tem_concreto("sobre " + m.conteudo)]
        if not concretas:
            # Nenhuma memória serve pra pesquisar. Devolver None é a
            # resposta certa: melhor não ter tópico de curiosidade agora
            # do que pesquisar sobre nada.
            return None

        escolhida = random.choice(concretas)
        return f"novidades sobre: {escolhida.conteudo[:120]}"

    def _tts_tocando(self, guild_id: str) -> bool:
        return self.estado(guild_id).falando

    def _ocupada_em_alguma_guild(self) -> bool:
        """Tem algum turno em andamento em qualquer guild?

        Qualquer uma, não só a de origem: os modelos são compartilhados,
        então uma extração de memória disparada pela guild A atrapalharia
        a resposta em andamento da guild B do mesmo jeito. Num setup
        pessoal isso quase sempre é uma call só, mas a versão errada
        desta função só apareceria no dia em que não fosse.
        """
        return any(est.processando.locked() or est.falando
                   for est in self.guilds.values())

    def _cancelar_laco_consciencia(self, guild_id: str) -> None:
        gid = str(guild_id)
        tarefa = self._tarefas_consciencia.pop(gid, None)
        if tarefa and not tarefa.done():
            tarefa.cancel()

    def _ligar_visao_se_precisar(self, guild_id: str) -> None:
        """Liga a visão quando a PRIMEIRA call entra, não uma vez por
        call -- há uma tela só, então múltiplas guilds "ativas" (caso raro
        num setup pessoal) continuam compartilhando a mesma captura.
        """
        self._guilds_com_call.add(str(guild_id))
        # Corpo físico (robot_tools._ciclo_iniciativa) só age sozinho
        # quando tem gente numa call -- mesmo _guilds_com_call de sempre,
        # só que sinalizado pra outra thread via threading.Event.
        robot_tools.definir_em_call(bool(self._guilds_com_call))
        # A visão de TELA só liga se o robô não estiver conectado -- ver
        # _contexto_visual_para. Enquanto ela tem corpo, o monitor não
        # interessa, e o tick de tela é inferência paga à toa no mesmo
        # servidor que a visão do robô usa. _laco_visao também confere a
        # cada volta, porque o robô pode conectar depois da call começar.
        if (self.visao is not None and not self.visao.ativo
                and not robot_tools.robo_conectado()):
            self.visao.ligar()
            if self._tarefa_visao is None or self._tarefa_visao.done():
                self._tarefa_visao = asyncio.create_task(self._laco_visao())

        # Visão do robô -- liga o SISTEMA (fica pronto pra produzir cena
        # assim que houver quadro), sem forçar conexão com o robô aqui
        # (isso continua sob demanda, ver _ligar_robo_consciencia_se_precisar).
        if self.visao_robo is not None and not self.visao_robo.ativo:
            self.visao_robo.ligar()
            if self._tarefa_visao_robo is None or self._tarefa_visao_robo.done():
                self._tarefa_visao_robo = asyncio.create_task(self._laco_visao_robo())

        self._pre_aquecer_tts_se_precisar()

    def _pre_aquecer_tts_se_precisar(self) -> None:
        """Abre a conexão WebSocket da Cartesia ASSIM QUE a call começa,
        não na primeira fala real -- ver CartesiaTTSEngine.pre_aquecer
        pro achado real completo (timeout visto sempre na primeira fala
        de uma call recém-entrada, nunca no meio -- padrão de cold-start
        de handshake TLS+autenticação com um serviço externo).

        Só faz sentido pra motor com esse método (Cartesia hoje; Pocket
        TTS não tem WebSocket nem cold-start pra amortizar -- é local).
        getattr com default None cobre os dois casos sem `isinstance`.

        Dispara como task e não espera -- pré-aquecimento é otimização,
        não requisito; se demorar, a call já começou normal, e a
        primeira fala real só cai pro caminho mais lento (abrir do zero)
        se isto ainda não tiver terminado a tempo, exatamente como era
        antes desta mudança.
        """
        motor = getattr(self.tts, "motor", None) if self.tts else None
        pre_aquecer = getattr(motor, "pre_aquecer", None)
        if pre_aquecer is None:
            return
        asyncio.get_event_loop().run_in_executor(None, pre_aquecer)

    def _ligar_robo_consciencia_se_precisar(self, guild_id: str) -> None:
        robot_tools.definir_consciencia_callback(
            lambda texto: self._propagar_comentario_robo(texto)
        )
        # SEM warm-up automático aqui de propósito (removido -- ver
        # histórico se precisar do porquê da versão antiga). Conectar
        # sob demanda agora é: o botão "Conectar agora" no dashboard
        # (robot_tools.conectar_dashboard), ou qualquer ferramenta
        # robo_* real chamada em conversa (_chamar já sobe a thread na
        # hora, via _iniciar_thread_robo, e esperar _pronto funciona
        # pra QUALQUER chamador desde o fix da corrida -- não só quem
        # cria a thread). Entrar na call não é mais gatilho de conexão.

    def _propagar_comentario_robo(self, texto: str) -> None:
        for gid in list(self._guilds_com_call):
            self.consciencia(gid).evento_corporal(texto)

    def _desligar_robo_consciencia_se_precisar(self) -> None:
        if self._guilds_com_call:
            return
        robot_tools.definir_consciencia_callback(None)

    def _fechar_tts_se_precisar(self) -> None:
        """Fecha recursos persistentes do TTS quando a última call sai."""
        if self._guilds_com_call or not self.tts:
            return
        motor = getattr(self.tts, "motor", None)
        if motor and hasattr(motor, "fechar"):
            motor.fechar()

    def _desligar_visao_se_precisar(self, guild_id: str) -> None:
        """Só desliga quando a ÚLTIMA call sai -- não a cada guild que sai
        individualmente, senão duas calls simultâneas desligariam a visão
        uma da outra."""
        self._guilds_com_call.discard(str(guild_id))
        robot_tools.definir_em_call(bool(self._guilds_com_call))
        if self._guilds_com_call:
            return
        if self.visao is not None:
            self.visao.desligar()
            if self._tarefa_visao and not self._tarefa_visao.done():
                self._tarefa_visao.cancel()
        if self.visao_robo is not None:
            self.visao_robo.desligar()
            if self._tarefa_visao_robo and not self._tarefa_visao_robo.done():
                self._tarefa_visao_robo.cancel()

    async def _laco_visao(self) -> None:
        """Chama SistemaVisual.tick() periodicamente. tick() é síncrono e
        bloqueante (captura de tela +, ocasionalmente, uma chamada de
        rede de alguns segundos ao MiniCPM-V) -- roda via to_thread para
        não travar o resto do event loop (voz, texto, consciência) durante
        esses ~2s de análise.
        """
        while True:
            await asyncio.sleep(self.cfg.visao.tick_intervalo)
            # Robô conectado: pula o tick de tela por completo. Ela tem um
            # corpo olhando pra algum lugar; o monitor não é o que ela está
            # vendo, e cada tick custa ~3s de inferência no mesmo
            # gemma-3-4b que a visão do robô precisa. Checado a cada volta
            # (e não só na hora de ligar) porque o robô pode conectar
            # depois da call já ter começado.
            if robot_tools.robo_conectado():
                continue
            try:
                evento = await asyncio.to_thread(self.visao.tick)
                if evento:
                    print(f"[visao] evento: {evento}")
                    # Empurra em toda guild com call ativa -- caso comum é
                    # uma só; múltiplas é o caso raro de duas calls ao
                    # mesmo tempo, e não há como saber qual delas "é sobre"
                    # o que está na tela, então todas recebem o impulso.
                    for gid in list(self._guilds_com_call):
                        self.consciencia(gid).evento_visual(evento)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Visão é auxiliar -- uma falha aqui nunca deve derrubar
                # voz, texto ou consciência, que são o núcleo da conversa.
                print(f"[visao] erro no laço: {e}")

    async def _laco_visao_robo(self) -> None:
        """Mesmo padrão exato de _laco_visao, câmera do robô em vez da
        tela. tick() aqui pode falhar-silenciosamente MUITO mais vezes
        no início (robô ainda não conectou) -- isso é esperado, não é
        erro, ver SistemaVisualRobo.tick()/CapturaRobo.
        """
        while True:
            await asyncio.sleep(self.cfg.visao.tick_intervalo)
            try:
                evento = await asyncio.to_thread(self.visao_robo.tick)
                if evento:
                    print(f"[visao-robo] evento: {evento}")
                    for gid in list(self._guilds_com_call):
                        # evento_visual_robo, nao evento_visual: tipo
                        # proprio, com forca maior. O que a camera do robo
                        # mostra e a unica coisa que so ela esta vendo.
                        self.consciencia(gid).evento_visual_robo(evento)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[visao-robo] erro no laço: {e}")

    async def _laco_corpo(self) -> None:
        """Drena eventos corporais (transição de segurança, recusa de
        comando -- ver robot_tools.drenar_eventos_corpo) e empurra pra
        Consciencia de toda guild com call ativa, mesmo padrão de
        _laco_visao. Roda sempre, mesmo sem robô configurado -- drenar
        uma fila vazia é barato, e evita mais um "só liga se" espalhado.
        """
        while True:
            await asyncio.sleep(2.0)
            try:
                eventos = robot_tools.drenar_eventos_corpo()
            except Exception as e:
                print(f"[corpo] erro ao drenar eventos: {e}")
                continue
            for evento in eventos:
                print(f"[corpo] evento: {evento}")
                for gid in list(self._guilds_com_call):
                    self.consciencia(gid).evento_corporal(evento)

    async def _laco_jogo(self) -> None:
        """Drena eventos de Minecraft (fala de jogador no chat do jogo,
        dano, morte -- ver minecraft_tools.drenar_eventos_jogo) e empurra
        pra Consciencia de toda guild com call ativa. Mesmo padrão exato
        de _laco_corpo, inclusive rodar sempre: drenar fila vazia é
        barato, e evita mais um "só liga se" espalhado."""
        while True:
            await asyncio.sleep(2.0)
            try:
                eventos = minecraft_tools.drenar_eventos_jogo()
            except Exception as e:
                print(f"[jogo] erro ao drenar eventos: {e}")
                continue
            for descricao, forca in eventos:
                print(f"[jogo] evento: {descricao}")
                for gid in list(self._guilds_com_call):
                    self.consciencia(gid).evento_jogo(descricao, forca)

    async def _laco_consciencia(self, guild_id: str) -> None:
        """Bate no portão de tempos em tempos. Não decide nada -- só pergunta."""
        c = self.consciencia(guild_id)
        while True:
            await asyncio.sleep(self.cfg.consciencia.intervalo_tick)
            try:
                est = self.estado(guild_id)
                c.ocupada = est.processando.locked() or self._tts_tocando(guild_id)

                v = c.tick(self.eva.estado.estado)
                if self.cfg.debug and not v.passou:
                    print(f"[consciencia] {v}")

                if v.impulso is not None and v.impulso.tipo == "vazio":
                    # SÓ com a EVA livre. Estas duas disparam chamadas de
                    # LLM em segundo plano (pesquisa de curiosidade e
                    # sugestão de assunto), e no log de 26/08 elas rodaram
                    # CONCORRENTES com a resposta que a pessoa estava
                    # esperando -- o decisor caiu de 65-74 tok/s pra 10.70
                    # tok/s naquele momento exato, 6x mais lento por
                    # contenção de GPU. Uma sugestão de assunto que chega
                    # três segundos depois não perde nada; uma resposta que
                    # chega seis segundos depois perde a conversa.
                    #
                    # `c.ocupada` acabou de ser calculado acima, então usa
                    # ele em vez de reconsultar o estado.
                    if not c.ocupada:
                        self._tentar_curiosidade(guild_id, c)
                        self._tentar_puxar_assunto(guild_id, c)

                if not v.passou:
                    continue

                async with est.processando:
                    c.ocupada = True
                    try:
                        r = await self.eva.falar_sozinha_async(
                            v.impulso.conteudo, usuario=c.ultimo_falante, modo_voz=True)
                        if r.resposta:
                            print(f"[eva espontânea] {r.resposta}")
                            # `usuario` importa mesmo na fala espontânea:
                            # falar_sozinha() grava o turno em nome de
                            # c.ultimo_falante (ver orchestrator), então
                            # é esse o turno a corrigir se ela for
                            # cortada puxando assunto.
                            #
                            # ACHADO REAL (02/09/2026): antes chamava
                            # self.falar() incondicionalmente -- o
                            # caminho 100% bloqueante, que espera o áudio
                            # INTEIRO da resposta antes do primeiro som.
                            # Fala espontânea tende a ser mais longa que
                            # resposta direta (não tem o teto curto de
                            # MODO: VOZ pressionando tanto), então o
                            # custo do bloqueante aparece justamente
                            # onde di mais: log real mostrou 17-23s até
                            # o primeiro som em fala espontânea, mesmo
                            # depois do pré-aquecimento do WebSocket da
                            # Cartesia já ter resolvido isso pro caminho
                            # de resposta direta -- o problema aqui
                            # nunca foi o WebSocket, foi sempre ter
                            # pulado o streaming por completo.
                            if self.cfg.voz.voz_streaming:
                                await self.falar_texto_pronto_stream(
                                    guild_id, r.resposta, usuario=c.ultimo_falante)
                            else:
                                await self.falar(guild_id, r.resposta,
                                                 usuario=c.ultimo_falante)
                            c.ela_falou(espontanea=True)
                    finally:
                        c.ocupada = False
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[consciencia] erro: {e}")

    # ------------------------------------------------------------ envio

    async def _enviar(self, obj: dict) -> None:
        if self.ws:
            await self.ws.send(json.dumps(obj))

    async def entrar(self, guild_id: str, canal_id: str) -> None:
        await self._enviar({"type": "join", "guild_id": str(guild_id),
                            "channel_id": str(canal_id)})

    async def sair(self, guild_id: str) -> None:
        self._cancelar_laco_consciencia(guild_id)
        # Mesma redundância do laço de consciência: desliga já, na hora
        # que decidimos sair, sem esperar o "left" confirmado pelo bridge
        # voltar (que também dispara _desligar_visao_se_precisar, de novo
        # -- idempotente, seguro chamar duas vezes). Sem isso, um bridge
        # lento ou uma queda de rede antes da confirmação chegar deixaria
        # a captura de tela ligada indefinidamente depois de "sair" da call.
        self._desligar_visao_se_precisar(guild_id)
        self._desligar_robo_consciencia_se_precisar()
        self._fechar_tts_se_precisar()
        await self._enviar({"type": "leave", "guild_id": str(guild_id)})

    async def falar(self, guild_id: str, texto: str,
                    usuario: str | None = None) -> None:
        """Sintetiza e envia para o Discord.

        `usuario` é de quem é o turno no histórico -- só usado se a fala
        for interrompida no meio, pra saber qual turno corrigir. Fala
        espontânea (consciência) passa None e simplesmente não corrige
        nada, o que está certo: se ela é cortada puxando assunto sozinha,
        o assunto morre ali mesmo.
        """
        if not self.tts or not texto.strip():
            return

        est = self.estado(guild_id)
        try:
            fala = await asyncio.to_thread(self.tts.sintetizar, texto)

            # O backend Pocket já devolve PCM 48kHz estéreo alinhado --
            # passar de novo pelo ffmpeg seria reprocessar sem motivo, e
            # cada conversão extra é uma chance de introduzir artefato.
            if fala.formato == "pcm48":
                pcm = fala.audio
            else:
                pcm = await asyncio.to_thread(para_pcm_discord, fala.audio, fala.formato)
            pcm = alinhar_frames(pcm)
        except (ErroTTS, ErroAudio) as e:
            print(f"[tts] {e}")
            return

        # Estado da fala em andamento. Precisa estar completo ANTES do
        # primeiro byte sair: o áudio de quem fala por cima já pode estar
        # a caminho, e sem `texto_em_voz` preenchido o teste de eco não
        # tem com o que comparar -- o eco passaria como fala legítima e
        # ela se interromperia sozinha.
        est.texto_em_voz = texto.strip()
        est.usuario_em_voz = usuario
        est.bytes_em_voz = len(pcm)
        est.corte_pedido = False
        est.cortar.clear()

        est.falando = True
        print(f"[voz] falando {duracao_segundos(pcm):.1f}s")
        # O cabeçalho JSON avisa o bridge que o próximo quadro é áudio --
        # dentro do lock (ver __init__): sem isso, uma fala espontânea e
        # esta poderiam intercalar cabeçalho/payload uma da outra.
        async with self._envio_audio_lock:
            try:
                await self._enviar({"type": "play", "guild_id": str(guild_id)})
                await self.ws.send(pcm)
            except Exception:
                est.falando = False
                est.fim_da_fala = time.time()
                raise

        # "Tempo até o primeiro som": do fim da transcrição até o áudio
        # sair pelo WebSocket. Só faz sentido se este `falar` veio de uma
        # resposta a algo que a pessoa disse AGORA -- guarda contra
        # imprimir um número gigante e sem sentido quando `falar` é
        # chamado por fala espontânea (falar_sozinha), que não tem STT
        # nenhum acontecendo antes.
        if est.t_stt_fim and (time.time() - est.t_stt_fim) < 30:
            print(f"[tempo] até o primeiro som (bloqueante): "
                  f"{int((time.time() - est.t_stt_fim) * 1000)}ms")

    async def _falar_stream(self, guild_id: str, mensagem: str, usuario: str,
                            contexto_visual: str | None,
                            modo_multicanal: bool = False):
        """Chamado por `_responder_a_stream` quando EVA_VOZ_STREAMING=1
        (padrão -- ver config.py e o histórico na docstring de
        `_responder_a_stream`).

        Substitui responder_async()+falar() no turno de voz: o LLM gera
        frase a frase, cada frase é sintetizada e mandada pro bridge assim
        que fica pronta -- a EVA começa a falar antes de terminar de pensar
        a resposta inteira.

        Antes do PRIMEIRO play_start, acumula um pré-buffer local de
        `tts_pre_buffer_ms` (config.py) em vez de mandar cada pedaço assim
        que chega -- é a folga que evita a fala cortada: se a próxima
        frase demorar mais pra sintetizar do que a atual leva pra tocar,
        essa folga absorve a diferença antes que a fila do bridge.js
        esvazie e ele preencha com silêncio. Só o INÍCIO da resposta
        espera; depois do pré-buffer preenchido, cada pedaço vai direto.
        """
        if not self.tts:
            return None
        if not self.tts.motor._carregado:
            await self.tts.motor.inicializar()

        est = self.estado(guild_id)
        est.texto_em_voz = ""
        est.usuario_em_voz = usuario
        est.bytes_em_voz = 0
        est.corte_pedido = False
        est.cortar.clear()

        fila: queue.Queue = queue.Queue()
        asyncio.create_task(asyncio.to_thread(
            self._produzir_stream_de_voz, mensagem, usuario, contexto_visual,
            fila, modo_multicanal, est.cortar))

        return await self._consumir_fila_de_voz(guild_id, est, fila)

    async def falar_texto_pronto_stream(self, guild_id: str, texto: str,
                                        usuario: str | None = None) -> None:
        """Fala um texto JÁ PRONTO em streaming (play_start/play_chunk/
        play_end), frase por frase -- mesmo mecanismo de `_falar_stream`,
        mas sem o lado do LLM: aqui o texto inteiro já existe (fala
        espontânea, ver `_laco_consciencia`), só falta cortar em frases e
        sintetizar cada uma.

        ACHADO REAL (02/09/2026): fala espontânea (falar_sozinha) sempre
        chamava `self.falar()` -- o caminho 100% bloqueante, um request
        de síntese só, esperando o áudio INTEIRO antes do primeiro som --
        mesmo com EVA_VOZ_STREAMING=1 ligado. Isso não aparecia como erro
        nenhum (nem "[tts-stream] streaming falhou" nem qualquer outro
        log): o caminho bloqueante nunca foi tentado, e portanto nunca
        falhou -- ele só foi o único chamado. Confirmado comparando dois
        logs de call real: mesmo DEPOIS do fix de pré-aquecimento do
        WebSocket da Cartesia (que eliminou os timeouts do caminho de
        resposta direta), toda fala espontânea continuou levando
        17-23s até o primeiro som, sempre pela linha "[tempo] até o
        primeiro som (bloqueante)" -- o pré-aquecimento não ajuda aqui
        porque o problema nunca foi o WebSocket, foi o CAMINHO escolhido.

        Ainda síncrono internamente da mesma forma que `_falar_stream`:
        a produção (quebra em frases + síntese) roda numa thread via
        asyncio.to_thread, e o consumo da fila é compartilhado com
        `_falar_stream` via `_consumir_fila_de_voz` -- mesma lógica de
        pré-buffer, mesmo play_start/play_chunk/play_end, sem duplicar
        nada.
        """
        if not self.tts or not texto.strip():
            return
        if not self.tts.motor._carregado:
            await self.tts.motor.inicializar()

        est = self.estado(guild_id)
        est.texto_em_voz = ""
        est.usuario_em_voz = usuario
        est.bytes_em_voz = 0
        est.corte_pedido = False
        est.cortar.clear()

        fila: queue.Queue = queue.Queue()
        asyncio.create_task(asyncio.to_thread(
            self._produzir_stream_de_texto_pronto, texto, fila, est.cortar))

        await self._consumir_fila_de_voz(guild_id, est, fila)

    def _produzir_stream_de_texto_pronto(self, texto: str, fila: "queue.Queue",
                                         cortar: threading.Event | None = None) -> None:
        """Mesmo padrão de `_produzir_stream_de_voz`, sem o lado do LLM:
        o texto já existe inteiro, só corta em frases (mesmo extrator
        usado no streaming de verdade, pra tratar abreviação/decimal
        igual) e sintetiza cada uma via `_sintetizar_e_enfileirar`.
        """
        from ..voice.audio_utils import extrair_frases_fechadas

        buffer = texto
        try:
            frases, resto = extrair_frases_fechadas(buffer)
            for frase in frases:
                if cortar is not None and cortar.is_set():
                    break
                fila.put(("frase", frase))
                self._sintetizar_e_enfileirar(frase, fila)
            cortada = cortar is not None and cortar.is_set()
            if resto.strip() and not cortada:
                fila.put(("frase", resto))
                self._sintetizar_e_enfileirar(resto, fila)
        except Exception as e:
            fila.put(("erro", str(e)))
            return
        fila.put(("fim", None))

    async def _consumir_fila_de_voz(self, guild_id: str, est, fila: "queue.Queue"):
        """Consome a fila (frase/áudio/fim/erro) e manda play_start/
        play_chunk/play_end pro bridge -- extraído de `_falar_stream`
        pra ser compartilhado com `falar_texto_pronto_stream` (fala
        espontânea) sem duplicar a lógica de pré-buffer/corte/limpeza.
        """
        resultado = None
        iniciou = False
        pre_buffer = bytearray()
        # bytes/ms de PCM 48kHz estéreo 16 bits: 48000 * 2 canais * 2 bytes / 1000
        alvo_pre_buffer = int(192 * self.cfg.voz.tts_pre_buffer_ms)

        async def _iniciar_reproducao(dados: bytes) -> None:
            est.falando = True
            async with self._envio_audio_lock:
                await self._enviar({"type": "play_start", "guild_id": str(guild_id)})
                await self._enviar({"type": "play_chunk", "guild_id": str(guild_id)})
                await self.ws.send(dados)
            if est.t_stt_fim and (time.time() - est.t_stt_fim) < 30:
                print(f"[tempo] até o primeiro som (streaming): "
                      f"{int((time.time() - est.t_stt_fim) * 1000)}ms")

        while True:
            tipo, dado = await asyncio.to_thread(fila.get)
            if tipo == "frase":
                # Chega ANTES do áudio da frase: é o que mantém
                # `texto_em_voz` em dia pro teste de eco enquanto a
                # resposta ainda está sendo gerada.
                est.texto_em_voz = (est.texto_em_voz + " " + dado).strip()
                continue
            if tipo == "audio":
                if est.corte_pedido:
                    # Interrompida: o bridge já está encerrando no fim do
                    # que tem em fila. Mandar mais áudio agora só faria
                    # ela continuar falando depois do corte.
                    continue
                est.bytes_em_voz += len(dado)
                if not iniciou:
                    pre_buffer.extend(dado)
                    if len(pre_buffer) < alvo_pre_buffer:
                        continue
                    await _iniciar_reproducao(bytes(pre_buffer))
                    pre_buffer.clear()
                    iniciou = True
                    continue
                async with self._envio_audio_lock:
                    await self._enviar({"type": "play_chunk", "guild_id": str(guild_id)})
                    await self.ws.send(dado)
            elif tipo == "fim":
                resultado = dado
                break
            elif tipo == "erro":
                print(f"[voz-stream] {dado}")
                break

        if not iniciou and pre_buffer and not est.corte_pedido:
            # Resposta curta o bastante pra nunca bater o alvo de
            # pré-buffer -- manda o que tem em vez de nunca falar nada.
            await _iniciar_reproducao(bytes(pre_buffer))
            iniciou = True

        if iniciou:
            await self._enviar({"type": "play_end", "guild_id": str(guild_id)})
        else:
            # Nenhum som saiu: não vai vir `play_done` nenhum, então a
            # limpeza que normalmente acontece lá tem que acontecer aqui
            # -- senão `texto_em_voz` fica preso com um texto que nunca
            # foi falado, e o teste de eco passa a comparar contra ele.
            est.falando = False
            est.fim_da_fala = time.time()
            est.ultimo_texto_falado = est.texto_em_voz or est.ultimo_texto_falado
            est.texto_em_voz = ""
            est.usuario_em_voz = None
            est.corte_pedido = False
            est.cortar.clear()
        return resultado

    def _produzir_stream_de_voz(self, mensagem, usuario, contexto_visual, fila,
                                modo_multicanal: bool = False,
                                cortar: threading.Event | None = None) -> None:
        """Roda inteiro numa thread dedicada: monta contexto, consome o
        streaming do LLM, sintetiza frase a frase, põe áudio na fila. Nada
        bloqueante toca o event loop direto -- mesma política do resto do
        bridge_client.
        """
        try:
            stream = self.eva.responder(
                mensagem, usuario=usuario, stream=True, modo_voz=True,
                contexto_visual=contexto_visual, modo_multicanal=modo_multicanal)
        except Exception as e:
            fila.put(("erro", str(e)))
            return

        from ..voice.audio_utils import extrair_frases_fechadas

        buffer = ""
        try:
            for pedaco in stream:
                # Interrompida: para de sintetizar frase nova na hora. O
                # que já está na fila do bridge continua e termina a frase
                # atual -- é o "termina a frase e cede". Sair do laço aqui
                # também abandona o resto do streaming do LLM, que é o que
                # a gente quer: o texto seguinte não vai ser dito.
                if cortar is not None and cortar.is_set():
                    break
                buffer += pedaco
                frases, buffer = extrair_frases_fechadas(buffer)
                for frase in frases:
                    fila.put(("frase", frase))
                    self._sintetizar_e_enfileirar(frase, fila)
                    if cortar is not None and cortar.is_set():
                        break
        except Exception as e:
            fila.put(("erro", str(e)))
            return

        cortada = cortar is not None and cortar.is_set()
        if buffer.strip() and not cortada:
            fila.put(("frase", buffer))
            self._sintetizar_e_enfileirar(buffer, fila)

        fila.put(("fim", stream.resultado))

    def _sintetizar_e_enfileirar(self, frase: str, fila: "queue.Queue") -> None:
        """Sintetiza uma frase e põe os pedaços de áudio na fila, na
        ordem em que ficam prontos -- via generate_audio_stream, não
        espera o clipe inteiro da frase terminar.

        Se o streaming falhar ANTES de produzir qualquer pedaço, tenta
        uma vez no modo bloqueante (gerar_frase_sync) como rede de
        segurança -- cobre falha específica do caminho de streaming sem
        perder a frase inteira, mesmo espírito da retentativa do STT.
        Se já tiver produzido pedaço e falhar no meio, NÃO cai pro
        bloqueante: isso duplicaria o início da frase, que já foi
        enfileirado -- melhor perder o resto da frase do que repetir o
        começo dela.
        """
        produziu_algo = False
        try:
            for pedaco in self.tts.motor.gerar_frase_stream_sync(frase):
                fila.put(("audio", pedaco))
                produziu_algo = True
            return
        except Exception as e:
            if produziu_algo:
                print(f"[tts-stream] streaming falhou no meio de {frase[:40]!r}: "
                      f"{e} -- frase parcial, sem retentativa (evita duplicar áudio)")
                return
            print(f"[tts-stream] streaming falhou, tentando modo bloqueante: {e}")

        try:
            pcm = self.tts.motor.gerar_frase_sync(frase)
        except Exception as e:
            print(f"[tts-stream] modo bloqueante também falhou em {frase[:40]!r}: {e}")
            return
        if pcm:
            fila.put(("audio", pcm))

    # ---------------------------------------------------------- recepção

    async def _ao_receber_audio(self, guild_id: str, user_id: str, pcm: bytes,
                                durante_fala: bool = False) -> None:
        est = self.estado(guild_id)

        # `durante_fala` vem do bridge.js e diz se o microfone abriu
        # ENQUANTO ela falava. Não dá pra reconstruir isso aqui: a captura
        # só fecha depois de EVA_VOZ_SILENCIO de silêncio, então quando
        # este áudio chega ela já pode ter terminado. A janela de eco
        # continua valendo depois do fim da fala pelo mesmo motivo -- o
        # Discord ainda entrega pacotes atrasados por um tempinho.
        pode_ser_eco = (durante_fala or est.falando
                        or (time.time() - est.fim_da_fala) < self.janela_eco)

        # Descarta silêncio antes de gastar chamada de API
        if esta_silencioso(pcm):
            return

        # Já sabemos que é áudio real (não silêncio) -- avisa a
        # consciência AGORA, antes do STT (que leva 1-2s de `await` logo
        # abaixo). Fecha a janela de corrida onde a proatividade podia
        # falar por cima do início desta fala -- ver Consciencia.
        # audio_detectado() para o histórico completo do bug.
        self.consciencia(guild_id).audio_detectado()

        # Transcrição fica FORA do lock de propósito: se isso estivesse
        # dentro do `async with est.processando`, a fala da segunda pessoa
        # nunca conseguiria nem ser transcrita enquanto a da primeira ainda
        # estivesse em andamento -- inviabilizaria agregação de verdade,
        # porque a decisão "tem mais gente falando" depende de conseguir
        # processar as duas em paralelo.
        _t_audio_chegou = time.time()
        wav = pcm_para_wav(pcm)
        try:
            t = await self._transcrever(wav)
        except Exception as e:
            print(f"[stt] {e}")
            return
        print(f"[tempo] áudio recebido -> transcrição pronta: "
              f"{int((time.time() - _t_audio_chegou) * 1000)}ms")
        est.t_stt_fim = time.time()

        if parece_ruido(t):
            return

        # ---------------------------------------------------- eco vs fala
        referencia = est.texto_em_voz or est.ultimo_texto_falado
        if pode_ser_eco and referencia:
            if _parece_eco(t.texto, referencia):
                # Não é ninguém falando: é a voz dela voltando pelo
                # microfone de quem está sem fone. Avisa o bridge pra
                # nem decodificar esse usuário durante o playback daqui
                # pra frente -- é o que devolve a economia de CPU que o
                # fix v2 do bridge.js tinha, sem fechar o microfone de
                # quem está de fone.
                print(f"[eco] descartado de {user_id}: {t.texto[:60]!r}")
                await self._enviar({
                    "type": "fonte_de_eco", "guild_id": str(guild_id),
                    "user_id": str(user_id),
                    "ttl_ms": int(self.ttl_fonte_de_eco * 1000),
                })
                return

            if est.falando:
                # Fala de verdade por cima dela: interrompe.
                #
                # Exige mais de UMA palavra. Em call real, um clipe de
                # 0.9s virou "Yes." e cortou a fala dela no meio -- ruído
                # ou eco que o Whisper resolveu como palavra solta, em
                # inglês, então `_parece_eco` não tinha como pegar (o
                # texto dela é português). Interromper é uma ação cara e
                # visível; uma palavra solta não é evidência suficiente
                # de que alguém realmente quis falar.
                #
                # A fala NÃO é descartada por isso -- ela segue pro fluxo
                # normal logo abaixo e entra em `pendentes`. A pessoa é
                # respondida quando a EVA terminar; só não ganha o corte.
                if len(_normalizar_para_eco(t.texto)) > 1:
                    await self._interromper(
                        guild_id, motivo=f"{user_id} falou por cima")
                else:
                    print(f"[interrupção] ignorada, fala curta demais: "
                          f"{t.texto[:40]!r}")

        print(f"[call] {user_id}: {t.texto}")
        self.consciencia(guild_id).alguem_falou(
            str(user_id), t.texto, nome=self._nome(user_id))

        # ------------------------------------------- ela ainda está ocupada
        # Nada aqui descarta a fala. Ou ela vira turno agora, ou entra em
        # `pendentes` e vira turno assim que der -- ver `_drenar_pendentes`.
        if est.processando.locked() or est.falando:
            est.pendentes.append((str(user_id), t.texto, time.time()))
            # Continua contando como "esta pessoa falou agora" mesmo indo
            # pra fila -- senão, a fala seguinte de outra pessoa não veria
            # ninguém ativo e não abriria a janela multicanal.
            est.ultima_fala_por[str(user_id)] = time.time()
            if not est.falando and est.mensagem_em_curso is not None:
                # Ela está GERANDO e nada saiu como som ainda: a resposta
                # que está vindo já nasceu incompleta, porque a pessoa
                # emendou mais coisa. Invalida e junta as duas falas num
                # turno só, em vez de falar uma resposta obsoleta e depois
                # responder o resto separado.
                est.geracao_invalidada = True
                print(f"[turno] fala nova durante a geração -- "
                      f"as duas viram um turno só")
            else:
                print(f"[turno] fala guardada, ela ainda está ocupada "
                      f"({len(est.pendentes)} pendente(s))")
            self._agendar_drenagem(guild_id)
            return

        agora = time.time()
        outra_pessoa_ativa = any(
            uid != str(user_id) and (agora - ts) < self.janela_multiplas_pessoas
            for uid, ts in est.ultima_fala_por.items()
        )
        est.ultima_fala_por[str(user_id)] = agora

        if not outra_pessoa_ativa:
            # Caminho de sempre: uma pessoa só, responde na hora, sem
            # espera nenhuma acrescentada.
            await self._responder_a(guild_id, t.texto, str(user_id))
            return

        # Mais de uma pessoa ativa: agrega em vez de responder já. Cancela
        # a janela pendente (se tiver) e abre outra -- é isso que faz o
        # debounce esperar silêncio de TODO MUNDO, não um teto fixo desde
        # a primeira fala.
        print(f"[multicanal] {user_id} entrou na janela de agregação "
              f"({len(est.buffer_multicanal) + 1} fala(s) até agora)")
        est.buffer_multicanal.append((str(user_id), t.texto, agora))
        if est.tarefa_agregacao is not None and not est.tarefa_agregacao.done():
            est.tarefa_agregacao.cancel()
        est.tarefa_agregacao = asyncio.create_task(
            self._fechar_janela_multicanal(guild_id))

    # ------------------------------------------------------ interrupção

    async def _interromper(self, guild_id: str, motivo: str = "") -> None:
        """Corta a fala em andamento.

        Dois efeitos, nesta ordem: sinaliza pra thread de síntese parar de
        gerar frases novas (só o caminho de streaming tem isso; no
        bloqueante o áudio já foi todo gerado) e pede o corte ao bridge.
        O bridge decide COMO cortar -- fim da frase atual no streaming,
        fade no bloqueante. Ver `stopPlay` no bridge.js.
        """
        est = self.estado(guild_id)
        if est.corte_pedido:
            return
        est.corte_pedido = True
        est.cortar.set()
        print(f"[interrupção] {motivo or 'corte pedido'}")
        try:
            await self._enviar({"type": "stop_play", "guild_id": str(guild_id)})
        except Exception as e:
            print(f"[interrupção] falha ao pedir corte: {e}")

    def _corrigir_historico_apos_corte(self, guild_id: str, bytes_ouvidos: int,
                                       bytes_totais: int) -> None:
        """Troca, no histórico, a resposta inteira pelo trecho que virou som.

        Por que isso importa mais do que parece: `_pos_processar` grava o
        turno "assistant" quando o LLM termina de gerar, muito antes de
        virar áudio. Uma resposta cortada na metade fica registrada
        inteira, e na volta ela entra no prompt do turno seguinte como
        exemplo do próprio comportamento -- o mesmo mecanismo de few-shot
        acidental que já derrubou o tique de abertura repetida neste
        projeto. Só que aqui é pior: ela passa a "lembrar" de ter dito
        coisas que ninguém ouviu, e responde como se tivessem sido ditas.
        """
        est = self.estado(guild_id)
        texto = est.texto_em_voz
        usuario = est.usuario_em_voz
        if not texto or not usuario:
            return

        fracao = (bytes_ouvidos / bytes_totais) if bytes_totais else 1.0
        dito = _cortar_texto_na_fracao(texto, fracao)
        if dito == texto.strip():
            return

        try:
            if dito:
                self.eva.memoria.corrigir_ultimo_turno(
                    "assistant", dito, usuario=usuario)
                print(f"[interrupção] histórico corrigido para o que saiu no "
                      f"áudio ({int(fracao * 100)}%): {dito[-60:]!r}")
            else:
                self.eva.memoria.remover_ultimo_turno("assistant", usuario=usuario)
                print("[interrupção] cortada antes de sair som -- "
                      "turno removido do histórico")
        except Exception as e:
            print(f"[interrupção] não consegui corrigir o histórico: {e}")

    # ------------------------------------------------- turnos pendentes

    def _agendar_drenagem(self, guild_id: str) -> None:
        est = self.estado(guild_id)
        if est.tarefa_drenagem is not None and not est.tarefa_drenagem.done():
            return
        est.tarefa_drenagem = asyncio.create_task(self._drenar_pendentes(guild_id))

    async def _drenar_pendentes(self, guild_id: str) -> None:
        """Espera ela ficar livre e responde tudo que ficou esperando.

        Junta as falas acumuladas num turno só -- é o comportamento certo
        tanto pra pessoa que emendou uma frase na outra quanto pra duas
        pessoas falando ao mesmo tempo (nesse caso vira o mesmo bloco
        multicanal que `_fechar_janela_multicanal` já montava).

        Enquanto espera, `pendentes` continua recebendo -- por isso a
        leitura da lista é feita só depois do laço de espera, e a tarefa
        se re-agenda se algo tiver chegado no meio do caminho.
        """
        est = self.estado(guild_id)
        # Teto de espera. `est.falando` só volta a False quando o
        # `play_done` do bridge chega -- se ele se perder (queda de rede,
        # bridge reiniciado no meio de uma fala), sem este teto a fala da
        # pessoa ficaria presa em `pendentes` para sempre, que é
        # exatamente o sintoma que estamos consertando, só que mais raro
        # e mais difícil de enxergar. Preso é preso.
        limite = time.time() + 60.0
        try:
            while est.processando.locked() or est.falando:
                if time.time() > limite:
                    print("[turno] esperei 60s e ela continua ocupada -- "
                          "respondendo o pendente assim mesmo")
                    est.falando = False
                    break
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return

        itens, est.pendentes = est.pendentes, []
        if not itens:
            return

        vozes = {uid for uid, _, _ in itens}
        ultimo_usuario = itens[-1][0]

        if len(itens) == 1:
            bloco = itens[0][1]
            multicanal = False
        elif len(vozes) == 1:
            # Mesma pessoa emendando: vira uma fala só, sem marcação
            # nenhuma de canal -- pro modelo isso é uma frase comprida,
            # não uma conversa entre gente diferente.
            bloco = " ".join(texto for _, texto, _ in itens)
            multicanal = False
        else:
            bloco = "\n".join(
                f"[voz] {self._nome(uid) or uid}: {texto}" for uid, texto, _ in itens)
            multicanal = True

        print(f"[turno] respondendo {len(itens)} fala(s) que estavam esperando")
        await self._responder_a(guild_id, bloco, ultimo_usuario,
                                modo_multicanal=multicanal)

        if est.pendentes:
            self._agendar_drenagem(guild_id)

    async def _responder_a(self, guild_id: str, mensagem: str, usuario: str,
                           modo_multicanal: bool = False) -> None:
        """Gera e fala a resposta de um turno -- caminho compartilhado
        pelo modo de uma pessoa só e pelo modo multicanal agregado.

        Escolhe entre as duas estratégias abaixo via `EVA_VOZ_STREAMING`
        (config.py) -- ver docstrings de cada uma para o histórico.
        """
        if self.cfg.voz.voz_streaming:
            await self._responder_a_stream(guild_id, mensagem, usuario, modo_multicanal)
        else:
            await self._responder_a_bloqueante(guild_id, mensagem, usuario, modo_multicanal)

    async def _responder_a_stream(self, guild_id: str, mensagem: str, usuario: str,
                                  modo_multicanal: bool = False) -> None:
        """Caminho em streaming: fala já na primeira frase pronta, via
        `_falar_stream` (play_start/play_chunk/play_end).

        DESLIGADO POR PADRÃO (EVA_VOZ_STREAMING=0) -- SEGUNDA tentativa,
        SEGUNDA reversão. Histórico:

        1ª tentativa: áudio saiu fragmentado. Suspeita: `portuguese_24l`
        (TTS antigo, 24 camadas sem destilação) rodando com RTF perto do
        limite de tempo real em CPU -- `gerar_frase_stream_sync` não
        produzia áudio mais rápido do que o bridge.js consumia, a fila
        esvaziava, e o preenchimento de silêncio soava fragmentado.

        2ª tentativa (depois de trocar pro modelo destilado + forçar
        POCKET_TTS_DEVICE=cpu -- havia GPU AMD entrando por auto-detecção
        via ROCm, kernel de atenção que o próprio PyTorch marca como
        experimental): áudio CONTINUOU ruim. Ou seja, a causa não era só
        velocidade de síntese. Suspeita agora é o mecanismo de PLAYBACK:
        bridge.js monta o áudio com um timer manual de 20ms (setInterval)
        puxando de um buffer preenchido conforme os pedaços chegam --
        muito mais frágil que o caminho bloqueante, que entrega o clipe
        inteiro pronto pro @discordjs/voice de uma vez, sem nenhum timer
        nosso competindo com o event loop (que também está fazendo I/O de
        rede) pra bater 20ms certinho.

        Não removido do código -- só não roda por padrão. Se um dia valer
        revisitar (ex: mudar a estratégia de playback em si, não só a
        síntese), o caminho já está aqui, testado duas vezes, com o motivo
        de cada reversão documentado -- não precisa reconstruir do zero
        nem repetir os mesmos dois experimentos.
        """
        est = self.estado(guild_id)
        async with est.processando:
            est.mensagem_em_curso = mensagem
            est.geracao_invalidada = False
            try:
                contexto_visual = await self._contexto_visual_para(mensagem)
                r = await self._falar_stream(
                    guild_id, mensagem, usuario, contexto_visual,
                    modo_multicanal=modo_multicanal)
            finally:
                em_curso = est.mensagem_em_curso
                est.mensagem_em_curso = None

            if r is None or r.erro:
                if r and r.erro:
                    print(f"[eva] {r.erro}")
                return
            if not r.resposta:
                return

            # Só descarta se NADA virou som. Aqui isso é raro de verdade
            # (o primeiro áudio sai em ~120ms depois da primeira frase
            # pronta); se já saiu som, o caminho certo é a interrupção,
            # não o descarte -- a pessoa ouviu parte da resposta.
            if est.geracao_invalidada and est.bytes_em_voz == 0:
                est.geracao_invalidada = False
                try:
                    self.eva.memoria.remover_ultimo_turno("assistant", usuario=usuario)
                    self.eva.memoria.remover_ultimo_turno("user", usuario=usuario)
                except Exception as e:
                    print(f"[turno] não consegui limpar o turno descartado: {e}")
                if em_curso:
                    est.pendentes.insert(0, (usuario, em_curso, time.time()))
                print("[turno] resposta descartada -- vai ser refeita com as "
                      "duas falas juntas")
                self._agendar_drenagem(guild_id)
                return
            est.geracao_invalidada = False

            print(f"[eva] {r.resposta}")
            self.consciencia(guild_id).ela_falou()

            # Lacuna de conhecimento: a mensagem tocou em algo que pode
            # estar desatualizado no que a EVA sabe. Não bloqueia nada --
            # dispara como task solta; se terminar, vira impulso de
            # iniciativa; se não terminar a tempo ou não achar nada, não
            # acontece nada (sem aviso, sem erro visível).
            if r.plano.possivel_lacuna:
                asyncio.create_task(
                    self._pesquisar_e_registrar(guild_id, r.plano.possivel_lacuna))

    async def _responder_a_bloqueante(self, guild_id: str, mensagem: str, usuario: str,
                                      modo_multicanal: bool = False) -> None:
        """Caminho original, preservado como fallback via EVA_VOZ_STREAMING=0.

        Espera a resposta inteira do LLM, depois sintetiza tudo de uma vez
        (`falar()`/`gerar_para_discord` -- split por sentença, normalização
        única, mensagem "play" só, sem fragmentar), só então toca. Mais
        devagar, mas é o caminho que `falar_sozinha` (fala espontânea)
        sempre usou e nunca teve problema de fragmentação.
        """
        est = self.estado(guild_id)
        async with est.processando:
            est.mensagem_em_curso = mensagem
            est.geracao_invalidada = False
            try:
                contexto_visual = await self._contexto_visual_para(mensagem)
                r = await asyncio.to_thread(
                    self.eva.responder, mensagem, usuario=usuario, modo_voz=True,
                    contexto_visual=contexto_visual, modo_multicanal=modo_multicanal,
                )
            finally:
                em_curso = est.mensagem_em_curso
                est.mensagem_em_curso = None

            if r is None or r.erro:
                if r and r.erro:
                    print(f"[eva] {r.erro}")
                return
            if not r.resposta:
                return

            if est.geracao_invalidada:
                # A pessoa emendou outra fala enquanto isto era gerado.
                # Esta resposta não vale mais: ela responde só metade do
                # que foi dito, e falar metade agora pra responder o resto
                # depois é justamente o ping-pong que a gente não quer.
                #
                # CUSTO ASSUMIDO: esta geração inteira vai pro lixo, e o
                # llama-server já gastou o tempo dela. Não dá pra abortar
                # -- llm.py usa urllib bloqueante, e mesmo cancelando a
                # thread o servidor continuaria gerando e a próxima
                # requisição ficaria na fila atrás dela. Cancelar não
                # economizaria nada; só esconderia o custo.
                #
                # Essa janela encolhe muito com EVA_VOZ_STREAMING=1: lá o
                # som começa na primeira frase pronta, então "gerando sem
                # nada tocando" dura menos de um segundo, e o caso normal
                # passa a ser interrupção (que não desperdiça nada) em vez
                # de invalidação.
                est.geracao_invalidada = False
                try:
                    self.eva.memoria.remover_ultimo_turno("assistant", usuario=usuario)
                    self.eva.memoria.remover_ultimo_turno("user", usuario=usuario)
                except Exception as e:
                    print(f"[turno] não consegui limpar o turno descartado: {e}")
                if em_curso:
                    est.pendentes.insert(0, (usuario, em_curso, time.time()))
                print("[turno] resposta descartada -- vai ser refeita com as "
                      "duas falas juntas")
                self._agendar_drenagem(guild_id)
                return

            print(f"[eva] {r.resposta}")
            await self.falar(guild_id, r.resposta, usuario=usuario)
            self.consciencia(guild_id).ela_falou()

            if r.plano.possivel_lacuna:
                asyncio.create_task(
                    self._pesquisar_e_registrar(guild_id, r.plano.possivel_lacuna))

    async def _fechar_janela_multicanal(self, guild_id: str) -> None:
        """Espera o debounce e, se ninguém mais falou nesse tempo, monta o
        bloco multicanal com tudo que chegou e dispara UMA resposta.

        Se uma fala nova chegar antes do sleep terminar, esta tarefa é
        CANCELADA por `_ao_receber_audio` antes de uma nova ser criada --
        então chegar até o fim do sleep aqui já significa "ninguém mais
        falou por `janela_debounce_multicanal` segundos", sem precisar
        checar isso de novo.
        """
        try:
            await asyncio.sleep(self.janela_debounce_multicanal)
        except asyncio.CancelledError:
            return

        est = self.estado(guild_id)
        itens = est.buffer_multicanal
        est.buffer_multicanal = []
        est.tarefa_agregacao = None
        if not itens:
            return

        linhas = [f"[voz] {self._nome(uid) or uid}: {texto}" for uid, texto, _ in itens]
        bloco = "\n".join(linhas)
        print(f"[multicanal] janela fechada, {len(itens)} fala(s) agregada(s):\n{bloco}")
        # `usuario=` fica sendo quem falou por último -- memória e
        # identidade são por pessoa, ainda não existe "memória do grupo".
        # Simplificação assumida, não esquecida (ver docstring de
        # EVA.responder em orchestrator.py).
        ultimo_usuario = itens[-1][0]
        await self._responder_a(guild_id, bloco, ultimo_usuario, modo_multicanal=True)

    # ------------------------------------------------------------ texto

    async def enviar_mensagem(self, canal_id: str, texto: str,
                              responder_a: str | None = None) -> None:
        await self._enviar({
            "type": "send_message",
            "channel_id": str(canal_id),
            "content": texto,
            "reply_to": str(responder_a) if responder_a else None,
        })

    async def _ao_receber_mensagem(self, d: dict) -> None:
        """Decide se responde e processa uma mensagem de texto.

        A regra de quando responder vive aqui, e não no bridge: assim há um
        lugar só para mudar, em vez de lógica espalhada entre os dois
        processos.
        """
        conteudo = (d.get("content") or "").strip()
        canal = str(d["channel_id"])
        anexos = d.get("attachments") or []

        # comandos
        prefixo = self.cfg.discord.prefixo.strip()
        if conteudo.lower().startswith(prefixo.lower()):
            await self._comando(d, conteudo[len(prefixo):].strip())
            return

        # Em servidor só responde quando mencionada ou em canal dedicado --
        # senão responderia toda conversa entre humanos.
        dedicado = self.cfg.discord.canal_dedicado
        deve = (d.get("is_dm") or d.get("mentioned")
                or (dedicado and canal == str(dedicado)) or anexos)
        if not deve:
            return

        # áudio anexado: transcreve antes
        if anexos:
            texto = await self._transcrever_anexo(anexos[0], canal)
            if not texto:
                return
        else:
            texto = conteudo

        if not texto:
            return

        gid = str(d.get("guild_id") or f"dm:{canal}")
        est = self.estado(gid)

        # Não descarta mensagem por estar ocupada. Era o mesmo `return`
        # seco do caminho de voz, com um agravante: no texto a pessoa VÊ a
        # mensagem dela entregue no canal e a EVA simplesmente não
        # responder. `async with` abaixo já enfileira naturalmente -- ela
        # responde quando terminar o que está fazendo.
        async with est.processando:
            await self._enviar({"type": "typing", "channel_id": canal})
            autor = str(d.get("author_id") or canal)
            self.consciencia(gid).registrar_nome(autor, d.get("author_name"))
            r = await self.eva.responder_async(
                texto, usuario=autor,
                contexto_visual=await self._contexto_visual_para(texto))

            if r.erro:
                await self.enviar_mensagem(
                    canal, f"[modelo indisponível] {r.erro[:300]}", d.get("message_id"))
                return
            if not r.resposta:
                return

            await self.enviar_mensagem(canal, r.resposta, d.get("message_id"))

            # se estiver numa call desse servidor, fala também
            if est.canal_id:
                await self.falar(gid, r.resposta, usuario=autor)

            if r.plano.possivel_lacuna:
                asyncio.create_task(
                    self._pesquisar_e_registrar(gid, r.plano.possivel_lacuna))

    async def _transcrever_anexo(self, anexo: dict, canal: str) -> str | None:
        if not self.stt.disponivel():
            await self.enviar_mensagem(
                canal, "Não consigo ouvir áudio — falta a chave da Groq.")
            return None
        try:
            import urllib.request
            dados = await asyncio.to_thread(
                lambda: urllib.request.urlopen(anexo["url"], timeout=30).read()
            )
            t = await self._transcrever(dados, anexo.get("name", "audio.ogg"))
        except Exception as e:
            await self.enviar_mensagem(canal, f"Não consegui transcrever: {str(e)[:200]}")
            return None

        if parece_ruido(t):
            await self.enviar_mensagem(canal, "Não entendi nada no áudio.")
            return None
        return t.texto

    async def _comando(self, d: dict, cmd: str) -> None:
        canal = str(d["channel_id"])
        gid = str(d.get("guild_id") or "")
        partes = cmd.split(maxsplit=1)
        nome = (partes[0] if partes else "").lower()
        arg = partes[1].strip() if len(partes) > 1 else ""

        if nome in ("entra", "join"):
            canal_voz = arg or d.get("voice_channel_id")
            if not canal_voz:
                await self.enviar_mensagem(
                    canal, f"Uso: `{self.cfg.discord.prefixo}entra <id do canal de voz>`\n"
                           "(clique com o botão direito no canal > Copiar ID, "
                           "com o Modo Desenvolvedor ligado)")
                return
            await self.entrar(gid, canal_voz)

        elif nome in ("sai", "leave"):
            await self.sair(gid)
            await self.enviar_mensagem(canal, "Saí da call.")

        elif nome in ("status", "diag"):
            dd = self.diagnostico()
            linhas = [
                f"**modelo**: {'conectado' if dd['llm_disponivel'] else 'não conectado'}",
                f"**STT**: {dd['stt']}",
                f"**TTS**: {dd['tts']}",
                f"**memórias**: {sum(dd['memorias'].values())}",
                f"**interações**: {dd['interacoes']}",
                f"**em call**: {', '.join(dd['guilds_ativas']) or 'não'}",
            ]
            await self.enviar_mensagem(canal, "\n".join(linhas))

        elif nome in ("memoria", "memória"):
            linhas = []
            for tipo in ("semantica", "episodica", "procedural", "personalidade"):
                itens = self.eva.memoria.listar(tipo, limite=10)
                if itens:
                    linhas.append(f"**{tipo}**")
                    linhas += [f"  - {m.conteudo}" for m in itens]
            await self.enviar_mensagem(
                canal, "\n".join(linhas) if linhas else "Nada guardado ainda.")

        elif nome == "esquecer":
            if not arg:
                await self.enviar_mensagem(
                    canal, f"Uso: `{self.cfg.discord.prefixo}esquecer <termo>`")
            else:
                n = await asyncio.to_thread(self.eva.esquecer, arg)
                await self.enviar_mensagem(canal, f"{n} memória(s) removida(s).")

        elif nome in ("ajuda", "help"):
            p = self.cfg.discord.prefixo
            await self.enviar_mensagem(canal,
                f"`{p}entra <id do canal>` — entro na call\n"
                f"`{p}sai` — saio da call\n"
                f"`{p}status` — diagnóstico\n"
                f"`{p}memoria` — o que eu sei sobre você\n"
                f"`{p}esquecer <termo>` — apago memórias\n\n"
                "Fora dos comandos: me mencione, mande DM, ou anexe um áudio.")
        else:
            await self.enviar_mensagem(
                canal, f"Não conheço `{nome}`. Use `{self.cfg.discord.prefixo}ajuda`.")

    # ------------------------------------------------------------- laço

    async def _tratar_mensagem(self, msg) -> None:
        # Quadro binário: é o áudio anunciado pelo cabeçalho anterior
        if isinstance(msg, (bytes, bytearray)):
            if not self._audio_pendente:
                return
            cab = self._audio_pendente
            self._audio_pendente = None
            await self._ao_receber_audio(
                cab["guild_id"], cab["user_id"], bytes(msg),
                durante_fala=bool(cab.get("durante_fala")))
            return

        try:
            d = json.loads(msg)
        except json.JSONDecodeError:
            return

        tipo = d.get("type")

        if tipo == "audio":
            self._audio_pendente = d
        elif tipo == "message":
            await self._ao_receber_mensagem(d)
        elif tipo == "message_sent":
            pass
        elif tipo == "ready":
            print(f"[bridge] conectado (versão {d.get('version')})")
        elif tipo == "joined":
            gid = str(d["guild_id"])
            self.estado(gid).canal_id = str(d.get("channel_id"))
            # Sessao nova a cada call. Sem isso, o historico da call
            # anterior entra no prompt como papel "assistant" e a EVA copia
            # o proprio comportamento passado -- inclusive tique de modelo
            # ja trocado (ver SESSAO DE CONVERSA em memory/store.py).
            self.eva.memoria.nova_sessao()
            self._cancelar_laco_consciencia(gid)
            self._tarefas_consciencia[gid] = asyncio.create_task(
                self._laco_consciencia(gid))
            self._ligar_visao_se_precisar(gid)
            self._ligar_robo_consciencia_se_precisar(gid)
            print(f"[bridge] entrou em '{d.get('channel_name')}'")
        elif tipo == "left":
            gid = str(d["guild_id"])
            self.estado(gid).canal_id = None
            self._cancelar_laco_consciencia(gid)
            self._desligar_visao_se_precisar(gid)
            self._desligar_robo_consciencia_se_precisar()
            self._fechar_tts_se_precisar()
            motivo = f" ({d['reason']})" if d.get("reason") else ""
            print(f"[bridge] saiu do canal{motivo}")
        elif tipo == "reconnecting":
            print("[bridge] conexão de voz caiu, reconectando...")
        elif tipo == "reconnected":
            print("[bridge] reconectado")
        elif tipo == "play_done":
            gid = str(d["guild_id"])
            est = self.estado(gid)
            est.falando = False
            est.fim_da_fala = time.time()
            if d.get("cortada"):
                self._corrigir_historico_apos_corte(
                    gid,
                    int(d.get("bytes_reproduzidos") or 0),
                    int(d.get("bytes_totais") or est.bytes_em_voz or 0),
                )
            # Limpa DEPOIS da correção: `texto_em_voz` é a entrada dela.
            # A partir daqui não há fala em andamento, então qualquer
            # áudio que chegar atrasado não tem mais com o que ser
            # comparado -- e a janela de eco (`fim_da_fala`) é quem cobre
            # esse rabicho.
            est.ultimo_texto_falado = est.texto_em_voz or est.ultimo_texto_falado
            est.texto_em_voz = ""
            est.usuario_em_voz = None
            est.bytes_em_voz = 0
            est.corte_pedido = False
            est.cortar.clear()
        elif tipo == "error":
            print(f"[bridge] erro: {d.get('message')}")

    async def rodar(self, reconectar: bool = True) -> None:
        """Conecta ao bridge e processa mensagens até ser interrompido."""
        if websockets is None:
            raise SystemExit("websockets não instalado. Rode: pip install websockets")

        # Fora do laço de reconexão -- o dashboard não deveria reiniciar
        # (e perder o histórico de uptime/estado) só porque a conexão com
        # o bridge.js caiu e reconectou.
        if self.cfg.dashboard.ativa:
            from dashboard import ServidorDashboard
            self._dashboard = ServidorDashboard(self)
            self._dashboard.iniciar()

        self._tarefa_corpo = asyncio.create_task(self._laco_corpo())
        self._tarefa_jogo = asyncio.create_task(self._laco_jogo())

        tentativa = 0
        while True:
            try:
                async with websockets.connect(self.url, max_size=None) as ws:
                    self.ws = ws
                    tentativa = 0
                    print(f"[bridge] conectado em {self.url}")
                    async for msg in ws:
                        try:
                            await self._tratar_mensagem(msg)
                        except Exception as e:
                            # Uma mensagem problemática não pode derrubar o
                            # laço inteiro -- a conversa continua.
                            print(f"[erro ao tratar mensagem] {type(e).__name__}: {e}")
            except (OSError, ConnectionError) as e:
                if not reconectar:
                    raise
                tentativa += 1
                espera = min(30, 2 * tentativa)
                print(f"[bridge] sem conexão ({e}). Nova tentativa em {espera}s...")
                print(f"          o bridge está rodando? node bridge.js")
                await asyncio.sleep(espera)
            except Exception as e:
                if not reconectar:
                    raise
                print(f"[bridge] conexão perdida: {type(e).__name__}: {e}")
                await asyncio.sleep(3)
            finally:
                self.ws = None

    def trocar_tts(self, backend: str) -> dict:
        """Troca o backend de TTS EM RUNTIME, sem reiniciar o processo.

        Ao contrário de `visao.ativa`/`usar_embeddings` (que só são lidos
        UMA VEZ na inicialização -- ver docstring de dashboard.py), dar
        suporte a troca ao vivo aqui é simples: `self.tts` é só uma
        referência que `falar`/`_falar_stream` leem a cada chamada, então
        trocar essa referência JÁ é a mudança -- não precisa recriar
        ClienteBridge nem nada em volta.

        Levanta ErroTTS com a mensagem exata de `criar_tts` se o backend
        pedido falhar (chave faltando, pacote não instalado, etc.) --
        quem chama (dashboard) mostra isso direto pra você, em vez de você
        precisar ir atrás de log de console. `self.tts` só é sobrescrito
        em caso de SUCESSO -- uma troca que falha não derruba o backend
        que já estava funcionando.
        """
        novo = criar_tts(
            backend=backend,
            idioma=self.cfg.voz.tts_idioma,
            quantizar=self.cfg.voz.tts_quantize,
        )
        self.tts = novo
        self.erro_tts = None
        self.cfg.voz.tts_backend = backend
        print(f"[tts] backend trocado em runtime para: {novo.nome}")
        return {"backend": novo.nome}

    def diagnostico(self) -> dict:
        d = self.eva.diagnostico()
        d.update({
            "bridge_url": self.url,
            "stt": (f"{self.cfg.voz.stt_backend}: ok" if self.stt.disponivel()
                    else f"{self.cfg.voz.stt_backend}: indisponível"),
            "tts": self.tts.nome if self.tts else f"indisponível ({self.erro_tts})",
            "guilds_ativas": [g for g, e in self.guilds.items() if e.canal_id],
        })
        return d

    def fechar(self) -> None:
        for gid in list(self._tarefas_consciencia):
            self._cancelar_laco_consciencia(gid)
        if self._tarefa_visao and not self._tarefa_visao.done():
            self._tarefa_visao.cancel()
        if self._tarefa_corpo and not self._tarefa_corpo.done():
            self._tarefa_corpo.cancel()
        if self._tarefa_jogo and not self._tarefa_jogo.done():
            self._tarefa_jogo.cancel()
        if self.tts:
            motor = getattr(self.tts, "motor", None)
            if motor and hasattr(motor, "fechar"):
                motor.fechar()
        if self.visao:
            self.visao.fechar()
        if self._dashboard:
            self._dashboard.parar()
        self.eva.fechar()

    def desligar_tudo(self) -> None:
        """Encerramento explícito e IMEDIATO, via botão do dashboard.

        Por que isso existe além de fechar(): o ThreadPoolExecutor por
        trás de asyncio.to_thread() registra um atexit que espera
        QUALQUER chamada em andamento terminar antes do processo morrer
        de verdade -- se isso acontece no meio de uma síntese de TTS,
        chamada ao LM Studio, ou análise de visão, o terminal fica preso
        até aquela chamada terminar sozinha. fechar() limpa o que dá pra
        limpar de forma síncrona (tarefas de consciência, visão,
        dashboard, banco); os._exit() depois disso pula esse atexit
        inteiro -- bruto de propósito, é a saída rápida que devia
        existir.
        """
        try:
            self.fechar()
        except Exception as e:
            print(f"[desligar] erro durante limpeza (ignorado): {e}")
        import os
        os._exit(0)