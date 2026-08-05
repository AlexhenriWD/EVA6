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

Node -> Python:
    {"type":"ready", "version":"2.0"}
    {"type":"joined", "guild_id":..., "channel_id":..., "channel_name":...}
    {"type":"left", "guild_id":..., "reason":...}
    {"type":"reconnecting", "guild_id":...}
    {"type":"reconnected", "guild_id":..., "channel_id":...}
    {"type":"error", "message":...}
    {"type":"audio", "guild_id":..., "user_id":..., "bytes":N}
        seguido de um quadro BINÁRIO com N bytes de PCM
    {"type":"play_done", "guild_id":...}

Áudio nos dois sentidos: PCM 48kHz, estéreo, s16le.

DECISÕES DE DESIGN
------------------
O ciclo da EVA (STT -> memória -> LLM -> TTS) leva segundos e é síncrono.
Rodá-lo direto no laço de eventos travaria o WebSocket, e o bridge ficaria
sem resposta -- inclusive para o keepalive. Por isso tudo que bloqueia vai
para thread via asyncio.to_thread.

Enquanto a EVA fala, o áudio recebido é ignorado. O bridge já evita
decodificar nesse período (o comentário sobre CPU no bridge.js), mas a
guarda aqui cobre o intervalo entre o fim da reprodução e o Discord parar
de mandar pacotes -- sem isso, ela ouviria o próprio eco de quem estiver
sem fone.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

from ..config import EVAConfig, carregar_config
from ..orchestrator import EVA
from ..voice.audio import (
    ErroAudio,
    alinhar_frames,
    duracao_segundos,
    esta_silencioso,
    para_pcm_discord,
    pcm_para_wav,
)
from ..voice.stt import GroqSTT, parece_ruido
from ..voice.tts import ErroTTS, criar_tts


@dataclass
class EstadoGuild:
    canal_id: str | None = None
    falando: bool = False
    # Instante em que a fala terminou. Usado para ignorar o eco por um
    # tempinho depois -- o Discord continua entregando pacotes atrasados.
    fim_da_fala: float = 0.0
    processando: asyncio.Lock = field(default_factory=asyncio.Lock)


class ClienteBridge:
    def __init__(self, config: EVAConfig | None = None, url: str = "ws://localhost:8765"):
        self.cfg = config or carregar_config()
        self.url = url
        self.eva = EVA(self.cfg)

        self.stt = GroqSTT(
            api_key=self.cfg.voz.stt_chave,
            modelo=self.cfg.voz.stt_modelo,
            idioma=self.cfg.voz.stt_idioma,
        )
        self.tts = None
        self.erro_tts: str | None = None
        try:
            self.tts = criar_tts(
                backend=self.cfg.voz.tts_backend or None,
                idioma=self.cfg.voz.tts_idioma,
            )
        except ErroTTS as e:
            self.erro_tts = str(e)

        self.ws = None
        self.guilds: dict[str, EstadoGuild] = {}
        from ..consciousness import Consciencia
        self.consciencias: dict[str, Consciencia] = {}
        self._tarefas_consciencia: dict[str, asyncio.Task] = {}

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
        self._dashboard = None  # criado em rodar(), só se cfg.dashboard.ativa
        self._guilds_com_call: set[str] = set()

        # O bridge manda o cabeçalho JSON e logo depois o quadro binário.
        # Guardamos o cabeçalho para saber de quem é o áudio que vem a seguir.
        self._audio_pendente: dict | None = None
        # Silêncio a ignorar após a EVA falar, em segundos
        self.janela_eco = 0.6

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

    def _contexto_visual_para(self, texto: str) -> str | None:
        """A cena para injetar neste turno, ou None -- decide ANTES de
        chamar EVA.responder(), porque contexto_visual é parâmetro de
        entrada, não algo que o orquestrador busca sozinho (EVA não
        conhece SistemaVisual de propósito, mesma separação que já existe
        entre EVA e Consciencia).

        Dois caminhos para "sim, injeta": a pessoa referenciou a tela
        explicitamente (visao_relevante, decision.py -- "olha isso",
        "minha tela"), OU a cena mudou muito recentemente (janela_
        relevancia_recente) e ainda é razoável supor que seja sobre isso
        mesmo sem menção direta. Fora esses dois casos, NÃO injeta --
        contexto visual perene em toda resposta é o mesmo risco de
        "narradora" da consciência, só que no nível da conversa inteira.
        """
        if self.visao is None:
            return None
        from ..decision import visao_relevante

        if visao_relevante(texto):
            return self.visao.contexto_atual()

        cena = self.visao.cena
        if cena and cena.idade_segundos() < self.cfg.visao.janela_relevancia_recente:
            return self.visao.contexto_atual()

        return None

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

    def _tts_tocando(self, guild_id: str) -> bool:
        return self.estado(guild_id).falando

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
        if self.visao is None or self.visao.ativo:
            return
        self.visao.ligar()
        if self._tarefa_visao is None or self._tarefa_visao.done():
            self._tarefa_visao = asyncio.create_task(self._laco_visao())

    def _desligar_visao_se_precisar(self, guild_id: str) -> None:
        """Só desliga quando a ÚLTIMA call sai -- não a cada guild que sai
        individualmente, senão duas calls simultâneas desligariam a visão
        uma da outra."""
        self._guilds_com_call.discard(str(guild_id))
        if self.visao is None or self._guilds_com_call:
            return
        self.visao.desligar()
        if self._tarefa_visao and not self._tarefa_visao.done():
            self._tarefa_visao.cancel()

    async def _laco_visao(self) -> None:
        """Chama SistemaVisual.tick() periodicamente. tick() é síncrono e
        bloqueante (captura de tela +, ocasionalmente, uma chamada de
        rede de alguns segundos ao MiniCPM-V) -- roda via to_thread para
        não travar o resto do event loop (voz, texto, consciência) durante
        esses ~2s de análise.
        """
        while True:
            await asyncio.sleep(self.cfg.visao.tick_intervalo)
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
                if not v.passou:
                    continue

                async with est.processando:
                    c.ocupada = True
                    r = await self.eva.falar_sozinha_async(
                        v.impulso.conteudo, usuario=c.ultimo_falante, modo_voz=True)
                    if r.resposta:
                        print(f"[eva espontânea] {r.resposta}")
                        await self.falar(guild_id, r.resposta)
                        c.ela_falou(espontanea=True)
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
        await self._enviar({"type": "leave", "guild_id": str(guild_id)})

    async def falar(self, guild_id: str, texto: str) -> None:
        """Sintetiza e envia para o Discord."""
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

        est.falando = True
        print(f"[voz] falando {duracao_segundos(pcm):.1f}s")
        # O cabeçalho JSON avisa o bridge que o próximo quadro é áudio
        await self._enviar({"type": "play", "guild_id": str(guild_id)})
        await self.ws.send(pcm)

    # ---------------------------------------------------------- recepção

    async def _ao_receber_audio(self, guild_id: str, user_id: str, pcm: bytes) -> None:
        est = self.estado(guild_id)

        # Ignora o próprio eco: enquanto fala, e por uma janela curta depois
        if est.falando or (time.time() - est.fim_da_fala) < self.janela_eco:
            return

        # Uma conversa por vez: se já está respondendo, descarta em vez de
        # enfileirar -- resposta atrasada a uma fala antiga confunde mais
        # do que ajuda numa conversa por voz.
        if est.processando.locked():
            return

        # Descarta silêncio antes de gastar chamada de API
        if esta_silencioso(pcm):
            return

        async with est.processando:
            wav = pcm_para_wav(pcm)
            try:
                t = await asyncio.to_thread(
                    self.stt.transcrever_bytes, wav, "call.wav",
                    self.cfg.voz.stt_vocabulario,
                )
            except Exception as e:
                print(f"[stt] {e}")
                return

            if parece_ruido(t):
                return

            print(f"[call] {user_id}: {t.texto}")
            self.consciencia(guild_id).alguem_falou(
                str(user_id), t.texto, nome=self._nome(user_id))

            # `usuario=` é o que separa a memória de cada pessoa na call.
            # Sem ele, tudo cai no dono da instância e a EVA cita para um
            # o que o outro contou. `modo_voz=True` acrescenta a linha
            # "MODO: VOZ" treinada e baixa o teto de 400 para 120 tokens --
            # 400 tokens são uns 40 segundos de fala.
            r = await self.eva.responder_async(
                t.texto, usuario=str(user_id), modo_voz=True,
                contexto_visual=self._contexto_visual_para(t.texto))
            if r.erro:
                print(f"[eva] {r.erro}")
                return
            if not r.resposta:
                return

            print(f"[eva] {r.resposta}")
            await self.falar(guild_id, r.resposta)
            self.consciencia(guild_id).ela_falou()

            # Lacuna de conhecimento: a mensagem tocou em algo que pode
            # estar desatualizado no que a EVA sabe. Não bloqueia nada --
            # dispara como task solta; se terminar, vira impulso de
            # iniciativa; se não terminar a tempo ou não achar nada, não
            # acontece nada (sem aviso, sem erro visível).
            if r.plano.possivel_lacuna:
                asyncio.create_task(
                    self._pesquisar_e_registrar(guild_id, r.plano.possivel_lacuna))

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
        if est.processando.locked():
            return

        async with est.processando:
            await self._enviar({"type": "typing", "channel_id": canal})
            autor = str(d.get("author_id") or canal)
            self.consciencia(gid).registrar_nome(autor, d.get("author_name"))
            r = await self.eva.responder_async(
                texto, usuario=autor,
                contexto_visual=self._contexto_visual_para(texto))

            if r.erro:
                await self.enviar_mensagem(
                    canal, f"[modelo indisponível] {r.erro[:300]}", d.get("message_id"))
                return
            if not r.resposta:
                return

            await self.enviar_mensagem(canal, r.resposta, d.get("message_id"))

            # se estiver numa call desse servidor, fala também
            if est.canal_id:
                await self.falar(gid, r.resposta)

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
            t = await asyncio.to_thread(
                self.stt.transcrever_bytes, dados, anexo.get("name", "audio.ogg"),
                self.cfg.voz.stt_vocabulario,
            )
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
            await self._ao_receber_audio(cab["guild_id"], cab["user_id"], bytes(msg))
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
            self._cancelar_laco_consciencia(gid)
            self._tarefas_consciencia[gid] = asyncio.create_task(
                self._laco_consciencia(gid))
            self._ligar_visao_se_precisar(gid)
            print(f"[bridge] entrou em '{d.get('channel_name')}'")
        elif tipo == "left":
            gid = str(d["guild_id"])
            self.estado(gid).canal_id = None
            self._cancelar_laco_consciencia(gid)
            self._desligar_visao_se_precisar(gid)
            motivo = f" ({d['reason']})" if d.get("reason") else ""
            print(f"[bridge] saiu do canal{motivo}")
        elif tipo == "reconnecting":
            print("[bridge] conexão de voz caiu, reconectando...")
        elif tipo == "reconnected":
            print("[bridge] reconectado")
        elif tipo == "play_done":
            est = self.estado(str(d["guild_id"]))
            est.falando = False
            est.fim_da_fala = time.time()
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

    def diagnostico(self) -> dict:
        d = self.eva.diagnostico()
        d.update({
            "bridge_url": self.url,
            "stt": "ok" if self.stt.disponivel() else "sem GROQ_API_KEY",
            "tts": self.tts.nome if self.tts else f"indisponível ({self.erro_tts})",
            "guilds_ativas": [g for g, e in self.guilds.items() if e.canal_id],
        })
        return d

    def fechar(self) -> None:
        for gid in list(self._tarefas_consciencia):
            self._cancelar_laco_consciencia(gid)
        if self._tarefa_visao and not self._tarefa_visao.done():
            self._tarefa_visao.cancel()
        if self.visao:
            self.visao.fechar()
        if self._dashboard:
            self._dashboard.parar()
        self.eva.fechar()