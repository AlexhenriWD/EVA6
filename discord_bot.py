"""
Discord Bot Module — EVA V4
============================

Removidos:
- !diary command (EvaDiary nunca lida pelo LLM)
- vtuber_bridge (não usado ativamente)
- referencias ao _mc_autonomous_loop

Mantidos e limpos:
- Voice commands (!join, !leave, !speak)
- Vision commands (!visionmodels, !visionswitch, !visionwatch, !whatsee)
- Robot commands (!robot, !autonomousrobot)
- Minecraft commands (!mc)
- System commands (!voicestats, !ttstest, !memorystats, !proactive, !autonomous)
- on_message: mencão + DM + canal MC
- Proactive loop
"""

import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, List

import discord
from discord.ext import commands, tasks

try:
    from robot.eva_robot_system import AgentMode
except ImportError:
    AgentMode = None

try:
    from minecraft.minecraft_server_manager import MinecraftServerManager
    _MC_MANAGER_AVAILABLE = True
except ImportError:
    MinecraftServerManager = None
    _MC_MANAGER_AVAILABLE = False
    print("⚠️ MinecraftServerManager não disponível")
    
class EvaDiscordBot:
    """Encapsula toda a lógica do Discord Bot."""

    def __init__(
        self,
        eva_brain,
        voice_system=None,
        tts_manager=None,
        unified_vision=None,
        social_memory=None,
        robot_integration=None,
        identity_manager=None,
        memory_manager=None,
        tools_manager=None,
        visual_reactions=None,
        proactive_system=None,
        minecraft_integration=None,
        consciousness=None,
    ):
        self.eva_brain            = eva_brain
        self.voice_system         = voice_system
        self.tts_manager          = tts_manager
        self.unified_vision       = unified_vision
        self.social_memory        = social_memory
        self.robot_integration    = robot_integration
        self.identity_manager     = identity_manager
        self.memory_manager       = memory_manager
        self.tools_manager        = tools_manager
        self.visual_reactions     = visual_reactions
        self.proactive_system     = proactive_system
        self.minecraft_integration = minecraft_integration
        self.minecraft_bridge     = None
         # 🎮 Server Manager (inicia/para o Node.js)
        self._mc_server_manager = None
        if minecraft_integration is not None and _MC_MANAGER_AVAILABLE:
            node_dir = os.getenv("MC_NODE_DIR", "minecraft")
            self._mc_server_manager = MinecraftServerManager(
                mc_integration=minecraft_integration,
                node_dir=node_dir,
                rpc_port=int(os.getenv("RPC_PORT", "8765")),
                auto_restart=os.getenv("MC_AUTO_RESTART", "true").lower() == "true",
                max_restarts=int(os.getenv("MC_MAX_RESTARTS", "5")),
            )
            self._mc_server_manager.on_node_up   = self._notify_mc_node_up
            self._mc_server_manager.on_node_down = self._notify_mc_node_down
            print(f"✅ MinecraftServerManager pronto | node_dir={node_dir}")
        self.consciousness        = consciousness   # EVAConsciousness

        if self.voice_system:
            # Só sobrescreve o callback se não foi definido externamente
            if not self.voice_system.transcription_callback:
                self.voice_system.transcription_callback = self._on_voice_transcription

        # Bot setup
        intents = discord.Intents.default()
        intents.voice_states   = True
        intents.guilds         = True
        intents.guild_messages = True
        intents.message_content = True
        intents.members        = True

        self.bot = commands.Bot(
            command_prefix='!',
            intents=intents,
            heartbeat_timeout=60.0,
            case_insensitive=True,
        )

        self.eva_is_processing = False
        self.last_eva_response_time: Dict[str, float] = {}
        self.voice_contexts: Dict = {}

        self._register_events()
        self._register_commands()
        self.proactive_loop.start()

    def set_minecraft_bridge(self, bridge):
        self.minecraft_bridge = bridge

    def set_consciousness(self, consciousness):
        """Injeta a EVAConsciousness após construção do bot."""
        self.consciousness = consciousness
        print("🧠 EVAConsciousness conectada ao Discord bot")

   

    # ════════════════════════════════════════════════════════════
    # EVENTS
    # ════════════════════════════════════════════════════════════


    # ── MINECRAFT SERVER CALLBACKS (métodos da classe) ───────────────────

    async def _notify_mc_node_up(self, msg: str):
        channel_id = self._get_mc_notify_channel()
        if channel_id:
            ch = self.bot.get_channel(channel_id)
            if ch:
                await ch.send(f"🟢 **Minecraft** | {msg}")
        self._set_minecraft_context_active(True)

    async def _notify_mc_node_down(self, msg: str):
        channel_id = self._get_mc_notify_channel()
        if channel_id:
            ch = self.bot.get_channel(channel_id)
            if ch:
                await ch.send(f"🔴 **Minecraft** | {msg}")
        self._set_minecraft_context_active(False)

    def _get_mc_notify_channel(self):
        raw   = os.getenv("MC_DISCORD_CHANNEL_ID", "")
        clean = raw.split("#")[0].strip()
        return int(clean) if clean.isdigit() else None

    def _set_minecraft_context_active(self, active: bool):
        try:
            # Atualizar capability no identity_system (defensivo — suporta .core ou direto)
            id_sys = getattr(self.eva_brain, "identity_system", None)
            if id_sys:
                if hasattr(id_sys, "core") and hasattr(id_sys.core, "set_capability"):
                    id_sys.core.set_capability("minecraft", active)
                elif hasattr(id_sys, "set_capability"):
                    id_sys.set_capability("minecraft", active)

            # Atualizar contexto no tools_manager
            rg = getattr(self.eva_brain, "response_generator", None)
            if rg and hasattr(rg, "tools_manager") and rg.tools_manager:
                if active and self.minecraft_integration:
                    try:
                        mc_state = self.minecraft_integration.get_live_snapshot()
                        # set_minecraft_context(user_id, data, ttl) — usar "global" como chave
                        rg.tools_manager.set_minecraft_context("global", mc_state, ttl=30.0)
                    except Exception:
                        pass
                else:
                    try:
                        rg.tools_manager.set_minecraft_context("global", None, ttl=0.0)
                    except Exception:
                        pass

            # Conectar/desconectar Minecraft na EVAConsciousness
            if self.consciousness:
                if active and self.minecraft_integration:
                    self.consciousness.set_minecraft_integration(self.minecraft_integration)
                    # Conectar Brain V6 se disponível
                    brain_obj = getattr(self.eva_brain, 'mc_eva_brain', None)
                    if brain_obj and hasattr(self.consciousness, 'set_mc_brain'):
                        self.consciousness.set_mc_brain(brain_obj)
                    # Ativar consciousness mesmo sem call de voz
                    # Para que o loop processe ações MC mesmo sozinha
                    guild_ids = list(self.consciousness._guilds.keys()) if hasattr(self.consciousness, '_guilds') else []
                    if not guild_ids:
                        # Nenhuma guild ativa — criar uma guild virtual para MC
                        # Usar a primeira guild do bot como referência
                        for guild in self.bot.guilds:   # self.bot.guilds, não self.guilds
                            mc_guild_id = guild.id
                            # Pegar usuários na call, se houver; senão lista vazia
                            vc_members = []
                            for vc in guild.voice_channels:
                                vc_members = [str(m.id) for m in vc.members if not m.bot]
                                if vc_members:
                                    break
                            self.consciousness.activate_guild(mc_guild_id, vc_members)
                            print(f"🎮 [Consciousness] Guild MC ativada via Minecraft: {mc_guild_id}")
                            break
                    else:
                        # Guilds existem — garantir que estão ativas
                        for gid in guild_ids:
                            gs = self.consciousness._guilds.get(gid)
                            if gs and not gs.active:
                                gs.active = True
                                print(f"🎮 [Consciousness] Guild {gid} reativada via MC")
                else:
                    self.consciousness.set_minecraft_integration(None)

            status = "ATIVA ✅" if active else "INATIVA ⭕"
            print(f"[EVA] Consciência Minecraft: {status}")
        except Exception as e:
            print(f"[EVA] Erro ao setar minecraft context: {e}")
            import traceback; traceback.print_exc()

    def _register_events(self):

        @self.bot.event
        async def on_ready():
            await self._on_ready()

        @self.bot.event
        async def on_message(message):
            await self._on_message(message)

        @self.bot.event
        async def on_command_error(ctx, error):
            try:
                await ctx.send(f"❌ Erro: {type(error).__name__}: {str(error)[:150]}")
            except Exception:
                pass
            raise error

        @self.bot.event
        async def on_voice_state_update(member, before, after):
            await self._on_voice_state_update(member, before, after)

        @self.bot.event
        async def on_disconnect():
            await self._on_disconnect()

    async def _on_ready(self):
        print(f'✅ {self.bot.user} online!')
        print(f'📊 Servidores: {len(self.bot.guilds)}')
        if self.proactive_system:
            self.proactive_system.enable()
            print('🧠 Proactive system habilitado')

    async def _on_message(self, message):
        if message.author == self.bot.user:
            return

        await self.bot.process_commands(message)

        is_dm       = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.bot.user in message.mentions

        # Minecraft bridge — responde perguntas sobre o jogo no canal MC
        if self.minecraft_bridge and not is_dm:
            try:
                mc_response = await self.minecraft_bridge.on_discord_message(
                    message=message,
                    author=message.author,
                    channel=message.channel,
                )
                if mc_response:
                    await message.reply(mc_response)
                    if not is_mentioned:
                        return
            except Exception as e:
                print(f"[Bot] minecraft_bridge error: {e}")

        # Fase 4: texto-como-voz — se o autor está na mesma call que a EVA,
        # trata a mensagem digitada como se fosse fala transcrita: responde
        # com context_type='voice' e fala na call, em vez de só responder
        # por texto. Não precisa @mencionar — estar na call já é o gatilho.
        # Pensado pra debug (principalmente Minecraft, o próximo passo):
        # digitar em vez de precisar falar toda hora.
        if not is_dm and not message.content.startswith(self.bot.command_prefix):
            call_content = message.content.replace(f'<@{self.bot.user.id}>', '').strip()
            if call_content:
                handled = await self._handle_call_text_message(message, call_content)
                if handled:
                    return

        if not (is_dm or is_mentioned):
            return

        user_id = str(message.author.id)
        content = message.content.replace(f'<@{self.bot.user.id}>', '').strip()
        if not content:
            return

        # Registrar social
        if self.social_memory:
            try:
                await self.social_memory.register_or_update_user(
                    user_id=user_id,
                    discord_username=message.author.name,
                    display_name=message.author.display_name,
                )
                await self.social_memory.update_conversation_style(user_id, content)
            except Exception:
                pass

        # Registrar proactive response
        if self.proactive_system:
            self.proactive_system.register_user_response(
                guild_id=message.guild.id if message.guild else 0,
                user_id=user_id,
                message=content,
            )

        identity = await self.identity_manager.register_user(message.author) \
            if self.identity_manager else {}

        async with message.channel.typing():
            response = await self.eva_brain.process_message(
                user_id=user_id,
                message=content,
                identity=identity,
                context_type='dm' if is_dm else 'text',
                discord_message=message,
            )

            if isinstance(response, dict):
                text = response.get('response') or response.get('text', '')
            else:
                text = str(response)

            if text:
                # Quebrar mensagens longas
                if len(text) > 1900:
                    for i in range(0, len(text), 1900):
                        await message.reply(text[i:i+1900])
                else:
                    await message.reply(text)

    async def _on_voice_state_update(self, member, before, after):
        if not self.voice_system:
            return
        guild_id = member.guild.id

        # Usuário entrou no mesmo canal da EVA
        if after.channel and self.bot.user in after.channel.members:
            if member != self.bot.user:
                if self.consciousness:
                    asyncio.create_task(
                        self.consciousness.on_user_joined(guild_id, str(member.id))
                    )

        # Usuário saiu do canal onde EVA estava
        if before.channel and self.bot.user in before.channel.members:
            if member != self.bot.user and not after.channel:
                if self.consciousness:
                    asyncio.create_task(
                        self.consciousness.on_user_left(guild_id, str(member.id))
                    )

        # EVA sozinha → sai automaticamente
        if before.channel and self.bot.user in before.channel.members:
            humans = [m for m in before.channel.members if not m.bot]
            if not humans:
                if self.consciousness:
                    self.consciousness.deactivate_guild(guild_id)
                await self.voice_system.disconnect(guild_id)

    async def _notify_proactive_user_joined(self, guild_id, user_id):
        """Mantido por compatibilidade — EVAConsciousness cuida da saudação."""
        if self.consciousness:
            asyncio.create_task(
                self.consciousness.on_user_joined(guild_id, str(user_id))
            )

    async def _on_disconnect(self):
        print("⚠️ Bot desconectado do Discord")

    async def _handle_call_text_message(self, message, content: str) -> bool:
        """
        Fase 4 — texto-como-voz.

        Se o autor da mensagem está no mesmo canal de voz em que a EVA está
        conectada (guild-level; não valida canal exato — ver observação no
        chat), trata o texto digitado como se fosse uma fala transcrita:
        gera a resposta com context_type='voice' (mesmo tom/brevidade que
        ela usaria falando) e responde na call via TTS, além de ecoar o
        texto no canal (bom pra debug — dá pra ler e ouvir ao mesmo tempo).

        Espelha o que handle_voice_transcription() faz em main.py pro áudio
        real do VoiceBridgeClient, incluindo a notificação pra
        consciousness.on_speech_transcribed() — sem isso, o sistema de
        humor/proatividade não saberia que o usuário "respondeu" e o
        contador de "sendo ignorado" não seria resetado.

        Returns:
            True se a mensagem foi tratada aqui (quem chamou não deve
            processá-la de novo como texto normal). False se as condições
            (EVA conectada em voz + autor em algum canal de voz) não
            bateram — nesse caso o fluxo normal de texto/DM continua.
        """
        if not self.voice_system or not message.guild:
            return False

        guild_id = message.guild.id
        if not self.voice_system.is_connected(guild_id):
            return False

        # message.author aqui é discord.Member (a mensagem veio de uma guild,
        # não de DM) — .voice só existe em Member, não em User.
        voice_state = getattr(message.author, "voice", None)
        if not voice_state or not voice_state.channel:
            return False  # autor não está em nenhum canal de voz agora

        user_id = str(message.author.id)

        # Mesmo lock que o VoiceBridgeClient usa pra processar áudio —
        # evita responder duas vezes se ele falar e digitar quase junto.
        # hasattr() porque isso é específico do VoiceBridgeClient; se um dia
        # voice_system for outra implementação, degrada pra sem-lock.
        lock = None
        if hasattr(self.voice_system, "_response_lock"):
            lock = self.voice_system._response_lock.setdefault(guild_id, asyncio.Lock())

        async def _process():
            if self.consciousness:
                try:
                    self.consciousness.on_speech_transcribed(guild_id, user_id, content)
                except Exception as e:
                    print(f"[Bot] consciousness.on_speech_transcribed (texto-na-call) erro: {e}")

            identity = await self.identity_manager.register_user(message.author) \
                if self.identity_manager else {}

            # Fase 4: mesmo lock que a consciousness usa pras falas espontâneas
            # dela (begin_responding/end_responding) — cobre geração E fala,
            # não só geração. Sem isso, uma fala espontânea podia começar a
            # gerar/falar bem na hora em que você mandava texto na call, e as
            # duas competiam pelo mesmo slot do LM Studio e atropelavam o
            # áudio uma da outra.
            if self.consciousness:
                await self.consciousness.begin_responding()
            try:
                try:
                    response = await self.eva_brain.process_message(
                        user_id=user_id,
                        message=content,
                        identity=identity,
                        context_type="voice",
                        call_context={"source": "text_while_in_call"},
                        use_tools=True,
                        use_vision=True,
                    )
                except Exception as e:
                    import traceback
                    print(f"⚠️ texto-na-call generate erro: {e}")
                    traceback.print_exc()
                    return

                resp_text = (
                    response.get("response") or response.get("text", "")
                    if isinstance(response, dict) else str(response or "")
                )
                if not resp_text:
                    return

                try:
                    await self.voice_system.speak_text(guild_id, resp_text)
                except Exception as e:
                    print(f"⚠️ speak_text (texto-na-call) erro: {e}")

                # Ecoa por texto também — pra debug, ver e ouvir ao mesmo tempo.
                try:
                    await message.reply(f"🎙️ {resp_text}")
                except Exception:
                    pass
            finally:
                if self.consciousness:
                    self.consciousness.end_responding()

        if lock:
            async with lock:
                await _process()
        else:
            await _process()

        return True

    async def _on_voice_transcription(self, guild_id, user_id, text, call_context=None):
        """Callback chamado pelo voice_system quando há transcrição."""
        if not self.eva_brain:
            return

        # NÃO chamar consciousness.on_speech_transcribed aqui —
        # VoiceBridgeClient._process_user_audio já fez isso antes de invocar este callback.
        # Chamar de novo causaria buffer duplicado e silêncio calculado errado.

        # get_identity é síncrono — sem await
        identity = self.identity_manager.get_identity(str(user_id)) \
            if self.identity_manager else {}
        if identity is None:
            identity = {}

        # Usar generate() — IntegratedResponseGenerator V5 não tem generate_stream()
        try:
            response = await self.eva_brain.process_message(
                user_id=str(user_id),
                message=text,
                identity=identity,
                context_type="voice",
                call_context=call_context or {},
                use_tools=True,
                use_vision=True,
            )
        except Exception as e:
            import traceback
            print(f"⚠️ voice transcription generate erro: {e}")
            traceback.print_exc()
            return

        resp_text = (
            response.get("response") or response.get("text", "")
            if isinstance(response, dict) else str(response or "")
        )
        if resp_text and self.voice_system and self.voice_system.is_connected(int(guild_id)):
            try:
                await self.voice_system.speak_text(int(guild_id), resp_text)
            except Exception as e:
                print(f"⚠️ speak_text erro: {e}")

    # ════════════════════════════════════════════════════════════
    # PROACTIVE LOOP
    # ════════════════════════════════════════════════════════════

    @tasks.loop(seconds=30)
    async def proactive_loop(self):
        """
        Loop de keepalive — EVAConsciousness tem seu próprio loop interno.
        Este loop apenas verifica se a consciência está ativa e faz health check.
        """
        if not self.consciousness:
            return
        # Health check: se a consciência parou por algum erro, reiniciar
        if hasattr(self.consciousness, '_running') and not self.consciousness._running:
            print("⚠️ [Bot] EVAConsciousness parou — reiniciando loop...")
            try:
                self.consciousness.start()
            except Exception as e:
                print(f"❌ [Bot] Falha ao reiniciar consciousness: {e}")

    @proactive_loop.before_loop
    async def before_proactive_loop(self):
        await self.bot.wait_until_ready()

    # ════════════════════════════════════════════════════════════
    # COMMANDS
    # ════════════════════════════════════════════════════════════

    def _register_commands(self):

        # ── VOICE ─────────────────────────────────────────────

        @self.bot.command(name='join')
        async def join_voice(ctx):
            await self._join_voice(ctx)

        @self.bot.command(name='leave')
        async def leave_voice(ctx):
            await self._leave_voice(ctx)

        @self.bot.command(name='speak')
        async def speak_test(ctx, *, text: str = None):
            await self._speak_test(ctx, text)

        # ── VISION ────────────────────────────────────────────

        @self.bot.command(name='visionmodels')
        async def vision_models_cmd(ctx):
            await self._vision_models(ctx)

        @self.bot.command(name='visionswitch')
        async def vision_switch_cmd(ctx, provider: str = None, *, model: str = None):
            await self._vision_switch(ctx, provider, model)

        @self.bot.command(name='visionwatch')
        async def visionwatch_cmd(ctx, action: str = "start", focus: str = "full"):
            guild_id = ctx.guild.id
            user_id  = str(ctx.author.id)
            if not self.visual_reactions:
                await ctx.send("❌ Visual Reaction System não disponível.")
                return
            if action.lower() == "start":
                self.visual_reactions.enable_for_guild(
                    guild_id=guild_id, user_id=user_id,
                    focus=focus, react_to_changes=True, analysis_interval=5.0,
                )
                await ctx.send("👁️ Vision watch iniciado.")
            elif action.lower() == "stop":
                self.visual_reactions.disable_for_guild(guild_id)
                await ctx.send("🔇 Vision watch parado.")
            else:
                await ctx.send("Uso: !visionwatch start|stop [focus]")

        @self.bot.command(name='whatsee')
        async def whats_see_cmd(ctx):
            await self._whats_see(ctx)

        # ── ROBOT ─────────────────────────────────────────────

        @self.bot.command(name='robot')
        async def robot_command(ctx, subcommand: str = None, *args):
            await self._robot_command(ctx, subcommand, args)

        @self.bot.command(name='autonomousrobot')
        async def autonomous_robot_cmd(ctx, action: str = 'status'):
            if not self.robot_integration:
                await ctx.send("❌ Robô não conectado.")
                return
            await ctx.send(f"🤖 Robot status: em desenvolvimento")

        # ── MINECRAFT ─────────────────────────────────────────

        @self.bot.command(name='mc')
        async def mc_command(ctx, action: str = "status", *, args: str = ""):
            await self._mc_command(ctx, action, args)

        # ── SYSTEM ────────────────────────────────────────────

        @self.bot.command(name='voicestats')
        async def voice_stats_cmd(ctx):
            await self._voice_stats(ctx)

        @self.bot.command(name='ttstest')
        async def tts_test_cmd(ctx, *, text: str = "Testando o sistema de síntese de voz"):
            await self._tts_test(ctx, text)

        @self.bot.command(name='memorystats')
        async def memory_stats_cmd(ctx):
            await self._memory_stats(ctx)

        @self.bot.command(name='proactive')
        async def proactive_cmd(ctx, action: str = 'status'):
            await self._proactive_cmd(ctx, action)

        @self.bot.command(name='autonomous')
        async def autonomous_cmd(ctx, action: str = 'status'):
            await self._autonomous_cmd(ctx, action)

        @self.bot.command(name='status')
        async def status_cmd(ctx):
            await self._system_status(ctx)

    # ════════════════════════════════════════════════════════════
    # IMPLEMENTAÇÕES DOS COMANDOS
    # ════════════════════════════════════════════════════════════

    # ── Voice ─────────────────────────────────────────────────

    async def _join_voice(self, ctx):
        if not self.voice_system:
            await ctx.reply("❌ Voice system não disponível")
            return
        if not ctx.author.voice:
            await ctx.reply("❌ Você precisa estar em um canal de voz")
            return

        guild_id = ctx.guild.id
        channel  = ctx.author.voice.channel  # objeto VoiceChannel

        # Se o bot já está conectado em algum canal desta guild,
        # desconectar antes para evitar "Already connected to a voice channel"
        existing_vc = ctx.guild.voice_client
        if existing_vc:
            try:
                await existing_vc.disconnect(force=True)
            except Exception:
                pass
        # Também limpar estado interno do voice_system se houver
        if guild_id in self.voice_system.guilds:
            await self.voice_system.disconnect(guild_id)

        success = await self.voice_system.connect(guild_id, channel)
        if success:
            if self.proactive_system:
                self.proactive_system.enable_for_guild(
                    guild_id, text_channel_id=ctx.channel.id
                )

            # Ativar EVAConsciousness e disparar apresentação natural
            if self.consciousness:
                user_ids = [str(m.id) for m in channel.members if not m.bot]
                asyncio.create_task(
                    self.consciousness.on_eva_joined(guild_id, user_ids)
                )

            await ctx.reply("✅ Conectei!")
        else:
            await ctx.reply("❌ Falha ao conectar")

    async def _leave_voice(self, ctx):
        if not self.voice_system:
            return
        guild_id = ctx.guild.id
        await self.voice_system.disconnect(guild_id)
        if self.proactive_system:
            self.proactive_system.disable_for_guild(guild_id)
        # Desativar consciência ao sair
        if self.consciousness:
            self.consciousness.deactivate_guild(guild_id)
        await ctx.reply("👋 Saí do canal de voz")

    async def _speak_test(self, ctx, text):
        if not text:
            await ctx.reply("Uso: !speak <texto>")
            return
        if not self.tts_manager:
            await ctx.reply("❌ TTS não disponível")
            return
        await ctx.reply(f"🔊 Falando: {text[:50]}...")
        try:
            audio = await self.tts_manager.generate_for_discord(text)
            if audio and self.voice_system and self.voice_system.is_connected(ctx.guild.id):
                await self.voice_system.play_audio(ctx.guild.id, audio)
        except Exception as e:
            await ctx.reply(f"❌ Erro TTS: {e}")

    # ── Vision ────────────────────────────────────────────────

    async def _vision_models(self, ctx):
        if not self.unified_vision:
            await ctx.reply("❌ Vision system não disponível")
            return
        available = self.unified_vision.get_available_models() \
            if hasattr(self.unified_vision, 'get_available_models') else {}
        await ctx.reply(f"👁️ Modelos de visão: {available}")

    async def _vision_switch(self, ctx, provider, model):
        if not self.unified_vision:
            await ctx.reply("❌ Vision system não disponível")
            return
        await ctx.reply(f"🔄 Provider: {provider} | Modelo: {model}")

    async def _whats_see(self, ctx):
        user_id = str(ctx.author.id)
        if not self.tools_manager:
            await ctx.reply("❌ Tools manager não disponível")
            return
        result = await self.tools_manager.execute_tool(
            'analyze_screen', {'focus': 'full'}, user_id=user_id
        )
        await ctx.reply(f"👁️ {result[:1800]}")

    # ── Robot ─────────────────────────────────────────────────

    async def _robot_command(self, ctx, subcommand, args):
        if not self.robot_integration:
            await ctx.reply("❌ Robô não conectado")
            return
        if subcommand is None:
            await ctx.reply("Subcomandos: status, move, stop, express")
            return
        await ctx.reply(f"🤖 Robot [{subcommand}]: em desenvolvimento")

    # ── Minecraft ─────────────────────────────────────────────

    async def _mc_command(self, ctx, action: str, args: str):
        """Handler para !mc <action> [args]"""
        mc     = self.minecraft_integration
        action = (action or "status").lower().strip()
        args   = (args or "").strip()

        # ═══════════════════════════════════════════════════════════════
        # GRUPO 1 — Comandos que NÃO precisam de minecraft_integration
        # (ServerManager gerencia o processo Node.js independentemente)
        # ═══════════════════════════════════════════════════════════════

        if action == "start":
            mgr = getattr(self, "_mc_server_manager", None)
            if not mgr:
                await ctx.send(
                    "❌ MinecraftServerManager não disponível.\n"
                    "Verifique se `MINECRAFT_ENABLED=true` está no `.env`."
                )
                return
            if mgr.node_running:
                pid = mgr._process.pid if mgr._process else "?"
                rpc = "✅" if mc and mc.connected else "❌ (use `!mc connect`)"
                await ctx.send(f"⚠️ Node.js já está rodando.\nPID: `{pid}` | RPC: {rpc}")
                return
            msg = await ctx.send("🚀 Iniciando bot Minecraft...")
            async def _notify(text: str):
                await msg.edit(content=text)
            ok = await mgr.start_node(notify_fn=_notify)
            if ok and mc and mc.connected:
                self._set_minecraft_context_active(True)
            return

        if action in ("stop-server", "stop_server", "kill"):
            mgr = getattr(self, "_mc_server_manager", None)
            if not mgr:
                await ctx.send("❌ MinecraftServerManager não disponível.")
                return
            if not mgr.node_running:
                await ctx.send("⚠️ Node.js não está rodando.")
                return
            msg = await ctx.send("🔌 Encerrando Node.js...")
            await mgr.stop_node()
            self._set_minecraft_context_active(False)
            await msg.edit(content="✅ Node.js encerrado e RPC desconectado.")
            return

        if action in ("node-status", "node_status", "ps"):
            mgr = getattr(self, "_mc_server_manager", None)
            if not mgr:
                await ctx.send("❌ MinecraftServerManager não disponível.")
                return
            s = mgr.get_status()
            embed = discord.Embed(
                title="🖥️ Status do Processo Node.js",
                color=discord.Color.green() if s["node_running"] else discord.Color.red(),
            )
            embed.add_field(name="Processo",         value="✅ Rodando" if s["node_running"] else "❌ Parado", inline=True)
            embed.add_field(name="RPC Python",       value="✅ Conectado" if s["rpc_connected"] else "❌ Desconectado", inline=True)
            embed.add_field(name="PID",              value=str(s["pid"] or "—"), inline=True)
            embed.add_field(name='Reinicializações', value="{}/{}".format(s.get('restart_count',0), s.get('max_restarts',0)), inline=True)
            embed.add_field(name="Auto-restart",     value="✅" if s["auto_restart"] else "❌", inline=True)
            embed.add_field(name='Diretório Node',   value=f"`{s.get('node_dir','?')}`", inline=False)
            embed.set_footer(text="!mc start | !mc stop-server | !mc connect")
            await ctx.send(embed=embed)
            return

        if action == "help":
            embed = discord.Embed(title="🎮 Comandos Minecraft — EVA", color=discord.Color.green())
            embed.add_field(
                name="🖥️ Processo Node.js",
                value=(
                    "`!mc start` — iniciar Node.js + conectar EVA\n"
                    "`!mc stop-server` — encerrar Node.js\n"
                    "`!mc node-status` — PID, restarts, estado"
                ), inline=False)
            embed.add_field(
                name="📡 Conexão RPC",
                value=(
                    "`!mc connect` — conectar ao RPC (Node já rodando)\n"
                    "`!mc disconnect` — desconectar RPC\n"
                    "`!mc status` — estado completo do bot"
                ), inline=False)
            embed.add_field(
                name="🎮 Ações",
                value=(
                    "`!mc follow <nome>` — seguir jogador\n"
                    "`!mc stop` — parar tudo\n"
                    "`!mc say <msg>` — falar no chat\n"
                    "`!mc mine <bloco> [qty]` — minerar\n"
                    "`!mc craft <item> [qty]` — craftar\n"
                    "`!mc goto <x> <y> <z>` — ir para coordenadas\n"
                    "`!mc attack <alvo>` — atacar\n"
                    "`!mc mode <modo>` — mudar modo (assistant/companion/autonomous)"
                ), inline=False)
            await ctx.send(embed=embed)
            return

        # ═══════════════════════════════════════════════════════════════
        # GRUPO 2 — Comandos que precisam de minecraft_integration
        # ═══════════════════════════════════════════════════════════════

        if not mc:
            await ctx.send(
                "❌ Minecraft integration não disponível.\n"
                "Verifique se `MINECRAFT_ENABLED=true` está no `.env` e use `!mc start`."
            )
            return

        # status — funciona mesmo sem RPC conectado
        if action == "status":
            is_conn = getattr(mc, "connected", False)
            if is_conn:
                try:
                    await mc.get_state()
                except Exception:
                    pass
            state = mc.state
            embed = discord.Embed(
                title="🎮 EVA no Minecraft",
                color=discord.Color.green() if is_conn else discord.Color.orange(),
            )
            embed.add_field(name="RPC", value="✅ Conectado" if is_conn else "❌ Desconectado\nUse `!mc connect`", inline=True)
            embed.add_field(name="Modo", value=mc.mode, inline=True)
            if is_conn and state:
                embed.add_field(name="HP",    value=f"{state.health:.0f}/20", inline=True)
                embed.add_field(name="Fome",  value=f"{state.food:.0f}/20",   inline=True)
                embed.add_field(name="Bioma", value=state.biome,              inline=True)
                embed.add_field(name="Dim",   value=state.dimension,          inline=True)
                if state.position:
                    embed.add_field(
                        name="Posição",
                        value=f"x={state.position.x:.0f} y={state.position.y:.0f} z={state.position.z:.0f}",
                        inline=False,
                    )
                if state.nearby_hostiles:
                    embed.add_field(
                        name="⚠️ Ameaças",
                        value=", ".join(f"{e.name}({e.distance:.0f}m)" for e in state.nearby_hostiles[:3]),
                        inline=False,
                    )
            else:
                embed.set_footer(text="Node.js deve estar rodando antes de !mc connect")
            await ctx.send(embed=embed)
            return

        if action == "connect":
            if getattr(mc, "connected", False):
                await ctx.send("✅ Já conectado! Use `!mc status` para ver o estado.")
                return
            msg = await ctx.send(f"🔌 Conectando ao RPC em `{mc.rpc_url}`...")
            ok = await mc.connect_rpc()
            if ok:
                try:
                    await mc.get_state()
                except Exception:
                    pass
                state = mc.state
                await msg.edit(content=(
                    f"✅ Conectado ao Node.js!\n"
                    f"Bot: `{getattr(state, 'bot_username', '?')}` | "
                    f"HP: {getattr(state, 'health', 0):.0f}/20 | "
                    f"Modo: `{mc.mode}`"
                ))
                self._set_minecraft_context_active(True)
            else:
                await msg.edit(content=(
                    f"❌ Falha ao conectar em `{mc.rpc_url}`.\n"
                    "Verifique se `node index.js` está rodando (`!mc start`)."
                ))
            return

        if action == "disconnect":
            await mc.disconnect()
            self._set_minecraft_context_active(False)
            await ctx.send("🔌 RPC desconectado.")
            return

        if action == "mode":
            mode = args.lower()
            if not mode:
                await ctx.send("❌ Uso: `!mc mode <assistant|companion|autonomous>`")
                return
            result = mc.set_mode(mode)
            await ctx.send(f"🎮 {result}")
            return

        # ── Comandos abaixo precisam de RPC conectado ──────────────────
        if not getattr(mc, "connected", False):
            await ctx.send("❌ Não conectado ao Node.js. Use `!mc connect` primeiro.")
            return

        # Tentar bridge primeiro para comandos sociais avançados
        if self.minecraft_bridge:
            BRIDGE_ACTIONS = {
                "where", "onde", "inventory", "inv", "inventário",
                "follow", "seguir", "stop", "para", "parar",
                "gather", "pegar", "mine", "minerar",
                "protect", "proteger", "helpme", "objective", "objetivo",
                "intents", "queue", "fila",
            }
            if action in BRIDGE_ACTIONS:
                try:
                    result = await self.minecraft_bridge.handle_discord_mc_command(ctx, action, args)
                    if result:
                        await ctx.send(result)
                    return
                except Exception as e:
                    print(f"[Bot] mc bridge command error: {e}")

        if action == "say":
            if not args:
                await ctx.send("❌ Uso: `!mc say <mensagem>`")
                return
            result = await mc.chat(args)
            await ctx.send(f"💬 {result}")
            return

        if action == "follow":
            target = args
            if not target:
                players = [p.name for p in mc.state.nearby_players]
                target = players[0] if players else ""
            if not target:
                await ctx.send("❌ Nenhum jogador próximo. Use: `!mc follow <nome>`")
                return
            result = await mc.follow(target)
            await ctx.send(f"🎮 {result}")
            return

        if action in ("stop", "para", "parar"):
            result = await mc.stop()
            await ctx.send(f"🎮 {result}")
            return

        if action in ("mine", "minerar"):
            parts = args.split()
            if not parts:
                await ctx.send("❌ Uso: `!mc mine <bloco> [quantidade]`")
                return
            block = parts[0]
            qty   = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            result = await mc.mine(block, qty)
            await ctx.send(f"⛏️ {result}")
            return

        if action == "craft":
            parts = args.split()
            if not parts:
                await ctx.send("❌ Uso: `!mc craft <item> [quantidade]`")
                return
            item = parts[0]
            qty  = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            result = await mc.craft(item, qty)
            await ctx.send(f"🔨 {result}")
            return

        if action == "goto":
            parts = args.split()
            if len(parts) < 3:
                await ctx.send("❌ Uso: `!mc goto <x> <y> <z>`")
                return
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                result = await mc.goto(x, y, z)
                await ctx.send(f"🎮 {result}")
            except ValueError:
                await ctx.send("❌ Coordenadas inválidas.")
            return

        if action == "attack":
            if not args:
                await ctx.send("❌ Uso: `!mc attack <alvo>`")
                return
            result = await mc.attack(args)
            await ctx.send(f"⚔️ {result}")
            return

        if action in ("inventory", "inv", "inventário"):
            try:
                result = await mc.get_inventory_detail()
                items  = result.get("inventory", [])
                if items:
                    lines_inv = [f"• {i['name']}×{i['count']}" for i in items[:20]]
                    await ctx.send("🎒 **Inventário:**\n" + "\n".join(lines_inv))
                else:
                    await ctx.send("🎒 Inventário vazio.")
            except Exception as e:
                await ctx.send(f"❌ Erro: {e}")
            return

        await ctx.send(f"❓ Ação desconhecida: `{action}`. Use `!mc help`.")


    # ── System ────────────────────────────────────────────────

    async def _voice_stats(self, ctx):
        if not self.voice_system:
            await ctx.reply("❌ Voice system não disponível")
            return
        stats = self.voice_system.get_stats()
        await ctx.reply(f"**🎙️ Voice Stats**\n```{stats}```")

    async def _tts_test(self, ctx, text):
        if not self.tts_manager:
            await ctx.reply("❌ TTS não disponível")
            return
        import time
        start = time.time()
        try:
            audio = await self.tts_manager.generate_for_discord(text)
            elapsed = time.time() - start
            if audio:
                duration = len(audio) / (48000 * 2 * 2)
                await ctx.reply(
                    f"✅ TTS OK | Geração: {elapsed:.2f}s | Áudio: {duration:.2f}s"
                )
            else:
                await ctx.reply("⚠️ TTS gerou áudio vazio")
        except Exception as e:
            await ctx.reply(f"❌ Erro TTS: {e}")

    async def _memory_stats(self, ctx):
        if not self.memory_manager:
            await ctx.reply("❌ Memory manager não disponível")
            return
        user_id = str(ctx.author.id)
        try:
            summary = await self.memory_manager.get_memory_summary(user_id)
            await ctx.reply(
                f"**🧠 Memória**\n"
                f"Curto prazo: {summary.get('short_term_cache_size', '?')} msgs\n"
                f"Médio prazo: {summary.get('total_messages', '?')} mensagens\n"
                f"Longo prazo: {summary.get('knowledge_items', '?')} itens"
            )
        except Exception as e:
            await ctx.reply(f"❌ Erro: {e}")

    async def _proactive_cmd(self, ctx, action):
        if not self.proactive_system:
            await ctx.reply("⚠️ Proactive system não disponível")
            return
        if action == 'status':
            stats = self.proactive_system.get_stats()
            await ctx.reply(
                f"**🧠 Proactive V3**\n"
                f"Global: {'✅ ATIVO' if stats['enabled'] else '❌ INATIVO'}\n"
                f"Guilds ativas: {stats['active_guilds']}\n"
                f"Iniciativas: {stats['total_initiatives']}\n"
                f"Loop rodando: {'✅' if stats.get('running') else '❌'}"
            )
        elif action == 'enable':
            self.proactive_system.enable()
            self.proactive_system.enable_for_guild(
                ctx.guild.id, text_channel_id=ctx.channel.id
            )
            await ctx.reply("✅ Proactive HABILITADO para esta guild")
        elif action == 'disable':
            self.proactive_system.disable()
            await ctx.reply("❌ Proactive DESABILITADO")
        else:
            await ctx.reply("Comandos: `status`, `enable`, `disable`")

    async def _autonomous_cmd(self, ctx, action):
        if not self.proactive_system:
            await ctx.reply("⚠️ Sistema autônomo não disponível")
            return
        await self._proactive_cmd(ctx, action)

    async def _system_status(self, ctx):
        cs = self.consciousness
        cs_status = "❌"
        if cs:
            cs_running = getattr(cs, '_running', False)
            cs_guilds  = len(getattr(cs, '_guilds', {}))
            cs_status  = f"✅ ({cs_guilds} guild(s))" if cs_running else "⚠️ parada"
        lines = [
            "**📊 EVA System Status**",
            f"LM Studio: {'✅' if self.eva_brain and self.eva_brain.lm_client else '❌'}",
            f"Voice: {'✅' if self.voice_system else '❌'}",
            f"TTS: {'✅' if self.tts_manager else '❌'}",
            f"Vision: {'✅' if self.unified_vision else '❌'}",
            f"Tools: {'✅' if self.tools_manager else '❌'}",
            f"Consciência: {cs_status}",
            f"Robot: {'✅' if self.robot_integration else '❌'}",
            f"Minecraft: {'✅' if self.minecraft_integration and getattr(self.minecraft_integration, 'connected', False) else '❌'}",
        ]
        await ctx.reply("\n".join(lines))

    # ════════════════════════════════════════════════════════════
    # RUN
    # ════════════════════════════════════════════════

    async def run(self, token: str):
        try:
            await self.bot.start(token)
        except discord.LoginFailure:
            print("❌ Token Discord inválido")
        except Exception as e:
            print(f"❌ Erro ao iniciar bot: {e}")
            raise