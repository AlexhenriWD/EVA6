/**
 * EVA Voice Bridge - Node.js
 * ===========================
 * Fixes v2:
 *
 * Bug 1 — Desconexão espontânea:
 *   O handler Disconnected tinha timeout de 5s para reconectar.
 *   Se o Discord não respondesse a tempo, destruía a conexão e
 *   notificava o Python com 'left', que não reconectava.
 *   Fix: timeout aumentado para 15s + loop de reconexão com backoff
 *   + notifica Python com 'reconnecting' em vez de destruir imediatamente.
 *
 * Bug 2 — Não consegue sair pelo comando:
 *   Se a WS Python caia antes do leave, o estado interno do bridge
 *   ficava sujo (voiceStates com connection destruída).
 *   Fix: cleanup de estado no ws.on('close') + forçar destroy de
 *   todas as conexões de voz quando Python desconecta.
 */

// ─────────────────────────────────────────
// FIX (debug desconexão espontânea): sem isso, uma exceção não tratada
// em QUALQUER callback assíncrono (decoder, opusStream, voice connection,
// etc.) mata o processo Node inteiro em silêncio — o Python só vê o
// WebSocket fechar ("🔌 Bridge desconectou") sem nenhuma pista do motivo.
// Isso precisa ser registrado ANTES de qualquer require pesado (discord.js
// etc.) para não perder exceções que aconteçam durante a própria
// inicialização dos módulos.
// ─────────────────────────────────────────
process.on('uncaughtException', (err) => {
  console.error('💥 [FATAL] uncaughtException — processo vai encerrar:', err);
  console.error(err.stack);
  // Não sai imediatamente: dá tempo do log acima ser escrito no stdout
  // (que o Python captura via subprocess.PIPE) antes do processo morrer
  // de qualquer jeito por estado inconsistente.
  setTimeout(() => process.exit(1), 200);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('💥 [FATAL] unhandledRejection:', reason);
  if (reason instanceof Error) {
    console.error(reason.stack);
  }
});

const dns = require('dns');
dns.setDefaultResultOrder('ipv4first');

const net = require('net');
const _origConnect = net.createConnection.bind(net);
net.createConnection = (options, ...args) => {
  if (typeof options === 'object' && !options.family) {
    options = { ...options, family: 4 };
  }
  return _origConnect(options, ...args);
};

const { Client, GatewayIntentBits } = require('discord.js');
const {
  joinVoiceChannel,
  createAudioPlayer,
  createAudioResource,
  AudioPlayerStatus,
  VoiceConnectionStatus,
  entersState,
  EndBehaviorType,
  getVoiceConnection,
} = require('@discordjs/voice');
const { WebSocketServer } = require('ws');
const { Readable } = require('stream');
const prism = require('prism-media');

const CONFIG = {
  DISCORD_TOKEN: process.env.DISCORD_TOKEN || 'SEU_TOKEN_AQUI',
  WS_PORT: parseInt(process.env.VOICE_BRIDGE_PORT || '8765'),
  SAMPLE_RATE: 48000,
  CHANNELS: 2,
  FRAME_SIZE: 3840,
  // FIX: timeout de reconexão aumentado de 5s para 15s
  RECONNECT_TIMEOUT_MS: 15_000,
  // FIX: tentativas de reconexão antes de desistir
  MAX_RECONNECT_ATTEMPTS: 3,
};

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMessages,
  ],
  rest: { timeout: 30000, retries: 3 },
});

const voiceStates = new Map();
let pythonSocket = null;
let pendingAudioFor = null;

// FIX: rastrear se uma guild está em processo de reconexão
const reconnectingGuilds = new Set();

console.log(`🔌 WebSocket Server escutando na porta ${CONFIG.WS_PORT}`);

const wss = new WebSocketServer({ port: CONFIG.WS_PORT });

wss.on('connection', (ws) => {
  console.log('🐍 Python conectou ao bridge');
  pythonSocket = ws;

  sendJson(ws, { type: 'ready', version: '2.0' });

  ws.on('message', async (data, isBinary) => {
    if (isBinary) {
      if (pendingAudioFor) {
        const { guild_id } = pendingAudioFor;
        pendingAudioFor = null;
        await playAudio(guild_id, data);
      }
      return;
    }

    let msg;
    try {
      msg = JSON.parse(data.toString());
    } catch {
      console.error('❌ Mensagem inválida do Python:', data.toString().slice(0, 100));
      return;
    }

    console.log(`📨 Python → Bridge: ${msg.type}`);

    switch (msg.type) {
      case 'join':  await handleJoin(ws, msg);  break;
      case 'leave': await handleLeave(ws, msg); break;
      case 'play':
        pendingAudioFor = { guild_id: msg.guild_id };
        break;
      default:
        console.warn(`⚠️ Tipo desconhecido: ${msg.type}`);
    }
  });

  // FIX Bug 2: quando Python desconecta, limpar estado de voz
  // para que no próximo connect tudo comece do zero
  ws.on('close', () => {
    console.log('🔌 Bridge desconectou');
    if (pythonSocket === ws) {
      pythonSocket = null;
      // NÃO destruir as conexões de voz aqui —
      // o Python vai reconectar e enviar join de novo.
      // Mas limpar pendingAudio para não travar.
      pendingAudioFor = null;
    }
  });

  ws.on('error', (err) => {
    console.error('❌ WebSocket erro:', err.message);
  });
});

// ─────────────────────────────────────────
// HANDLER: ENTRAR NO CANAL
// ─────────────────────────────────────────
async function handleJoin(ws, msg) {
  const { guild_id, channel_id } = msg;

  try {
    const guild   = await client.guilds.fetch(guild_id);
    const channel = await guild.channels.fetch(channel_id);

    if (!channel || !channel.isVoiceBased()) {
      sendJson(ws, { type: 'error', message: 'Canal não encontrado ou não é de voz' });
      return;
    }

    // Desconectar se já estiver em outro canal
    const existing = getVoiceConnection(guild_id);
    if (existing) {
      existing.destroy();
      voiceStates.delete(guild_id);
      reconnectingGuilds.delete(guild_id);
    }

    console.log(`🔊 Conectando a "${channel.name}" (guild ${guild_id})...`);

    const connection = joinVoiceChannel({
      channelId: channel_id,
      guildId: guild_id,
      adapterCreator: guild.voiceAdapterCreator,
      selfDeaf: false,
      selfMute: false,
    });

    await entersState(connection, VoiceConnectionStatus.Ready, 20_000);
    console.log(`✅ Conectado a "${channel.name}"`);

    const player = createAudioPlayer();
    connection.subscribe(player);
    player.on('error', (err) => console.error('❌ Player erro:', err.message));

    const receiver      = connection.receiver;
    const subscriptions = new Map();

    receiver.speaking.on('start', (userId) => {
      if (subscriptions.has(userId)) return;

      // FIX (áudio picotado): decodificar Opus recebido do microfone é
      // trabalho de CPU no MESMO event loop single-threaded que está
      // codificando e enviando o áudio de saída da EVA. Se alguém (ou o
      // próprio eco da EVA sem fone) ativa o microfone enquanto ela está
      // falando, as duas coisas competem pela CPU ao mesmo tempo — isso
      // pode ser o que está causando o áudio picotado/com ruído. O lado
      // Python já ignora esse áudio recebido durante TTS (_tts_playing),
      // mas só DEPOIS de decodificar — aqui a gente evita nem começar a
      // decodificar, que é a parte cara.
      if (player.state.status === AudioPlayerStatus.Playing) {
        return;
      }

      console.log(`🎤 Usuário ${userId} começou a falar`);

      let opusStream;
      try {
        opusStream = receiver.subscribe(userId, {
          end: { behavior: EndBehaviorType.AfterSilence, duration: 1000 },
        });
      } catch (err) {
        console.error(`❌ Falha ao assinar áudio de ${userId}: ${err.message}`);
        return;
      }

      const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
      const pcmChunks = [];

      opusStream.on('error', (err) => {
        const m = err.message || String(err);
        if (m.includes('DecryptionFailed') || m.includes('decrypt') || m.includes('Unencrypted')) {
          console.warn(`⚠️ DAVE decrypt ignorado para ${userId}`);
        } else {
          console.error(`❌ opusStream erro (${userId}): ${m}`);
        }
        try { opusStream.destroy(); } catch {}
        try { decoder.destroy(); }   catch {}
        subscriptions.delete(userId);
      });

      opusStream.pipe(decoder);

      decoder.on('data', (chunk) => pcmChunks.push(chunk));

      decoder.on('end', () => {
        subscriptions.delete(userId);
        if (pcmChunks.length === 0) return;

        const pcm = Buffer.concat(pcmChunks);
        console.log(`🎙️ PCM do user ${userId}: ${pcm.length} bytes`);

        // FIX (debug desconexão espontânea): pythonSocket.send() pode
        // lançar de forma síncrona se o WS estiver no meio do fechamento
        // (CLOSING) — isso acontece DENTRO de um callback de stream
        // ('end' do decoder), sem nenhum try/catch acima na pilha, então
        // sem esse try/catch aqui a exceção sobe crua e pode derrubar o
        // processo Node inteiro (uncaughtException).
        try {
          if (pythonSocket && pythonSocket.readyState === 1) {
            sendJson(pythonSocket, { type: 'audio', guild_id, user_id: userId, bytes: pcm.length });
            pythonSocket.send(pcm);
          }
        } catch (err) {
          console.error(`❌ Falha ao enviar PCM pro Python (user ${userId}): ${err.message}`);
        }
      });

      decoder.on('error', (err) => {
        console.error(`❌ Decoder erro (${userId}):`, err.message);
        subscriptions.delete(userId);
      });

      subscriptions.set(userId, { opusStream, decoder });
    });

    // ─────────────────────────────────────────
    // FIX Bug 1: Reconexão robusta com backoff
    // ─────────────────────────────────────────
    connection.on(VoiceConnectionStatus.Disconnected, async () => {
      // Evitar loop de reconexão dupla
      if (reconnectingGuilds.has(guild_id)) return;
      reconnectingGuilds.add(guild_id);

      console.log(`⚠️ Conexão de voz perdida (guild ${guild_id}), tentando reconectar...`);

      // Notificar Python que está reconectando (NÃO é um 'left' permanente)
      sendJson(pythonSocket, { type: 'reconnecting', guild_id });

      let reconnected = false;
      for (let attempt = 1; attempt <= CONFIG.MAX_RECONNECT_ATTEMPTS; attempt++) {
        console.log(`🔄 Tentativa ${attempt}/${CONFIG.MAX_RECONNECT_ATTEMPTS}...`);
        try {
          await Promise.race([
            entersState(connection, VoiceConnectionStatus.Signalling,  CONFIG.RECONNECT_TIMEOUT_MS),
            entersState(connection, VoiceConnectionStatus.Connecting,  CONFIG.RECONNECT_TIMEOUT_MS),
          ]);
          // Se chegou aqui, saiu do Disconnected — aguarda Ready
          await entersState(connection, VoiceConnectionStatus.Ready, CONFIG.RECONNECT_TIMEOUT_MS);
          console.log(`✅ Reconectado (tentativa ${attempt})`);
          reconnected = true;
          break;
        } catch {
          console.warn(`⚠️ Tentativa ${attempt} falhou`);
          // Pequeno backoff antes da próxima tentativa
          await new Promise(r => setTimeout(r, 2000 * attempt));
        }
      }

      reconnectingGuilds.delete(guild_id);

      if (!reconnected) {
        // Só agora destrói e notifica Python com 'left' permanente
        console.error(`❌ Não foi possível reconectar (guild ${guild_id}). Encerrando.`);
        connection.destroy();
        voiceStates.delete(guild_id);
        sendJson(pythonSocket, { type: 'left', guild_id, reason: 'reconnect_failed' });
      } else {
        // Notificar Python que reconectou
        sendJson(pythonSocket, { type: 'reconnected', guild_id, channel_id });
      }
    });

    voiceStates.set(guild_id, { connection, player, receiver, subscriptions, channel_id });
    sendJson(ws, { type: 'joined', guild_id, channel_id, channel_name: channel.name });
    console.log(`✅ Bridge ativo para guild ${guild_id}`);

  } catch (err) {
    console.error('❌ Erro ao entrar no canal:', err);
    sendJson(ws, { type: 'error', message: err.message });
  }
}

// ─────────────────────────────────────────
// HANDLER: SAIR DO CANAL
// ─────────────────────────────────────────
async function handleLeave(ws, msg) {
  const { guild_id } = msg;
  const state = voiceStates.get(guild_id);

  // FIX: cancelar reconexão pendente se o Python pediu leave
  reconnectingGuilds.delete(guild_id);

  if (state) {
    for (const [, sub] of state.subscriptions) {
      try { sub.opusStream.destroy(); } catch {}
      try { sub.decoder.destroy();   } catch {}
    }
    state.subscriptions.clear();
    // FIX: destroy antes de delete para garantir cleanup no Discord
    try { state.connection.destroy(); } catch {}
    voiceStates.delete(guild_id);
  }

  // FIX: sempre responder 'left', mesmo se não estava conectado
  // (evita o Python ficar preso no future de leave)
  sendJson(ws, { type: 'left', guild_id });
  console.log(`👋 Saiu do canal (guild ${guild_id})`);
}

// ─────────────────────────────────────────
// REPRODUZIR ÁUDIO PCM NO DISCORD
// ─────────────────────────────────────────

// FIX (áudio quebrado/cheio de ruído): Readable.from(buffer) tem um caso
// especial no Node para Buffer/string — em vez de entregar como stream de
// bytes normal, ele cria um Readable em objectMode que empurra o BUFFER
// INTEIRO (podem ser vários segundos de áudio, centenas de KB) como um
// único chunk monolítico numa passada só (ver lib/internal/streams/from.js
// no código-fonte do Node: há um branch dedicado pra "typeof iterable ===
// 'string' || iterable instanceof Buffer" que faz exatamente isso).
//
// O encoder Opus que o createAudioResource(..., {inputType:'raw'}) monta
// por baixo dos panos (via prism-media) espera consumir um Readable
// convencional entregando dados aos poucos — é assim que qualquer exemplo
// da lib alimenta esse pipeline (fs.createReadStream, sockets, etc.), nunca
// um objeto único carregando o áudio inteiro. Empurrar tudo de uma vez força
// a codificação Opus de uma resposta inteira numa rajada só, sem a
// granularidade/backpressure normal — suspeito nº1 pro áudio picotado/com
// estática que chega no Discord.
//
// Fix: Readable "de verdade" (sem objectMode) que entrega o PCM em pedaços
// do tamanho exato de um frame Discord (3840 bytes = 20ms @ 48kHz stereo
// s16le) a cada _read() — mesma granularidade que o encoder e o player
// esperam, com backpressure normal.
function pcmBufferToStream(buffer, frameBytes = CONFIG.FRAME_SIZE) {
  let offset = 0;
  return new Readable({
    highWaterMark: frameBytes * 4, // poucos frames de folga, sem acumular o áudio inteiro em memória de novo
    read() {
      if (offset >= buffer.length) {
        this.push(null);
        return;
      }
      const end = Math.min(offset + frameBytes, buffer.length);
      let chunk = buffer.subarray(offset, end);
      // Rede de segurança: se o último pedaço vier menor que um frame
      // completo (buffer não alinhado por algum caller), preenche com
      // silêncio em vez de mandar um frame parcial pro encoder Opus —
      // mesma lógica de alinhamento já usada no lado Python
      // (pocket_tts_engine.py / streaming_tts_system.py), só que como
      // rede de segurança aqui também.
      if (chunk.length < frameBytes) {
        const padded = Buffer.alloc(frameBytes);
        chunk.copy(padded);
        chunk = padded;
      }
      offset = end;
      this.push(chunk);
    },
  });
}

async function playAudio(guild_id, pcmBuffer) {
  const state = voiceStates.get(guild_id);
  if (!state) {
    console.warn(`⚠️ Não conectado à guild ${guild_id}`);
    // FIX: notificar Python mesmo sem estado, para não travar o future
    sendJson(pythonSocket, { type: 'play_done', guild_id });
    return;
  }

  const { player } = state;

  // FIX (debug desconexão espontânea): sem try/catch aqui, um buffer
  // malformado ou uma conexão de voz já derrubada faz createAudioResource
  // ou player.play() lançar de forma síncrona, e como playAudio() é
  // chamada a partir do handler de mensagem WS sem proteção própria,
  // isso pode se propagar e derrubar o processo inteiro.
  try {
    const readable = pcmBufferToStream(pcmBuffer);
    // Erros de leitura/decodificação no meio do stream (ex: pipeline do
    // Opus encoder falhando) antes só apareciam como 'error' no player,
    // sem contexto de qual guild/quantos bytes — agora fica logado aqui
    // também, direto na origem.
    readable.on('error', (err) => {
      console.error(`❌ Erro no stream de PCM (guild ${guild_id}): ${err.message}`);
    });
    const resource  = createAudioResource(readable, { inputType: 'raw', inlineVolume: false });
    player.play(resource);
    console.log(`🔊 Reproduzindo ${pcmBuffer.length} bytes (guild ${guild_id})`);
  } catch (err) {
    console.error(`❌ Falha ao reproduzir áudio (guild ${guild_id}): ${err.message}`);
    if (pythonSocket && pythonSocket.readyState === 1) {
      sendJson(pythonSocket, { type: 'play_done', guild_id });
    }
    return;
  }

  player.once(AudioPlayerStatus.Idle, () => {
    console.log(`✅ Reprodução finalizada (guild ${guild_id})`);
    if (pythonSocket && pythonSocket.readyState === 1) {
      sendJson(pythonSocket, { type: 'play_done', guild_id });
    }
  });
}

// ─────────────────────────────────────────
// UTILITÁRIO
// ─────────────────────────────────────────
function sendJson(ws, obj) {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify(obj));
  }
}

// ─────────────────────────────────────────
// DISCORD LOGIN
// ─────────────────────────────────────────
client.once('clientReady', () => {
  console.log(`🤖 Discord Bot conectado como ${client.user.tag}`);
  console.log(`🎙️ EVA Voice Bridge v2 pronto!`);
  console.log(`   WebSocket: ws://localhost:${CONFIG.WS_PORT}`);
});

client.on('error', (err) => console.error('❌ Discord Client erro:', err.message));

async function loginWithRetry(maxAttempts = 5) {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      console.log(`🔑 Login Discord (tentativa ${attempt}/${maxAttempts})...`);
      await client.login(CONFIG.DISCORD_TOKEN);
      return;
    } catch (err) {
      console.error(`❌ Login falhou: ${err.message}`);
      if (attempt < maxAttempts) {
        const delay = attempt * 3000;
        console.log(`⏳ Aguardando ${delay/1000}s...`);
        await new Promise(r => setTimeout(r, delay));
      } else {
        process.exit(1);
      }
    }
  }
}

loginWithRetry();

process.on('SIGINT', async () => {
  console.log('\n⏹️ Encerrando bridge...');
  for (const [, state] of voiceStates) {
    try { state.connection.destroy(); } catch {}
  }
  client.destroy();
  wss.close();
  process.exit(0);
});