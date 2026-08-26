/**
 * EVA Bridge - Node.js  (v3)
 * ==========================
 * Ponte entre o Discord e o processo Python. Fala WebSocket na porta 8765:
 * JSON para controle, quadros binários para PCM.
 *
 * O QUE MUDOU NA v3
 * -----------------
 * Adicionado o caminho de TEXTO, que não existia. O bridge_client.py do lado
 * Python já esperava por ele -- `_ao_receber_mensagem`, `enviar_mensagem` e
 * `typing` não tinham contraparte aqui, então texto simplesmente não chegava.
 *
 *   novo evento  message        bridge -> Python
 *   novo comando send_message   Python -> bridge
 *   novo comando typing         Python -> bridge
 *
 * E dois bugs que impediriam o texto de funcionar mesmo com o handler:
 *
 *   Faltava a intent MessageContent. Sem ela o Discord entrega
 *   `message.content` como string vazia -- o bot recebe o evento, e o
 *   conteúdo chega em branco. É um erro silencioso: nada falha, a EVA só
 *   responde a mensagens vazias para sempre.
 *
 *   Faltavam DirectMessages e os Partials. DM nunca dispara messageCreate
 *   sem eles, porque o canal não está em cache.
 *
 * TODAS as correções da v2 estão preservadas, com os comentários originais.
 * Elas vieram de bug real em produção; cada uma corrige algo audível.
 */

// ─────────────────────────────────────────
// FIX v2 (debug desconexão espontânea): sem isso, uma exceção não tratada
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

process.on('unhandledRejection', (reason) => {
  console.error('💥 [FATAL] unhandledRejection:', reason);
  if (reason instanceof Error) console.error(reason.stack);
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

const { Client, GatewayIntentBits, Partials, ChannelType } = require('discord.js');
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
  // FIX v2: timeout de reconexão aumentado de 5s para 15s
  RECONNECT_TIMEOUT_MS: 15_000,
  // FIX v2: tentativas de reconexão antes de desistir
  MAX_RECONNECT_ATTEMPTS: 3,
  // Limite duro do Discord por mensagem. Passar disso devolve 400.
  MAX_MSG_CHARS: 2000,
  // FIX (latência de voz): antes esse número estava fixo em 1000 aqui
  // embaixo, direto no EndBehaviorType.AfterSilence, sem nenhuma ligação
  // com EVA_VOZ_SILENCIO em config.py -- mudar o .env não fazia nada.
  // Agora os dois lados leem a MESMA variável (config.py em segundos,
  // aqui convertida pra ms). Esse número é um piso de latência que se
  // soma a QUALQUER backend de STT: a captura só termina depois desse
  // tanto de silêncio, então baixar ele afeta todo turno de voz. Abaixo
  // de ~500ms corre risco de cortar fala com pausa natural no meio --
  // meça numa call real.
  SILENCE_DURATION_MS: Math.round(parseFloat(process.env.EVA_VOZ_SILENCIO || '0.7') * 1000),
};

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.GuildMessages,
    // NOVO v3 — sem MessageContent o `content` chega vazio. É intent
    // privilegiada: ligue também no Developer Portal, em Bot > Privileged
    // Gateway Intents > Message Content Intent. Se não ligar lá, o login
    // falha com "Used disallowed intents".
    GatewayIntentBits.MessageContent,
    // NOVO v3 — sem isso, DM nunca dispara messageCreate.
    GatewayIntentBits.DirectMessages,
  ],
  // NOVO v3 — DM chega com o canal fora do cache; sem os partials o
  // discord.js descarta o evento antes de emitir.
  partials: [Partials.Channel, Partials.Message],
  rest: { timeout: 30000, retries: 3 },
});

const voiceStates = new Map();
let pythonSocket = null;
let pendingAudioFor = null;

// FIX v2: rastrear se uma guild está em processo de reconexão
const reconnectingGuilds = new Set();

console.log(`🔌 WebSocket Server escutando na porta ${CONFIG.WS_PORT}`);
console.log(`🎤 Timeout de silêncio (fim de fala): ${CONFIG.SILENCE_DURATION_MS}ms (EVA_VOZ_SILENCIO)`);

const wss = new WebSocketServer({ port: CONFIG.WS_PORT });

wss.on('connection', (ws) => {
  console.log('🐍 Python conectou ao bridge');
  pythonSocket = ws;

  sendJson(ws, { type: 'ready', version: '3.0' });

  ws.on('message', async (data, isBinary) => {
    if (isBinary) {
      if (pendingAudioFor) {
        const { guild_id, streaming } = pendingAudioFor;
        pendingAudioFor = null;
        if (streaming) {
          pushPlayChunk(guild_id, data);
        } else {
          await playAudio(guild_id, data);
        }
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
        pendingAudioFor = { guild_id: msg.guild_id, streaming: false };
        break;
      case 'play_start':
        startPlayStream(msg.guild_id);
        break;
      case 'play_chunk':
        pendingAudioFor = { guild_id: msg.guild_id, streaming: true };
        break;
      case 'play_end':
        endPlayStream(msg.guild_id);
        break;
      case 'stop_play':
        stopPlay(msg.guild_id);
        break;
      case 'fonte_de_eco':
        marcarFonteDeEco(msg.guild_id, msg.user_id, msg.ttl_ms);
        break;
      // NOVO v3
      case 'send_message': await handleSendMessage(ws, msg); break;
      case 'typing':       await handleTyping(msg);          break;
      default:
        console.warn(`⚠️ Tipo desconhecido: ${msg.type}`);
    }
  });

  // FIX v2 (Bug 2): quando Python desconecta, limpar estado de voz
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
// NOVO v3 — TEXTO: DISCORD → PYTHON
// ─────────────────────────────────────────
client.on('messageCreate', async (message) => {
  // Ignorar bots, inclusive a própria EVA. Sem isso ela responde à própria
  // resposta e o loop só para quando o rate limit do Discord corta.
  if (message.author?.bot) return;

  // Partial: em DM a mensagem pode chegar incompleta. Buscar antes de ler.
  if (message.partial) {
    try { await message.fetch(); } catch { return; }
  }

  if (!pythonSocket || pythonSocket.readyState !== 1) return;

  const isDM = message.channel?.type === ChannelType.DM;
  const mentioned = client.user ? message.mentions.users.has(client.user.id) : false;

  // Tira a menção do texto. Sem isso a EVA recebe "<@1234567890> oi" e o
  // id cru entra no contexto do modelo -- às vezes ele repete o número.
  let content = (message.content || '').trim();
  if (client.user) {
    content = content
      .replace(new RegExp(`<@!?${client.user.id}>`, 'g'), '')
      .trim();
  }

  // O canal de voz de quem falou. Deixa `!eva entra` funcionar sem argumento.
  const voiceChannelId = message.member?.voice?.channelId || null;

  const attachments = [...(message.attachments?.values() || [])].map((a) => ({
    url: a.url,
    name: a.name,
    content_type: a.contentType || '',
    size: a.size,
  }));

  sendJson(pythonSocket, {
    type: 'message',
    message_id: message.id,
    channel_id: message.channelId,
    guild_id: message.guildId || null,
    author_id: message.author.id,
    author_name: message.member?.displayName || message.author.username,
    content,
    is_dm: isDM,
    mentioned,
    voice_channel_id: voiceChannelId,
    attachments,
  });
});

// ─────────────────────────────────────────
// NOVO v3 — TEXTO: PYTHON → DISCORD
// ─────────────────────────────────────────
async function handleSendMessage(ws, msg) {
  const { channel_id, content, reply_to } = msg;
  if (!content) return;

  try {
    const channel = await client.channels.fetch(channel_id);
    if (!channel || !channel.isTextBased()) {
      sendJson(ws, { type: 'error', message: `canal ${channel_id} não é de texto` });
      return;
    }

    // Discord corta em 2000 caracteres e devolve 400. As respostas da EVA
    // ficam bem abaixo disso (p99 do dataset é 222 chars), mas erro de
    // modelo e saída de diagnóstico podem estourar -- e um 400 aqui vira
    // silêncio, que é pior do que uma mensagem partida.
    const partes = dividirTexto(content, CONFIG.MAX_MSG_CHARS);

    let enviada = null;
    for (let i = 0; i < partes.length; i++) {
      const payload = { content: partes[i] };
      // Só a primeira parte responde à mensagem original; as seguintes
      // vão soltas, senão viram uma pilha de respostas à mesma mensagem.
      if (i === 0 && reply_to) {
        payload.reply = { messageReference: reply_to, failIfNotExists: false };
      }
      enviada = await channel.send(payload);
    }

    sendJson(ws, {
      type: 'message_sent',
      channel_id,
      message_id: enviada ? enviada.id : null,
      partes: partes.length,
    });
  } catch (err) {
    console.error(`❌ Falha ao enviar mensagem (canal ${channel_id}): ${err.message}`);
    sendJson(ws, { type: 'error', message: err.message, channel_id });
  }
}

async function handleTyping(msg) {
  // Melhor esforço: o indicador de digitação é cosmético, e falhar nele
  // nunca deve impedir a resposta de sair.
  try {
    const channel = await client.channels.fetch(msg.channel_id);
    if (channel && channel.isTextBased()) await channel.sendTyping();
  } catch (err) {
    console.warn(`⚠️ typing falhou (canal ${msg.channel_id}): ${err.message}`);
  }
}

/** Divide respeitando quebra de linha e depois espaço, para não cortar palavra. */
function dividirTexto(texto, limite) {
  if (texto.length <= limite) return [texto];
  const partes = [];
  let resto = texto;
  while (resto.length > limite) {
    let corte = resto.lastIndexOf('\n', limite);
    if (corte < limite * 0.5) corte = resto.lastIndexOf(' ', limite);
    if (corte < limite * 0.5) corte = limite;
    partes.push(resto.slice(0, corte).trim());
    resto = resto.slice(corte).trim();
  }
  if (resto) partes.push(resto);
  return partes;
}

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

    // Criado ANTES dos listeners de propósito: `speaking.on('start')` e o
    // `decoder.on('end')` de cada usuário precisam consultar o estado
    // (fontesDeEco, corteSolicitado) e antes eles rodavam com o objeto
    // ainda não existente no Map.
    const state = {
      connection,
      player,
      receiver,
      subscriptions,
      channel_id,
      streamAtivo: null,
      filaAudio: Buffer.alloc(0),
      streamFinalizando: false,
      timerStream: null,
      // --- interrupção (barge-in) ---
      // user_id -> timestamp até quando esse usuário é tratado como
      // fonte de eco (caixa de som). Preenchido pelo Python via
      // `fonte_de_eco`; só tem efeito enquanto a EVA está falando.
      fontesDeEco: new Map(),
      corteSolicitado: false,
      corteAplicado: false,
      bytesReproduzidos: 0,
      bytesTotais: 0,
    };

    receiver.speaking.on('start', (userId) => {
      if (subscriptions.has(userId)) return;

      const tocando = player.state.status === AudioPlayerStatus.Playing;

      // HISTÓRICO — FIX v2 (áudio picotado): decodificar Opus recebido do
      // microfone é trabalho de CPU no MESMO event loop single-threaded
      // que está codificando e enviando o áudio de saída da EVA. A versão
      // anterior resolvia isso com `if (tocando) return;` — nem começava
      // a decodificar enquanto ela falava.
      //
      // POR QUE MUDOU: aquele `return` tornava a INTERRUPÇÃO impossível
      // por construção. O áudio de quem falava por cima dela nunca saía
      // do Node, então não existia nada do lado Python que pudesse
      // recuperar aquela fala — a pessoa respondia, e a resposta sumia.
      //
      // O custo de CPU continua real, então ele foi trocado por um gate
      // mais preciso em vez de eliminado: o Python identifica quem está
      // em CAIXA DE SOM (a transcrição volta sendo a própria fala da EVA)
      // e manda `fonte_de_eco` marcando aquele usuário. Enquanto a marca
      // vale, esse usuário específico não é decodificado durante o
      // playback — que era 100% do caso que o fix v2 queria evitar. Quem
      // está de fone continua podendo interromper.
      if (tocando) {
        const expira = state.fontesDeEco.get(userId);
        if (expira && expira > Date.now()) {
          return;
        }
      }

      console.log(`🎤 Usuário ${userId} começou a falar${tocando ? ' (por cima da EVA)' : ''}`);

      let opusStream;
      try {
        opusStream = receiver.subscribe(userId, {
          end: { behavior: EndBehaviorType.AfterSilence, duration: CONFIG.SILENCE_DURATION_MS },
        });
      } catch (err) {
        console.error(`❌ Falha ao assinar áudio de ${userId}: ${err.message}`);
        return;
      }

      // NOVO v3.1 — `new prism.opus.Decoder(...)` faz um require() interno
      // de @discordjs/opus / node-opus / opusscript (nessa ordem), e nenhum
      // deles vem instalado por padrão com o resto do stack de voz — não é
      // dependência transitiva do @discordjs/voice, é "instale um dos três
      // você mesmo". Sem esse try/catch aqui, esse require() faltando
      // lançava DENTRO do listener 'start' e virava um uncaughtException
      // GLOBAL a cada vez que alguém falava — matando o processo Node
      // inteiro (e junto o cliente Python, que via o socket cair e entrava
      // em loop de reconexão). Um usuário sem o pacote instalado corretamente
      // não deveria conseguir derrubar a call inteira; agora falha só a
      // captura desse usuário, uma vez, com mensagem que diz o que instalar.
      let decoder;
      try {
        decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
      } catch (err) {
        console.error(
          `❌ Decoder Opus indisponível (${err.message}). ` +
          `Rode: npm install @discordjs/opus`
        );
        try { opusStream.destroy(); } catch {}
        return;
      }
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

        // FIX v2: pythonSocket.send() pode lançar de forma síncrona se o WS
        // estiver no meio do fechamento (CLOSING) — isso acontece DENTRO de
        // um callback de stream ('end' do decoder), sem nenhum try/catch
        // acima na pilha, então sem esse try/catch aqui a exceção sobe crua
        // e pode derrubar o processo Node inteiro.
        try {
          if (pythonSocket && pythonSocket.readyState === 1) {
            // `durante_fala` é medido no INÍCIO da captura, não agora: o
            // decoder só fecha depois do silêncio (EVA_VOZ_SILENCIO), e
            // nesse intervalo a EVA já pode ter terminado de falar. O
            // Python usa isso pra decidir se roda a checagem de eco --
            // com o valor de agora, todo eco capturado no fim da fala
            // dela passaria batido como fala legítima.
            sendJson(pythonSocket, {
              type: 'audio', guild_id, user_id: userId, bytes: pcm.length,
              durante_fala: tocando,
            });
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
    // FIX v2 (Bug 1): Reconexão robusta com backoff
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
        sendJson(pythonSocket, { type: 'reconnected', guild_id, channel_id });
      }
    });

    voiceStates.set(guild_id, state);
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

  // FIX v2: cancelar reconexão pendente se o Python pediu leave
  reconnectingGuilds.delete(guild_id);

  if (state) {
    pararTimerStream(state);
    for (const [, sub] of state.subscriptions) {
      try { sub.opusStream.destroy(); } catch {}
      try { sub.decoder.destroy();   } catch {}
    }
    state.subscriptions.clear();
    if (state.streamAtivo) {
      try { state.streamAtivo.push(null); } catch {}
    }
    // FIX v2: destroy antes de delete para garantir cleanup no Discord
    try { state.connection.destroy(); } catch {}
    voiceStates.delete(guild_id);
  }

  // FIX v2: sempre responder 'left', mesmo se não estava conectado
  // (evita o Python ficar preso no future de leave)
  sendJson(ws, { type: 'left', guild_id });
  console.log(`👋 Saiu do canal (guild ${guild_id})`);
}

// ─────────────────────────────────────────
// REPRODUZIR ÁUDIO PCM NO DISCORD
// ─────────────────────────────────────────

// FIX v2 (áudio quebrado/cheio de ruído): Readable.from(buffer) tem um caso
// especial no Node para Buffer/string — em vez de entregar como stream de
// bytes normal, ele cria um Readable em objectMode que empurra o BUFFER
// INTEIRO (podem ser vários segundos de áudio, centenas de KB) como um
// único chunk monolítico numa passada só (ver lib/internal/streams/from.js
// no código-fonte do Node).
//
// O encoder Opus que o createAudioResource(..., {inputType:'raw'}) monta
// por baixo dos panos espera consumir um Readable convencional entregando
// dados aos poucos. Empurrar tudo de uma vez força a codificação Opus de
// uma resposta inteira numa rajada só, sem a granularidade/backpressure
// normal — causa do áudio picotado/com estática que chegava no Discord.
//
// Fix: Readable "de verdade" (sem objectMode) que entrega o PCM em pedaços
// do tamanho exato de um frame Discord (3840 bytes = 20ms @ 48kHz stereo
// s16le) a cada _read().
function pcmBufferToStream(buffer, state, frameBytes = CONFIG.FRAME_SIZE) {
  let offset = 0;
  return new Readable({
    highWaterMark: frameBytes * 4, // poucos frames de folga, sem acumular o áudio inteiro em memória de novo
    read() {
      // INTERRUPÇÃO: o Python pediu `stop_play` no meio da reprodução.
      // Não dá pra "terminar a frase atual" aqui — neste caminho o clipe
      // chega pronto, num buffer só, sem nenhuma marca de onde uma frase
      // acaba e outra começa (quem sabe isso é o TTS, do outro lado).
      // Então o corte é no próximo frame, com fade de saída pra não virar
      // clique. Corte em fronteira de frase de verdade existe só no
      // caminho de streaming (EVA_VOZ_STREAMING=1), onde cada frase é uma
      // unidade separada — ver startPlayStream.
      if (state && state.corteSolicitado) {
        if (state.corteAplicado || offset >= buffer.length) {
          this.push(null);
          return;
        }
        state.corteAplicado = true;
        const fim = Math.min(offset + frameBytes, buffer.length);
        let ultimo = buffer.subarray(offset, fim);
        if (ultimo.length < frameBytes) {
          const padded = Buffer.alloc(frameBytes);
          ultimo.copy(padded);
          ultimo = padded;
        }
        offset = fim;
        state.bytesReproduzidos = offset;
        this.push(aplicarFade(ultimo, 'saida'));
        return;
      }

      if (offset >= buffer.length) {
        this.push(null);
        return;
      }
      const end = Math.min(offset + frameBytes, buffer.length);
      let chunk = buffer.subarray(offset, end);
      // Rede de segurança: se o último pedaço vier menor que um frame
      // completo, preenche com silêncio em vez de mandar frame parcial pro
      // encoder Opus — mesma lógica de alinhamento do lado Python.
      if (chunk.length < frameBytes) {
        const padded = Buffer.alloc(frameBytes);
        chunk.copy(padded);
        chunk = padded;
      }
      offset = end;
      if (state) state.bytesReproduzidos = offset;
      this.push(chunk);
    },
  });
}

function pararTimerStream(state) {
  if (state && state.timerStream) {
    clearInterval(state.timerStream);
    state.timerStream = null;
  }
}

// FIX (estalo/"pipocando" na transição real<->silêncio): PCM cru pula
// abruptamente de amplitude real pra zero (ou zero pra real) toda vez
// que um frame de silêncio de preenchimento entra ou sai -- isso é uma
// transiente de alta frequência, ouve-se como clique/estalo. Fade ao
// longo do próprio frame de 20ms suaviza a transição sem ser percebido
// como um "fade" de verdade -- só elimina o degrau abrupto.
function aplicarFade(frame, direcao) {
  const saida = Buffer.from(frame);
  const totalAmostras = saida.length / 2; // int16 = 2 bytes cada
  for (let i = 0; i < totalAmostras; i++) {
    const offset = i * 2;
    const fator = direcao === 'saida' ? 1 - (i / totalAmostras) : (i / totalAmostras);
    const valor = saida.readInt16LE(offset);
    saida.writeInt16LE(Math.round(valor * fator), offset);
  }
  return saida;
}

async function playAudio(guild_id, pcmBuffer) {
  const state = voiceStates.get(guild_id);
  if (!state) {
    console.warn(`⚠️ Não conectado à guild ${guild_id}`);
    // FIX v2: notificar Python mesmo sem estado, para não travar o future
    sendJson(pythonSocket, { type: 'play_done', guild_id });
    return;
  }

  const { player } = state;

  // FIX v2: sem try/catch aqui, um buffer malformado ou uma conexão de voz
  // já derrubada faz createAudioResource ou player.play() lançar de forma
  // síncrona, e como playAudio() é chamada a partir do handler de mensagem
  // WS sem proteção própria, isso derruba o processo inteiro.
  // Cada reprodução começa sem corte pendente -- sem isso, um `stop_play`
  // que chegou tarde (depois do fim da fala anterior) mataria a fala
  // SEGUINTE logo no primeiro frame.
  state.corteSolicitado = false;
  state.corteAplicado = false;
  state.bytesReproduzidos = 0;
  state.bytesTotais = pcmBuffer.length;

  try {
    const readable = pcmBufferToStream(pcmBuffer, state);
    readable.on('error', (err) => {
      console.error(`❌ Erro no stream de PCM (guild ${guild_id}): ${err.message}`);
      pararTimerStream(state);
    });
    const resource = createAudioResource(readable, { inputType: 'raw', inlineVolume: false });
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
    pararTimerStream(state);
    const cortada = state.corteSolicitado;
    console.log(`✅ Reprodução finalizada (guild ${guild_id})`
      + (cortada ? ` — CORTADA em ${state.bytesReproduzidos}/${state.bytesTotais} bytes` : ''));
    if (pythonSocket && pythonSocket.readyState === 1) {
      // `bytes_reproduzidos` / `bytes_totais`: é com isso que o Python
      // corrige o histórico depois de uma interrupção -- grava só o
      // trecho que virou som, não a resposta inteira que ele tinha
      // gerado. Aproximado por natureza (o encoder Opus mantém alguns
      // frames em buffer), erra pra mais no máximo ~80ms.
      sendJson(pythonSocket, {
        type: 'play_done', guild_id, cortada,
        bytes_reproduzidos: state.bytesReproduzidos,
        bytes_totais: state.bytesTotais,
      });
    }
  });
}

// ─────────────────────────────────────────
// REPRODUÇÃO EM STREAMING (frase a frase, conforme a EVA sintetiza)
// ─────────────────────────────────────────
//
// TERCEIRA tentativa de streaming -- as duas anteriores foram revertidas
// (ver VozConfig.voz_streaming em config.py pro histórico completo). As
// duas primeiras suspeitavam do modelo de TTS ou da GPU; depois de trocar
// os dois e o áudio CONTINUAR ruim, ficou claro que o problema nunca foi
// síntese -- era este arquivo.
//
// A versão anterior usava `readable.push(frame)` dentro de um
// `setInterval(..., 20)`: um timer nosso EMPURRANDO dados pro stream a
// cada 20ms, por fora do controle de backpressure do Node. Isso ignora de
// propósito o motivo de Readable existir -- `push()` fora de `_read()`
// não espera o consumidor (o encoder Opus por trás de createAudioResource)
// estar pronto pra mais dados, e um setInterval no MESMO event loop que
// também faz I/O de rede (WebSocket recebendo pedaço de áudio do Python)
// não bate 20ms exato -- é só uma aproximação que degrada sob carga,
// exatamente o tipo de jitter que causa áudio picotado/com estática.
//
// O caminho bloqueante (`pcmBufferToStream` acima) NUNCA teve esse
// problema, e a diferença agora está clara: ele implementa `read()`,
// puxado PELO consumidor quando ele quer mais dado -- backpressure
// automática, sem timer nenhum. @discordjs/voice já pausa/retoma a
// leitura no ritmo certo de frame Opus (20ms) por dentro; não tinha
// necessidade de reimplementar esse relógio aqui.
//
// Fix: mesmo padrão pull-based do caminho bloqueante, só que a fila
// (`state.filaAudio`) recebe dados aos poucos via `pushPlayChunk` em vez
// de vir pronta de uma vez -- e quando `_read()` é chamado e não tem
// frame completo ainda (TTS mais lento que playback), preenche com
// silêncio (com fade, mesma lógica de antes) em vez de travar o stream.
function startPlayStream(guild_id) {
  const state = voiceStates.get(guild_id);
  if (!state) {
    sendJson(pythonSocket, { type: 'play_done', guild_id });
    return;
  }
  pararTimerStream(state); // no-op agora (não há mais timer), mantido por segurança
  if (state.streamAtivo) {
    try { state.streamAtivo.push(null); } catch {}
  }

  const SILENCIO = Buffer.alloc(CONFIG.FRAME_SIZE);

  // DIAGNÓSTICO TEMPORÁRIO: contadores pra achar em qual das três
  // hipóteses o áudio está se perdendo -- (1) pushPlayChunk descartando
  // por streamAtivo falsy, (2) _read() nunca achando frame real (fila
  // sempre vazia -- correlação binário/JSON quebrada), (3) tudo chegando
  // mas MAIORIA dos frames sendo silêncio (timing/backpressure). Remover
  // depois de identificar a causa real.
  state.statsStream = {
    bytesRecebidos: 0,
    chunksRecebidos: 0,
    chunksDescartados: 0,
    framesReal: 0,
    framesSilencio: 0,
  };

  const readable = new Readable({
    highWaterMark: CONFIG.FRAME_SIZE * 4,
    read() {
      let frame;
      let tipoAtual;

      if (state.filaAudio.length >= CONFIG.FRAME_SIZE) {
        frame = state.filaAudio.subarray(0, CONFIG.FRAME_SIZE);
        state.filaAudio = state.filaAudio.subarray(CONFIG.FRAME_SIZE);
        tipoAtual = 'real';
        state.statsStream.framesReal++;
        state.bytesReproduzidos += CONFIG.FRAME_SIZE;
      } else if (state.streamFinalizando && state.filaAudio.length === 0) {
        console.log(`[voz-stream-diag] fim: ${state.statsStream.chunksRecebidos} chunks / `
          + `${state.statsStream.bytesRecebidos} bytes recebidos, `
          + `${state.statsStream.chunksDescartados} descartados, `
          + `frames real=${state.statsStream.framesReal} silencio=${state.statsStream.framesSilencio}`);
        this.push(null);
        return;
      } else {
        // Ainda não chegou um frame inteiro (síntese mais lenta que
        // playback neste instante) -- preenche com silêncio em vez de
        // travar `_read()` esperando; a fila continua sendo alimentada
        // por `pushPlayChunk` em paralelo.
        frame = SILENCIO;
        tipoAtual = 'silencio';
        state.statsStream.framesSilencio++;
      }

      if (tipoAtual === 'silencio' && state.ultimoTipoFrame === 'real') {
        frame = aplicarFade(state.ultimoFrameReal || frame, 'saida');
      } else if (tipoAtual === 'real' && state.ultimoTipoFrame === 'silencio') {
        frame = aplicarFade(frame, 'entrada');
      }
      if (tipoAtual === 'real') {
        state.ultimoFrameReal = frame;
      }
      state.ultimoTipoFrame = tipoAtual;

      this.push(frame);
    },
  });
  readable.on('error', (err) => {
    console.error(`❌ Erro no stream de voz (guild ${guild_id}): ${err.message}`);
  });

  state.streamAtivo = readable;
  state.filaAudio = Buffer.alloc(0);
  state.streamFinalizando = false;
  state.ultimoTipoFrame = null;
  state.ultimoFrameReal = null;
  state.corteSolicitado = false;
  state.corteAplicado = false;
  state.bytesReproduzidos = 0;
  state.bytesTotais = 0;

  const resource = createAudioResource(readable, { inputType: 'raw', inlineVolume: false });
  state.player.play(resource);
  console.log(`🔊 Streaming de voz iniciado (guild ${guild_id})`);

  state.player.once(AudioPlayerStatus.Idle, () => {
    state.streamAtivo = null;
    const cortada = state.corteSolicitado;
    console.log(`✅ Streaming de voz finalizado (guild ${guild_id})`
      + (cortada ? ' — CORTADO por interrupção' : ''));
    if (pythonSocket && pythonSocket.readyState === 1) {
      sendJson(pythonSocket, {
        type: 'play_done', guild_id, cortada,
        bytes_reproduzidos: state.bytesReproduzidos,
        bytes_totais: state.bytesReproduzidos + state.filaAudio.length,
      });
    }
  });
}

function pushPlayChunk(guild_id, chunk) {
  const state = voiceStates.get(guild_id);
  if (!state || !state.streamAtivo) {
    // DIAGNÓSTICO TEMPORÁRIO: se isto imprimir, é a causa -- chunk de
    // áudio chegou e foi descartado silenciosamente porque o Readable
    // ainda não existia (ou já tinha sido derrubado) neste momento.
    console.log(`[voz-stream-diag] chunk DESCARTADO (guild ${guild_id}): `
      + `state=${!!state} streamAtivo=${!!(state && state.streamAtivo)} `
      + `${chunk ? chunk.length : 0} bytes`);
    if (state && state.statsStream) state.statsStream.chunksDescartados++;
    return;
  }
  state.filaAudio = Buffer.concat([state.filaAudio || Buffer.alloc(0), chunk]);
  state.statsStream.bytesRecebidos += chunk.length;
  state.statsStream.chunksRecebidos++;
}

// ─────────────────────────────────────────
// INTERRUPÇÃO (barge-in)
// ─────────────────────────────────────────
//
// Chamado quando o Python confirmou que alguém falou POR CIMA da EVA e
// que aquilo não é eco da própria voz dela. Os dois caminhos de
// reprodução cortam de formas diferentes, de propósito:
//
//   streaming  -> `streamFinalizando = true`. O `_read()` continua
//                 entregando o que já está na fila e só então encerra.
//                 Como o Python para de enfileirar frases novas no mesmo
//                 instante, o que sobra na fila é o resto da frase atual
//                 — é o "termina a frase e cede", exato.
//
//   bloqueante -> `corteSolicitado = true`, e o `_read()` de
//                 pcmBufferToStream entrega mais um frame com fade e
//                 encerra. Aqui não existe fronteira de frase pra
//                 respeitar (o clipe chega inteiro, sem marcação), então
//                 o corte é imediato e suave.
//
// Na prática a diferença é menor do que parece: entre a pessoa começar a
// falar e este `stop_play` chegar passa o tempo de captura + STT, e nesse
// intervalo a EVA já falou mais um pedaço — o efeito percebido nos dois
// casos é "ela terminou o que estava dizendo e parou".
function stopPlay(guild_id) {
  const state = voiceStates.get(guild_id);
  if (!state) return;
  if (state.player.state.status !== AudioPlayerStatus.Playing) return;

  state.corteSolicitado = true;
  if (state.streamAtivo) {
    state.streamFinalizando = true;
    console.log(`✂️ Interrompida (guild ${guild_id}) — encerrando no fim da frase atual`);
  } else {
    console.log(`✂️ Interrompida (guild ${guild_id}) — corte com fade`);
  }
}

// O Python descobriu que este usuário está em caixa de som (a transcrição
// dele durante a fala da EVA veio sendo a própria fala da EVA). Enquanto
// a marca valer, o microfone dele não é nem decodificado durante o
// playback — é o que preserva a economia de CPU do fix v2 sem impedir
// que quem está de fone interrompa.
function marcarFonteDeEco(guild_id, user_id, ttl_ms) {
  const state = voiceStates.get(guild_id);
  if (!state || !user_id) return;
  const ttl = Number(ttl_ms) > 0 ? Number(ttl_ms) : 30000;
  state.fontesDeEco.set(String(user_id), Date.now() + ttl);
  console.log(`🔇 ${user_id} marcado como fonte de eco por ${Math.round(ttl / 1000)}s `
    + `(caixa de som) — não será decodificado enquanto a EVA falar`);
}

function endPlayStream(guild_id) {
  const state = voiceStates.get(guild_id);
  if (!state || !state.streamAtivo) return;
  // DIAGNÓSTICO TEMPORÁRIO: tamanho da fila no momento em que o Python
  // avisa que terminou de mandar tudo -- se isto for pequeno/zero apesar
  // de dezenas de play_chunk no log, o problema é chunk sendo perdido
  // ANTES de chegar aqui (pushPlayChunk descartando, ou correlação
  // JSON/binário errada no ws.on('message') lá em cima).
  console.log(`[voz-stream-diag] play_end recebido: filaAudio ainda tem `
    + `${state.filaAudio.length} bytes não consumidos pelo _read()`);
  state.streamFinalizando = true;
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
  console.log(`🎙️ EVA Bridge v3 pronto (voz + texto)`);
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
      // Erro típico de quem esqueceu de ligar a intent no portal. Sem esta
      // dica, a mensagem crua do discord.js não diz onde resolver.
      if (String(err.message).includes('disallowed intents')) {
        console.error(
          '   → Ligue "Message Content Intent" no Discord Developer Portal:\n' +
          '     Applications > seu app > Bot > Privileged Gateway Intents'
        );
      }
      if (attempt < maxAttempts) {
        const delay = attempt * 3000;
        console.log(`⏳ Aguardando ${delay / 1000}s...`);
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