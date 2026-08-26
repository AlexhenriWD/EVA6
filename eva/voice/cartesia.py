"""
Motor TTS via Cartesia (Sonic) -- usa o SDK oficial (pacote `cartesia`),
não chamadas HTTP/WebSocket cruas. Trocado da versão anterior (urllib +
websockets.sync) porque o SDK cuida de autenticação, versionamento de API
e formato de resposta por conta própria -- menos código nosso pra manter
sincronizado com mudança de API do lado deles.

Instalação: pip install cartesia

Existe pra responder uma pergunta concreta: o áudio saindo ruim é culpa do
TTS local (Pocket, CPU/GPU, modelo) ou do MECANISMO DE PLAYBACK do
bridge.js (play_start/play_chunk/timer de 20ms)? Cartesia tem streaming
real de servidor -- se o áudio sair limpo com Cartesia usando o MESMO
bridge.js, o suspeito era o TTS. Se sair ruim do mesmo jeito, o suspeito
é o playback. Ver EVA_VOZ_STREAMING em config.py pra ligar o caminho de
streaming e comparar.

DOIS CAMINHOS, mesma dupla que o Pocket TTS expõe:

  gerar_para_discord()       bloqueante -- um request só. Usado pelo
                              caminho padrão (EVA_VOZ_STREAMING=0).
    gerar_frase_stream_sync()  streaming real via WebSocket persistente -- a
                                                            conexão sobe uma vez por call e cada frase
                                                            cria apenas um contexto novo dentro dela.

ACHADO REAL (erro em produção): "'TtsClientWithWebsocket' object has no
attribute 'generate'". O SDK oficial da Cartesia se declara oficialmente
instável ("This API is still in alpha. Please expect breaking changes")
e o nome desses dois métodos JÁ mudou mais de uma vez entre versões
publicadas -- em algum momento foi `client.tts.generate()`, em outro
`client.tts.bytes()`; o streaming foi `client.tts.sse()` e em versões
mais novas é `client.tts.generate_sse()`. Não dá pra confiar em UM nome
fixo funcionando depois do próximo `pip install --upgrade cartesia` sem
aviso nenhum. Por isso os dois métodos abaixo RESOLVEM o nome certo em
runtime (`_resolver_metodo`), tentando os candidatos conhecidos na ordem
em que apareceram nas versões do SDK, com cache pra não ficar checando
de novo a cada frase -- e se NENHUM candidato existir na versão
instalada, o erro final lista os métodos que EXISTEM de verdade em
`client.tts` agora, em vez de um AttributeError sem contexto nenhum.
Mesmo espírito de "silêncio seria pior" que já apareceu neste projeto.

FORMATO DE ÁUDIO: pedimos direto pcm_s16le -- a Cartesia já devolve
int16, sem precisar da conversão float32->int16 que o Pocket exige (é
vocoder neural, produz float cru; a API já entrega formato final). Só
falta resample+estéreo (audio_utils.py, mesma função que o Pocket usa).

Documentação: https://docs.cartesia.ai -- mas trate o nome exato dos
métodos como algo a CONFIRMAR na sua versão instalada
(`pip show cartesia`), não como fato estável -- ver ACHADO REAL acima.
"""

from __future__ import annotations

import base64
import os
from typing import Callable, Iterator

import numpy as np

from .audio import alinhar_frames
from .audio_utils import resample_and_stereo

try:
    from cartesia import Cartesia
    CARTESIA_DISPONIVEL = True
except ImportError:
    Cartesia = None
    CARTESIA_DISPONIVEL = False

FRAME_DISCORD = 3840  # 20ms em 48kHz estéreo s16le -- igual ao pocket.py

# Candidatos conhecidos pro método de geração bloqueante, na ordem em que
# apareceram em versões publicadas do SDK -- ver ACHADO REAL no docstring
# do módulo. "generate" primeiro porque é o nome no api.md mais recente
# do repositório oficial (cartesia-ai/cartesia-python, branch main) no
# momento em que isto foi escrito.
_CANDIDATOS_GERACAO = ("generate", "bytes")
# Idem para o streaming via SSE -- "sse" era o nome quando este arquivo
# foi escrito originalmente; "generate_sse" é o nome atual no api.md.
_CANDIDATOS_SSE = ("sse", "generate_sse")

# Idiomas que a Cartesia documenta suporte oficial. Fora dessa lista não
# trava, só sai sem garantia de qualidade -- mesmo espírito do aviso
# equivalente em pocket.py.
IDIOMAS = {
    "en", "fr", "de", "es", "pt", "zh", "ja", "hi", "it",
    "ko", "nl", "pl", "ru", "sv", "tr",
}


class ErroCartesiaTTS(Exception):
    pass


def _resolver_metodo(tts_client, candidatos: tuple[str, ...], contexto: str) -> Callable:
    """Acha qual dos nomes candidatos existe de verdade em `tts_client`
    (a instância `client.tts` do SDK), na ordem dada. Levanta erro claro
    com a lista real de métodos disponíveis se nenhum bater -- ver
    ACHADO REAL no docstring do módulo pro motivo de isto existir em vez
    de chamar `client.tts.generate(...)` direto.
    """
    for nome in candidatos:
        metodo = getattr(tts_client, nome, None)
        if metodo is not None:
            return metodo
    disponiveis = sorted(m for m in dir(tts_client) if not m.startswith("_"))
    raise ErroCartesiaTTS(
        f"nenhum método conhecido para {contexto} ({'/'.join(candidatos)}) "
        f"existe em client.tts nesta versão do SDK instalado. Métodos "
        f"disponíveis agora: {disponiveis}. O SDK da Cartesia é instável "
        f"e já renomeou isso antes -- rode 'pip show cartesia' pra ver a "
        f"versão, confira https://github.com/cartesia-ai/cartesia-python/"
        f"blob/main/api.md pro nome atual, e ajuste _CANDIDATOS_GERACAO/"
        f"_CANDIDATOS_SSE no topo de cartesia.py."
    )


def _extrair_bytes(resposta) -> bytes:
    """Normaliza o retorno do método de geração bloqueante -- versões
    diferentes do SDK devolvem coisas diferentes aqui: um objeto de
    resposta com `.read()` (estilo BinaryAPIResponse/httpx.Response),
    bytes crus direto, ou um generator de pedaços de bytes (quando o
    método é na verdade um endpoint "streamed" por baixo, caso de
    `client.tts.bytes()` em algumas versões). Trata os três em vez de
    assumir um só e quebrar quando o SDK trocar de novo.
    """
    if hasattr(resposta, "read"):
        return resposta.read()
    if isinstance(resposta, (bytes, bytearray)):
        return bytes(resposta)
    return b"".join(resposta)  # generator/iterável de pedaços de bytes


def _extrair_audio_do_evento(evento) -> bytes | None:
    """Normaliza um evento de streaming SSE -- o nome do campo com o
    áudio já apareceu como `.data` e como `.audio` em versões/tipos
    diferentes do SDK (ver TTSSSEEvent vs. o exemplo de
    websocket_connect() na documentação oficial). O áudio vem como texto
    em base64 no SSE e precisa ser decodificado antes de virar PCM.
    """
    for atributo in ("data", "audio"):
        valor = getattr(evento, atributo, None)
        if valor:
            if isinstance(valor, str):
                try:
                    return base64.b64decode(valor, validate=False)
                except (ValueError, TypeError, base64.binascii.Error):
                    continue
            if isinstance(valor, (bytes, bytearray)):
                return bytes(valor)
    if isinstance(evento, (bytes, bytearray)) and evento:
        return bytes(evento)
    return None


class CartesiaTTSEngine:
    """Cartesia Sonic via SDK oficial -- sem estado local pra carregar (ao
    contrário do Pocket, não existe "modelo" pra baixar/inicializar de
    verdade). `inicializar()`/`_carregado` continuam existindo só pra
    bater a mesma forma que bridge_client.py já espera de qualquer motor
    de TTS -- ver _falar_stream, que chama isso incondicionalmente.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        idioma: str = "pt",
        model_id: str | None = None,
        sample_rate: int = 44100,
    ):
        if not CARTESIA_DISPONIVEL:
            raise ErroCartesiaTTS(
                "pacote 'cartesia' não instalado. Rode: pip install cartesia"
            )
        self.api_key = api_key or os.environ.get("CARTESIA_API_KEY", "")
        self.voice_id = voice_id or os.environ.get("EVA_TTS_CARTESIA_VOICE", "")
        self.idioma = idioma
        if self.idioma not in IDIOMAS:
            print(f"[cartesia-tts] aviso: idioma '{self.idioma}' fora da lista "
                  f"documentada ({sorted(IDIOMAS)}). Tentando mesmo assim.")
        # "sonic-3.5" é o modelo mais novo em ago/2026 -- confira
        # docs.cartesia.ai/build-with-cartesia/tts-models se isso mudar.
        self.model_id = model_id or os.environ.get("EVA_TTS_CARTESIA_MODEL", "sonic-3.5")
        self.sample_rate = sample_rate
        self._carregado = False
        self._cliente = None
        # Cache do método resolvido (ver _resolver_metodo) -- resolvido
        # uma vez em _validar_sync, não a cada frase.
        self._metodo_geracao: Callable | None = None
        self._metodo_sse: Callable | None = None
        self._ws = None

    # ------------------------------------------------------- inicialização

    def _validar_sync(self) -> None:
        """Síncrona de propósito -- criar o cliente do SDK não faz
        request nenhum, só guarda a chave. Chamada tanto por
        inicializar() (que só existe pra bater a interface async que
        bridge_client.py espera) quanto direto pelos métodos síncronos
        -- o caminho bloqueante (falar()/sintetizar()) nunca passa por
        inicializar() explicitamente, só o de streaming faz isso.
        """
        if self._carregado:
            return
        if not self.api_key:
            raise ErroCartesiaTTS(
                "CARTESIA_API_KEY não definida. Pegue em play.cartesia.ai/keys "
                "e coloque no .env."
            )
        if not self.voice_id:
            raise ErroCartesiaTTS(
                "EVA_TTS_CARTESIA_VOICE não definida -- precisa do voice_id "
                "da voz clonada da EVA (o mesmo já usado no EVA-V5)."
            )
        self._cliente = Cartesia(api_key=self.api_key)
        # Resolve AGORA, não na primeira frase: se nenhum candidato
        # existir na versão instalada, falha aqui com mensagem clara
        # (mesmo espírito de inicializar() -- falhar cedo, não na
        # primeira fala real). O de streaming (_metodo_sse) fica
        # opcional: se não existir, gerar_frase_stream_sync avisa e cai
        # pro bloqueante sozinho, em vez de derrubar o backend inteiro
        # por causa só do caminho de streaming.
        self._metodo_geracao = _resolver_metodo(
            self._cliente.tts, _CANDIDATOS_GERACAO, "geração bloqueante (gerar_para_discord)")
        self._carregado = True

    async def inicializar(self) -> None:
        """Não carrega nada de verdade (é uma API, não um modelo local) --
        só valida chave/voz e cria o cliente do SDK, pra falhar cedo e
        com mensagem clara em vez de um erro confuso na primeira fala.
        """
        self._validar_sync()

    def _abrir_websocket(self):
        """Abre ou reutiliza a conexão WebSocket persistente da call."""
        if self._ws is not None and not getattr(self._ws, "closed", False):
            return self._ws
        websocket_connect = getattr(self._cliente.tts, "websocket_connect", None)
        if websocket_connect is None:
            # CONFIRMADO contra a doc oficial atual (docs.cartesia.ai/get-started/
            # realtime-text-to-speech-quickstart): websocket_connect() + .context()
            # é o método certo e atual, não um nome trocado entre versões -- o
            # suporte a WebSocket é um EXTRA opcional do pacote ('cartesia[websockets]',
            # não só 'cartesia'). Se foi instalado sem o extra, o cliente sobe normal
            # (gerar_para_discord/client.tts.generate() continua funcionando) mas este
            # método fica ausente. Ver também: pip show cartesia.
            raise ErroCartesiaTTS(
                "esta versão do SDK Cartesia não oferece websocket_connect() -- "
                "o suporte a WebSocket é um extra opcional do pacote, não vem por "
                "padrão. Rode: pip install -U \"cartesia[websockets]\""
            )
        # BUG REAL (confirmado inspecionando o SDK 4.0.1 direto): websocket_connect()
        # devolve um TTSResourceConnectionManager -- só o GERENCIADOR do context
        # manager, sem .context() nele mesmo. A conexão de verdade (TTSResourceConnection,
        # que TEM .context()/.close()) é o que .enter() devolve. A versão anterior
        # guardava o manager em self._ws e descartava esse retorno -- por isso o erro
        # em produção era literalmente "'TTSResourceConnectionManager' object has no
        # attribute 'context'". .enter() é o alias público documentado pelo próprio
        # SDK pra usar fora de um bloco `with` (nosso caso: conexão persistente pela
        # call inteira, fechada explicitamente em fechar()).
        self._ws = websocket_connect().enter()
        return self._ws

    def fechar(self) -> None:
        """Fecha a conexão persistente da Cartesia, se existir."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def _iterar_contexto(self, ctx):
        receive = getattr(ctx, "receive", None)
        return receive() if receive is not None else ctx

    def _voz(self) -> dict:
        return {"mode": "id", "id": self.voice_id}

    def _formato_saida(self) -> dict:
        return {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }

    # ------------------------------------------------------------ REST

    def gerar_para_discord(self, texto: str) -> bytes:
        """PCM s16le 48kHz estéreo, pronto pro bridge -- um request só,
        bloqueante. SÍNCRONA de propósito: quem chama já roda isto numa
        thread via asyncio.to_thread (ver falar() em bridge_client.py),
        e o SDK da Cartesia não precisa de asyncio pra chamada simples.
        """
        texto = texto.strip()
        if not texto:
            return b""
        self._validar_sync()

        try:
            resposta = self._metodo_geracao(
                model_id=self.model_id,
                transcript=texto,
                voice=self._voz(),
                language=self.idioma,
                output_format=self._formato_saida(),
            )
            pcm16_mono = _extrair_bytes(resposta)
        except Exception as e:
            raise ErroCartesiaTTS(f"Cartesia: {e}") from e

        if not pcm16_mono:
            return b""
        amostras = np.frombuffer(pcm16_mono, dtype="<i2")
        estereo = resample_and_stereo(amostras, self.sample_rate)
        return alinhar_frames(estereo.tobytes(), FRAME_DISCORD)

    # ------------------------------------------------------- streaming

    def gerar_frase_stream_sync(
        self, texto: str, chunk_bytes: int = FRAME_DISCORD * 6, lote_ms: int = 120,
    ) -> Iterator[bytes]:
        """Streaming via WebSocket persistente.

        `gerar_frase_stream_sync()`  streaming real via WebSocket persistente --
        a Cartesia manda pedaços conforme gera. Cada frase usa um contexto novo
        dentro da conexão, que é reutilizada durante a call.
        Acumula ~120ms antes de devolver pedaço, mesmo motivo do Pocket
        (evita fatiar tão fino que vire muita mensagem WebSocket pro
        bridge.js) -- aqui não é sobre artefato de resample (a Cartesia
        já manda amostra pronta), é só sobre granularidade de mensagem.
        """

        try:
            ws = self._abrir_websocket()
            ctx = ws.context(
                model_id=self.model_id,
                voice=self._voz(),
                language=self.idioma,
                output_format=self._formato_saida(),
            )
            ctx.push(texto)
            ctx.no_more_inputs()
            eventos = self._iterar_contexto(ctx)
        except Exception as e:
            self.fechar()
            raise ErroCartesiaTTS(f"Cartesia (websocket): {e}") from e

        buffer = bytearray()
        try:
            for evento in eventos:
                pcm16_mono = _extrair_audio_do_evento(evento)
                if not pcm16_mono:
                    continue
                amostras = np.frombuffer(pcm16_mono, dtype="<i2")
                estereo = resample_and_stereo(amostras, self.sample_rate)
                buffer.extend(estereo.tobytes())
                while len(buffer) >= chunk_bytes:
                    yield bytes(buffer[:chunk_bytes])
                    del buffer[:chunk_bytes]
        except Exception as e:
            self.fechar()
            raise ErroCartesiaTTS(f"Cartesia (websocket, no meio): {e}") from e

        if buffer:
            resto = len(buffer) % FRAME_DISCORD
            if resto:
                buffer.extend(b"\x00" * (FRAME_DISCORD - resto))
            while buffer:
                fim = min(chunk_bytes, len(buffer))
                yield bytes(buffer[:fim])
                del buffer[:fim]

    def disponivel(self) -> bool:
        return CARTESIA_DISPONIVEL and bool(self.api_key)

    def gerar_frase_sync(self, texto: str) -> bytes:
        """Modo bloqueante -- mesma forma que PocketTTSEngine.gerar_frase_sync.

        Existe porque bridge_client._sintetizar_e_enfileirar chama este
        método por nome como rede de segurança, quando o streaming
        (gerar_frase_stream_sync) falha ANTES de produzir qualquer pedaço
        de áudio. Sem ele, essa classe não tinha `gerar_frase_sync`
        nenhum -- a chamada dava AttributeError, caía no except genérico
        de `_sintetizar_e_enfileirar`, e a frase inteira era perdida em
        silêncio (só um print de erro) em vez de cair pro bloqueante que
        já funciona. É literalmente só reusar `gerar_para_discord`, que
        já faz um request síncrono e devolve PCM pronto -- não tem um
        modo "mais bloqueante ainda" pra API, ao contrário do Pocket
        (onde bloqueante e streaming são dois caminhos internos
        genuinamente diferentes no motor local).
        """
        return self.gerar_para_discord(texto)