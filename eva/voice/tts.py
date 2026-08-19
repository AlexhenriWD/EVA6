"""
Text-to-speech.

UM BACKEND SÓ: POCKET TTS
--------------------------
Piper e edge-tts foram removidos. A versão anterior mantinha três caminhos
de síntese porque acreditava-se que o Pocket TTS só falava inglês e francês
-- o que fontes públicas de fato dizem, e está incompleto: o pacote traz
variantes para PT, ES, IT e DE. Desde a v2.0.0 do pocket-tts (abr/2026)
existe também a versão DESTILADA de 6 camadas pra cada um desses idiomas
("portuguese", não só "portuguese_24l") -- ver eva/voice/pocket.py para
o mapeamento atual e o motivo de ter trocado o padrão.

Com o português resolvido, os outros dois passaram a custar mais do que
davam. Três backends significam três timbres diferentes dependendo do que
estivesse instalado, três formas de falhar, e um fallback silencioso que
troca a voz da EVA sem avisar ninguém. A clonagem de voz é o motivo de a
EVA ter voz própria; um fallback que a descarta não é um fallback útil.

A camada abstrata continua aqui. Não é abstração por precaução: `sintetizar`
é o contrato que bridge_client e discord consomem, e o Pocket devolve PCM
48k pronto para o Discord enquanto um backend futuro devolveria WAV. Quem
chama não deveria saber a diferença.

Instalação:  pip install pocket-tts
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ErroTTS(Exception):
    pass


@dataclass
class Fala:
    audio: bytes          # WAV, ou PCM cru quando formato == "pcm48"
    taxa: int = 24000     # amostras por segundo
    formato: str = "wav"


class BackendTTS(ABC):
    nome: str = "abstrato"
    idiomas: tuple[str, ...] = ()

    @abstractmethod
    def sintetizar(self, texto: str, voz: str | None = None) -> Fala:
        ...

    @abstractmethod
    def disponivel(self) -> bool:
        ...

    def suporta(self, idioma: str) -> bool:
        return idioma.lower()[:2] in {i.lower()[:2] for i in self.idiomas}


# ------------------------------------------------------------ Pocket TTS


class PocketTTS(BackendTTS):
    """Kyutai Pocket TTS -- 100M params, CPU, clonagem de voz zero-shot.

    A variante PT usa por padrão o modelo DESTILADO de 6 camadas
    ("portuguese", pocket-tts >= 2.0.0), na mesma família de velocidade do
    inglês. O preview de 24 camadas ("portuguese_24l") ainda existe e pode
    ser pedido via EVA_TTS_IDIOMA=portuguese_24l, mas deixou de ser o
    padrão -- era mais lento em CPU, provável causa raiz do streaming ter
    saído fragmentado (ver histórico em bridge_client.py). Meça a latência
    real no seu hardware antes de assumir tempo real de qualquer jeito.

    Delega para eva.voice.pocket.PocketTTSEngine, que carrega os achados de
    produção. Não chame o modelo direto sem eles -- cada um corrige um bug
    audível:

      1. split por sentença antes de gerar, senão a fala para no meio por
         EOS prematuro;
      2. normalizar UMA vez no áudio concatenado, nunca por sentença, senão
         o volume salta entre frases;
      3. acumular ~120ms antes de resamplear -- resamplear pedaços de 10ms
         e colar gera estática (transiente do filtro FIR).
    """

    nome = "pocket"
    idiomas = ("pt", "en", "fr", "de", "it", "es")

    def __init__(self, voz: str | None = None, idioma: str = "pt",
                 quantizar: bool | None = None, **kwargs):
        self.voz = voz
        self.idioma = idioma
        self.quantizar = quantizar
        self._motor = None

    def disponivel(self) -> bool:
        try:
            import pocket_tts  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def motor(self):
        """Instância do motor, criada sob demanda."""
        if self._motor is None:
            from .pocket import PocketTTSEngine
            self._motor = PocketTTSEngine(
                idioma=self.idioma, voz=self.voz, quantizar=self.quantizar)
        return self._motor

    def sintetizar(self, texto: str, voz: str | None = None) -> Fala:
        """Gera áudio já no formato do Discord.

        Devolve PCM 48kHz estéreo pronto -- não precisa passar pela conversão
        do ffmpeg depois.
        """
        import asyncio

        if voz and voz != self.voz:
            self.voz = voz
            self._motor = None  # força recriar com a voz nova

        async def _gerar():
            return await self.motor.gerar_para_discord(texto)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pcm = asyncio.run(_gerar())
        else:
            # já dentro de um laço (bridge): roda numa thread própria
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                pcm = ex.submit(asyncio.run, _gerar()).result(timeout=300)

        return Fala(audio=pcm, taxa=48000, formato="pcm48")


# ---------------------------------------------------------- Cartesia TTS


class CartesiaTTS(BackendTTS):
    """Cartesia Sonic -- API paga, streaming real via Server-Sent Events
    (não um chunking caseiro por cima de um clipe pronto). Usa o SDK
    oficial (pacote `cartesia`), não HTTP cru. Alternativa ao Pocket pra
    isolar se o áudio ruim é do TTS local ou do mecanismo de playback do
    bridge.js -- ver eva/voice/cartesia.py pro raciocínio completo.

    `voz` aqui NÃO é um caminho de WAV (isso é convenção do Pocket) --
    Cartesia usa voice_id (string/UUID da voz já clonada na plataforma
    deles). Por isso ignorado aqui de propósito; o motor lê
    EVA_TTS_CARTESIA_VOICE direto, pra não conflitar com EVA_TTS_VOZ
    (que continua sendo o WAV do Pocket) quando os dois backends
    convivem no mesmo .env.
    """

    nome = "cartesia"
    idiomas = ("pt", "en", "fr", "de", "es", "zh", "ja", "hi", "it",
              "ko", "nl", "pl", "ru", "sv", "tr")

    def __init__(self, voz: str | None = None, idioma: str = "pt", **kwargs):
        self.idioma = idioma
        self._motor = None

    def disponivel(self) -> bool:
        try:
            import cartesia  # noqa: F401
        except ImportError:
            return False
        import os
        return bool(os.environ.get("CARTESIA_API_KEY"))

    @property
    def motor(self):
        if self._motor is None:
            from .cartesia import CartesiaTTSEngine
            self._motor = CartesiaTTSEngine(idioma=self.idioma)
        return self._motor

    def sintetizar(self, texto: str, voz: str | None = None) -> Fala:
        """Gera áudio já no formato do Discord.

        Ao contrário do Pocket, gerar_para_discord() da Cartesia é
        SÍNCRONO puro (sem asyncio por dentro -- é um POST HTTP comum,
        sem lock de modelo local pra proteger). Não precisa da ginástica
        de asyncio.run que o Pocket precisa; quem chama já roda isto
        numa thread via asyncio.to_thread (ver falar() em
        bridge_client.py), então é seguro bloquear aqui direto.
        """
        pcm = self.motor.gerar_para_discord(texto)
        return Fala(audio=pcm, taxa=48000, formato="pcm48")


# --------------------------------------------------------------- fábrica


BACKENDS = {"pocket": PocketTTS, "cartesia": CartesiaTTS}


def criar_tts(backend: str | None = None, idioma: str = "pt", **kwargs) -> BackendTTS:
    """Cria o backend de TTS.

    `backend`: "pocket" (local, CPU, clonagem via WAV) ou "cartesia" (API
    paga, streaming real via WebSocket -- ver eva/voice/cartesia.py).
    Nome desconhecido levanta erro em vez de cair num substituto: trocar
    a voz da EVA em silêncio é pior do que não falar.
    """
    nome = (backend or "pocket").lower()
    if nome not in BACKENDS:
        raise ErroTTS(
            f"backend desconhecido: {nome}. Disponíveis: {sorted(BACKENDS)}.\n"
            "Piper e edge-tts foram removidos -- só 'pocket' (local) e "
            "'cartesia' (API) existem hoje."
        )

    inst = BACKENDS[nome](idioma=idioma, **kwargs)
    if not inst.disponivel():
        motivo = {
            "pocket": "Pocket TTS não está instalado. Rode: pip install pocket-tts",
            "cartesia": ("Cartesia indisponível -- confira se 'pip install cartesia' "
                        "foi rodado e se CARTESIA_API_KEY está no .env."),
        }.get(nome, f"backend '{nome}' não está disponível.")
        raise ErroTTS(motivo)
    if not inst.suporta(idioma):
        print(f"[tts] aviso: {nome} não tem variante para '{idioma}' "
              f"(tem {inst.idiomas}). A pronúncia vai sair errada.")
    return inst


def diagnostico() -> dict:
    """O que está instalado e o que suporta."""
    saida = {}
    for nome, classe in BACKENDS.items():
        try:
            inst = classe(idioma="pt")
            saida[nome] = {
                "instalado": inst.disponivel(),
                "idiomas": list(inst.idiomas),
                "suporta_pt": inst.suporta("pt"),
            }
        except Exception as e:
            saida[nome] = {"instalado": False, "erro": str(e)[:80]}
    return saida