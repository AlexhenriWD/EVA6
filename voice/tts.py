"""
Text-to-speech.

UM BACKEND SÓ: POCKET TTS
--------------------------
Piper e edge-tts foram removidos. A versão anterior mantinha três caminhos
de síntese porque acreditava-se que o Pocket TTS só falava inglês e francês
-- o que fontes públicas de fato dizem, e está incompleto: o pacote traz
variantes 24l para PT, ES, IT e DE, e o modelo `portuguese_24l` funciona.

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

    A variante PT é "preview": modelo maior, não destilado, mais lento que
    os números divulgados para inglês. Funciona bem; vale medir a latência
    real no seu hardware antes de assumir tempo real.

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

    def __init__(self, voz: str | None = None, idioma: str = "pt", **kwargs):
        self.voz = voz
        self.idioma = idioma
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
            self._motor = PocketTTSEngine(idioma=self.idioma, voz=self.voz)
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


# --------------------------------------------------------------- fábrica


BACKENDS = {"pocket": PocketTTS}


def criar_tts(backend: str | None = None, idioma: str = "pt", **kwargs) -> BackendTTS:
    """Cria o backend de TTS.

    `backend` existe para compatibilidade com quem já chamava assim; hoje só
    "pocket" é válido. Nome desconhecido levanta erro em vez de cair num
    substituto: trocar a voz da EVA em silêncio é pior do que não falar.
    """
    nome = (backend or "pocket").lower()
    if nome not in BACKENDS:
        raise ErroTTS(
            f"backend desconhecido: {nome}. O único disponível é 'pocket'.\n"
            "Piper e edge-tts foram removidos -- o Pocket TTS fala português "
            "via o modelo portuguese_24l."
        )

    inst = BACKENDS[nome](idioma=idioma, **kwargs)
    if not inst.disponivel():
        raise ErroTTS(
            "Pocket TTS não está instalado. Rode: pip install pocket-tts"
        )
    if not inst.suporta(idioma):
        print(f"[tts] aviso: pocket não tem variante para '{idioma}' "
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