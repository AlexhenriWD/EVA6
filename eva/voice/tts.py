"""
Text-to-speech, com backends intercambiáveis.

POR QUE PLUGÁVEL, E NÃO SÓ POCKET TTS
--------------------------------------
O Pocket TTS (Kyutai, 100M params, roda em CPU em tempo real, clona voz com
5s de áudio) é excelente -- mas na data desta implementação suporta apenas
inglês e francês. A EVA fala português: passar PT-BR para ele produziria
fonética francesa aplicada a texto português, com sotaque quebrado.

Então a interface aqui é abstrata e há três backends:

  pocket   Pocket TTS. Melhor qualidade e clonagem de voz. Use quando a EVA
           falar inglês, ou quando o suporte a PT for adicionado.
  piper    Piper TTS. Tem vozes PT-BR, roda offline em CPU, rápido.
           É a escolha padrão para português local.
  edge     edge-tts. Vozes PT-BR da Microsoft, qualidade alta, de graça,
           mas depende de rede e não é oficialmente uma API pública.

Trocar de backend não afeta o resto do sistema: quem chama só chama
`sintetizar(texto) -> bytes de WAV`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ErroTTS(Exception):
    pass


@dataclass
class Fala:
    audio: bytes          # WAV
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

    Instalação:  pip install pocket-tts

    SUPORTA PORTUGUÊS via o modelo `portuguese_24l`. (Fontes públicas
    costumam listar só inglês e francês, mas o pacote traz variantes 24l
    para PT, ES, IT e DE.)

    A variante PT é "preview": modelo maior, não destilado, mais lento que
    os números divulgados para inglês. Funciona bem; vale medir a latência
    real no seu hardware.

    Este backend delega para eva.voice.pocket.PocketTTSEngine, que carrega
    os achados de produção -- split por sentença, normalização única,
    resample em lote. Não use o modelo direto sem eles: cada um corrige um
    bug audível.
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

        Ao contrário dos outros backends, este devolve PCM 48kHz estéreo
        pronto -- não precisa passar pela conversão do ffmpeg depois.
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


# ----------------------------------------------------------------- Piper


class PiperTTS(BackendTTS):
    """Piper -- offline, rápido, com vozes PT-BR de boa qualidade.

    Instalação:
        pip install piper-tts
        python -m piper.download_voices pt_BR-faber-medium

    É o backend padrão para português: roda local, não depende de rede e
    não tem custo por uso.
    """

    nome = "piper"
    idiomas = ("pt", "en", "es", "fr", "de", "it")

    def __init__(self, modelo: str = "pt_BR-faber-medium", executavel: str | None = None):
        self.modelo = modelo
        self.executavel = executavel or shutil.which("piper") or "piper"

    def disponivel(self) -> bool:
        if shutil.which(self.executavel):
            return True
        try:
            import piper  # noqa: F401
            return True
        except ImportError:
            return False

    def sintetizar(self, texto: str, voz: str | None = None) -> Fala:
        modelo = voz or self.modelo

        # Caminho 1: biblioteca Python (mais rápido, mantém o modelo em memória)
        try:
            from piper import PiperVoice
            import io
            import wave

            if not hasattr(self, "_voz_carregada") or self._voz_nome != modelo:
                self._voz_carregada = PiperVoice.load(modelo)
                self._voz_nome = modelo

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                self._voz_carregada.synthesize_wav(texto, wf)
            return Fala(audio=buf.getvalue(), taxa=self._voz_carregada.config.sample_rate)
        except ImportError:
            pass
        except Exception as e:
            raise ErroTTS(f"falha no piper (lib): {e}") from e

        # Caminho 2: binário na linha de comando
        if not shutil.which(self.executavel):
            raise ErroTTS(
                "Piper não encontrado. Instale com: pip install piper-tts\n"
                f"E baixe uma voz: python -m piper.download_voices {self.modelo}"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            saida = tmp.name
        try:
            proc = subprocess.run(
                [self.executavel, "-m", modelo, "-f", saida],
                input=texto.encode("utf-8"),
                capture_output=True, timeout=120,
            )
            if proc.returncode != 0:
                raise ErroTTS(f"piper falhou: {proc.stderr.decode()[:200]}")
            with open(saida, "rb") as f:
                return Fala(audio=f.read(), taxa=22050)
        finally:
            if os.path.exists(saida):
                os.unlink(saida)


# -------------------------------------------------------------- edge-tts


class EdgeTTS(BackendTTS):
    """edge-tts -- vozes PT-BR da Microsoft, qualidade alta, sem custo.

    Instalação:  pip install edge-tts

    Depende de rede e usa um serviço que não é uma API pública oficial --
    pode mudar sem aviso. Bom para prototipar e para qualidade máxima em
    português; para produção estável, prefira Piper.
    """

    nome = "edge"
    idiomas = ("pt", "en", "es", "fr", "de", "it", "ja")

    def __init__(self, voz: str = "pt-BR-FranciscaNeural", velocidade: str = "+0%"):
        self.voz = voz
        self.velocidade = velocidade

    def disponivel(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False

    def sintetizar(self, texto: str, voz: str | None = None) -> Fala:
        try:
            import asyncio
            import edge_tts
        except ImportError as e:
            raise ErroTTS("edge-tts não instalado. Rode: pip install edge-tts") from e

        async def _gerar() -> bytes:
            com = edge_tts.Communicate(texto, voz or self.voz, rate=self.velocidade)
            pedacos = []
            async for evento in com.stream():
                if evento["type"] == "audio":
                    pedacos.append(evento["data"])
            return b"".join(pedacos)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # já dentro de um loop (ex: bot do Discord) -- roda numa thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as ex:
                    audio = ex.submit(asyncio.run, _gerar()).result(timeout=120)
            else:
                audio = asyncio.run(_gerar())
        except Exception as e:
            raise ErroTTS(f"falha no edge-tts: {e}") from e

        # edge-tts devolve MP3, não WAV
        return Fala(audio=audio, taxa=24000, formato="mp3")


# ------------------------------------------------------------- seletor


BACKENDS = {"pocket": PocketTTS, "piper": PiperTTS, "edge": EdgeTTS}


def criar_tts(
    backend: str | None = None,
    idioma: str = "pt",
    **kwargs,
) -> BackendTTS:
    """Cria o backend de TTS.

    Sem `backend` explícito, escolhe o primeiro disponível que suporte o
    idioma. A ordem prioriza rodar local (Piper) antes de depender de rede
    (edge), e o Pocket entra quando o idioma é compatível.
    """
    if backend:
        if backend not in BACKENDS:
            raise ErroTTS(f"backend desconhecido: {backend}. Opções: {list(BACKENDS)}")
        try:
            inst = BACKENDS[backend](idioma=idioma, **kwargs) if backend == "pocket" \
                else BACKENDS[backend](**kwargs)
        except TypeError:
            inst = BACKENDS[backend](**kwargs)
        if not inst.suporta(idioma):
            # aviso, não erro: pode ser escolha consciente
            print(f"[tts] aviso: {backend} não suporta '{idioma}' "
                  f"(suporta {inst.idiomas}). A pronúncia vai sair errada.")
        return inst

    # Ordem de preferência: Pocket primeiro (melhor qualidade e clonagem
    # de voz, roda local), depois Piper (local, mais rápido de carregar),
    # e edge por último porque depende de rede.
    for nome in ("pocket", "piper", "edge"):
        try:
            inst = BACKENDS[nome](idioma=idioma, **kwargs) if nome == "pocket" \
                else BACKENDS[nome](**kwargs)
        except TypeError:
            inst = BACKENDS[nome]()
        if inst.disponivel() and inst.suporta(idioma):
            return inst

    raise ErroTTS(
        f"nenhum backend de TTS disponível para '{idioma}'.\n"
        "Instale um destes:\n"
        "  pip install pocket-tts  (offline, clonagem de voz, suporta PT)\n"
        "  pip install piper-tts   (offline, mais leve)\n"
        "  pip install edge-tts    (online, vozes PT-BR da Microsoft)"
    )


def diagnostico() -> dict:
    """Quais backends estão instalados e o que cada um suporta."""
    saida = {}
    for nome, classe in BACKENDS.items():
        try:
            inst = classe(idioma="pt") if nome == "pocket" else classe()
            saida[nome] = {
                "instalado": inst.disponivel(),
                "idiomas": list(inst.idiomas),
                "suporta_pt": inst.suporta("pt"),
            }
        except Exception as e:
            saida[nome] = {"instalado": False, "erro": str(e)[:80]}
    return saida
