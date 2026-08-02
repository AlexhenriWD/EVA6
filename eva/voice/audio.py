"""
Conversão de áudio para o formato do Discord.

O bridge (Node) troca PCM cru 48kHz, estéreo, 16 bits little-endian, em
frames de 3840 bytes (20ms). O TTS entrega outra coisa:

    Piper      WAV, 22050Hz, mono
    edge-tts   MP3, 24000Hz, mono
    Pocket     WAV, 24000Hz, mono

Então tudo passa por aqui antes de ir para o Discord.

Duas estratégias:

FFMPEG (padrão): já é exigido pelo Discord de qualquer forma, lida com
qualquer formato de entrada e faz resample de qualidade. É o caminho
confiável.

NUMPY (alternativa): resample por interpolação linear, sem processo
externo. Mais rápido para WAV simples, e serve quando o ffmpeg não está
disponível. A qualidade é pior em conversões grandes, mas para fala é
aceitável.
"""

from __future__ import annotations

import io
import shutil
import struct
import subprocess
import wave

TAXA_DISCORD = 48000
CANAIS_DISCORD = 2
LARGURA_DISCORD = 2          # 16 bits
FRAME_BYTES = 3840           # 20ms @ 48kHz estéreo s16le


class ErroAudio(Exception):
    pass


def tem_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


# ------------------------------------------------------------- ffmpeg


def para_pcm_discord_ffmpeg(audio: bytes, formato_entrada: str | None = None) -> bytes:
    """Converte qualquer áudio para PCM 48kHz estéreo s16le via ffmpeg."""
    if not tem_ffmpeg():
        raise ErroAudio(
            "ffmpeg não encontrado no PATH.\n"
            "Windows: winget install ffmpeg\n"
            "Linux:   sudo apt install ffmpeg"
        )

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if formato_entrada:
        cmd += ["-f", formato_entrada]
    cmd += [
        "-i", "pipe:0",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(TAXA_DISCORD),
        "-ac", str(CANAIS_DISCORD),
        "pipe:1",
    ]

    proc = subprocess.run(cmd, input=audio, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise ErroAudio(f"ffmpeg falhou: {proc.stderr.decode('utf-8', 'replace')[:250]}")
    if not proc.stdout:
        raise ErroAudio("ffmpeg devolveu áudio vazio")
    return proc.stdout


# -------------------------------------------------------------- numpy


def _ler_wav(audio: bytes) -> tuple[bytes, int, int, int]:
    """Devolve (frames, taxa, canais, largura) de um WAV."""
    with wave.open(io.BytesIO(audio), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getsampwidth()


def para_pcm_discord_numpy(audio_wav: bytes) -> bytes:
    """Converte WAV para PCM 48kHz estéreo, usando numpy.

    Só aceita WAV -- MP3 exige decodificador, e aí o ffmpeg é obrigatório.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ErroAudio("numpy não instalado e ffmpeg indisponível") from e

    try:
        frames, taxa, canais, largura = _ler_wav(audio_wav)
    except wave.Error as e:
        raise ErroAudio(f"não é um WAV válido: {e}") from e

    if largura != 2:
        raise ErroAudio(f"esperado 16 bits, veio {largura * 8}")

    amostras = np.frombuffer(frames, dtype="<i2")
    if canais > 1:
        amostras = amostras.reshape(-1, canais).mean(axis=1).astype("<i2")

    # resample por interpolação linear
    if taxa != TAXA_DISCORD:
        n_saida = int(len(amostras) * TAXA_DISCORD / taxa)
        if n_saida <= 0:
            raise ErroAudio("áudio curto demais para converter")
        indices = np.linspace(0, len(amostras) - 1, n_saida)
        amostras = np.interp(indices, np.arange(len(amostras)), amostras.astype("f4"))
        amostras = amostras.astype("<i2")

    # mono -> estéreo (duplica o canal)
    estereo = np.repeat(amostras, 2)
    return estereo.astype("<i2").tobytes()


# --------------------------------------------------------------- api


def para_pcm_discord(audio: bytes, formato: str = "wav") -> bytes:
    """Converte áudio para o formato que o bridge espera.

    Usa ffmpeg quando disponível; cai para numpy se for WAV. MP3 sem
    ffmpeg não tem como ser convertido.
    """
    if not audio:
        raise ErroAudio("áudio vazio")

    if tem_ffmpeg():
        entrada = None if formato in ("wav", "mp3") else formato
        return para_pcm_discord_ffmpeg(audio, entrada)

    if formato == "wav":
        return para_pcm_discord_numpy(audio)

    raise ErroAudio(
        f"não consigo converter '{formato}' sem ffmpeg.\n"
        "Instale o ffmpeg, ou use um backend de TTS que produza WAV (piper)."
    )


def alinhar_frames(pcm: bytes, frame: int = FRAME_BYTES) -> bytes:
    """Completa o último frame com silêncio.

    O encoder Opus espera frames inteiros. Um frame parcial no fim produz
    estalo ou é descartado -- é a mesma proteção que o bridge faz do lado
    do Node, aplicada também aqui na origem.
    """
    resto = len(pcm) % frame
    if resto:
        pcm += b"\x00" * (frame - resto)
    return pcm


def duracao_segundos(pcm: bytes) -> float:
    return len(pcm) / (TAXA_DISCORD * CANAIS_DISCORD * LARGURA_DISCORD)


def pcm_para_wav(pcm: bytes, taxa: int = TAXA_DISCORD, canais: int = CANAIS_DISCORD) -> bytes:
    """Empacota PCM cru em WAV -- é o que o Whisper espera receber."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(canais)
        w.setsampwidth(LARGURA_DISCORD)
        w.setframerate(taxa)
        w.writeframes(pcm)
    return buf.getvalue()


def esta_silencioso(pcm: bytes, limiar: int = 300) -> bool:
    """Detecta se o trecho é praticamente silêncio.

    Serve para descartar áudio antes de gastar chamada de API: o Whisper
    alucina texto quando recebe silêncio, então é melhor nem enviar.
    """
    if not pcm:
        return True
    try:
        import numpy as np
        amostras = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
        if amostras.size == 0:
            return True
        rms = float(np.sqrt(np.mean(amostras.astype("f8") ** 2)))
        return rms < limiar
    except ImportError:
        # sem numpy: aproximação pelo pico
        pico = 0
        for i in range(0, min(len(pcm) - 1, 96000), 2):
            v = abs(struct.unpack_from("<h", pcm, i)[0])
            pico = max(pico, v)
        return pico < limiar * 3
