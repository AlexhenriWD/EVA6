"""
audio_utils.py
==============
Utilitários de áudio compartilhados entre motores TTS (Kokoro, Pocket TTS, etc).

Centraliza duas coisas que antes só existiam duplicadas dentro de cada motor:
1. Conversão MONO -> STEREO 48kHz (formato exigido pelo Discord)
2. Ponte de um generator SÍNCRONO/bloqueante (ex.: generate_audio_stream do
   Pocket TTS) para um async generator, sem bloquear o event loop.

Extraído da lógica original em voice/tts_engine.py (motor Kokoro) para poder
ser reaproveitado por qualquer motor novo sem duplicar código.
"""
import asyncio
import queue
import re
import threading
from math import gcd
from typing import AsyncIterator, Callable, Iterable, List, TypeVar

import numpy as np

try:
    from scipy import signal as scipy_signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


T = TypeVar("T")


# ══════════════════════════════════════════════════════════════════
# CONVERSÃO DE ÁUDIO — MONO (qualquer sample rate) → STEREO 48kHz
# ══════════════════════════════════════════════════════════════════

def resample_and_stereo(audio: np.ndarray, source_rate: int, target_rate: int = 48000) -> np.ndarray:
    """
    Converte áudio MONO → STEREO no target_rate (48kHz por padrão, requisito Discord).

    Input : array 1-D int16, qualquer sample rate
    Output: array 1-D int16, target_rate Hz, interleaved L/R stereo
            [L0, R0, L1, R1, ...] — exatamente o que o Discord espera

    Idêntico em comportamento ao convert_to_48k_stereo() original do motor
    Kokoro (voice/tts_engine.py), só que reaproveitável por qualquer motor.
    """
    # ---- garantir 1-D ----
    if audio.ndim != 1:
        audio = audio.flatten()

    # ---- garantir int16 ----
    if audio.dtype != np.int16:
        audio = np.clip(audio, -32768, 32767).astype(np.int16)

    # ---- resample (ainda mono) ----
    if source_rate != target_rate:
        if SCIPY_AVAILABLE:
            try:
                g = gcd(target_rate, source_rate)
                up, down = target_rate // g, source_rate // g
                resampled = scipy_signal.resample_poly(
                    audio.astype(np.float32), up=up, down=down
                )
                audio = np.clip(resampled, -32768, 32767).astype(np.int16)
            except Exception:
                audio = _naive_resample(audio, source_rate, target_rate)
        else:
            audio = _naive_resample(audio, source_rate, target_rate)

    # ---- MONO → STEREO intercalado ----
    # np.repeat([a, b, c], 2) → [a, a, b, b, c, c] = [L0,R0,L1,R1,L2,R2]
    return np.repeat(audio, 2)


def _naive_resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Fallback sem scipy — interpolação linear simples."""
    ratio = target_rate / source_rate
    n = round(len(audio) * ratio)
    return np.interp(
        np.linspace(0, len(audio) - 1, n),
        np.arange(len(audio)),
        audio.astype(np.float32),
    ).astype(np.int16)


def float_audio_to_int16(samples: np.ndarray, headroom: float = 0.70) -> np.ndarray:
    """
    Converte um array de áudio float (qualquer range) para int16, normalizando
    com headroom para evitar clipping/chiado. Mesma técnica usada no motor
    Kokoro original: normaliza para `headroom` (70%) do pico, não para 100%.

    ⚠️ Só usar em cima do ÁUDIO COMPLETO de uma vez (generate_audio_native).
    NÃO usar em pedacinhos pequenos de streaming — cada pedaço tem um pico
    local diferente, e normalizar por pico pedaço-a-pedaço causa saltos de
    volume entre pedaços (confirmado como causa real de distorção/estática
    em produção). Pra streaming, usar float32_to_int16_fixed_scale() abaixo.
    """
    if samples.ndim != 1:
        samples = samples.flatten()
    samples = samples.astype(np.float32)
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak > 0:
        samples = samples / peak * headroom
    return np.clip(samples * 32767, -32768, 32767).astype(np.int16)


def torch_audio_to_int16(audio_tensor, headroom: float = 0.70) -> np.ndarray:
    """Igual a float_audio_to_int16(), mas aceita um torch.Tensor diretamente."""
    samples = audio_tensor.detach().to("cpu").numpy()
    return float_audio_to_int16(samples, headroom=headroom)


def torch_audio_to_float32(audio_tensor) -> np.ndarray:
    """Converte um torch.Tensor de áudio pra numpy float32, SEM normalizar/escalar."""
    samples = audio_tensor.detach().to("cpu").numpy().astype(np.float32)
    if samples.ndim != 1:
        samples = samples.flatten()
    return samples


def float32_to_int16_fixed_scale(samples: np.ndarray, scale: float = 0.85) -> np.ndarray:
    """
    Converte float (esperado ~[-1, 1], range típico de vocoder neural) pra
    int16 com escala FIXA — não adaptativa ao pico de cada chamada.

    Usada no streaming (voice/pocket_tts_engine.py stream_for_discord):
    normalizar cada pedacinho pelo próprio pico local causa saltos de
    volume entre pedaços vizinhos, já que picos locais de um sinal
    contínuo variam bastante em janelas pequenas — confirmado como causa
    real de distorção/estática em produção. Com escala fixa, o volume
    fica consistente do início ao fim da fala, ao custo de não aproveitar
    o range dinâmico completo se um trecho específico for muito baixo
    (troca válida: preferível a ter estática).
    """
    if samples.ndim != 1:
        samples = samples.flatten()
    return np.clip(samples * 32767 * scale, -32768, 32767).astype(np.int16)


# ══════════════════════════════════════════════════════════════════
# SPLIT DE SENTENÇAS — usado para pipelining/streaming incremental
# ══════════════════════════════════════════════════════════════════

_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')


def split_into_sentences(text: str, min_chars: int = 20) -> List[str]:
    """
    Divide texto em sentenças, mesclando fragmentos curtos até atingir
    `min_chars`, para não gerar chunks minúsculos demais para o TTS.

    Chunk abaixo do mínimo produz áudio degradado -- pouco contexto
    fonético faz o modelo gerar prosódia instável, às vezes só um som sem
    a palavra inteira ("Bom." virando um ruído em vez da palavra). O bug
    original: o acúmulo `buf` era despejado no final do loop mesmo abaixo
    de `min_chars`, então QUALQUER resposta cujo texto inteiro fosse menor
    que o mínimo saía como um chunk curto de qualquer forma -- e isso
    inclui boa parte do dataset, cuja mediana de resposta é 74 caracteres
    e tem exemplos de 4 ("Boa.").

    Fix: se o último fragmento acumulado ainda estiver abaixo do mínimo,
    ele se funde ao fragmento anterior já fechado em vez de sair sozinho.
    Só fica curto mesmo se o texto INTEIRO for menor que min_chars -- nesse
    caso não há com o que fundir, e authorship do modelo já deveria evitar
    respostas tão curtas em modo voz.
    """
    raw = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    if not raw:
        return []

    merged: List[str] = []
    buf = ""
    for s in raw:
        buf = f"{buf} {s}".strip() if buf else s
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""

    if buf:
        if merged and len(buf) < min_chars:
            # funde ao chunk anterior em vez de mandar sozinho e curto --
            # ele já passou do mínimo, um pouco mais não degrada nada
            merged[-1] = f"{merged[-1]} {buf}".strip()
        else:
            # não há chunk anterior para fundir (texto inteiro é curto):
            # não tem como não ser curto, mas ao menos não fica pior
            merged.append(buf)
    return merged


_FIM_FRASE_RE = re.compile(r'(?<=[.!?])\s+')


def extrair_frases_fechadas(buffer: str, min_chars: int = 20) -> tuple[List[str], str]:
    """Do buffer acumulado do streaming do LLM, separa as frases já
    FECHADAS (terminam em .!? seguido de espaço) do resto, que ainda
    pode crescer.

    Prima de split_into_sentences: aquela decide sobre o TEXTO INTEIRO já
    pronto; esta decide sobre um buffer que ainda está sendo escrito,
    token a token. Fragmento curto (< min_chars) não sai sozinho -- fica
    esperando fundir com o próximo fechamento, mesmo espírito de
    split_into_sentences, só que prospectivo em vez de retrospectivo
    (aqui não dá pra olhar pra frente, só esperar mais texto chegar).

    Herda a mesma limitação de split_into_sentences: abreviação com ponto
    ("Dr.", "3.14") pode cortar frase no lugar errado. Regex simples não
    resolve isso com confiança; não é problema novo, é o mesmo trade-off
    que o caminho não-streaming já aceita.
    """
    partes = _FIM_FRASE_RE.split(buffer)
    if len(partes) <= 1:
        return [], buffer

    fechadas: List[str] = []
    pendente = ""
    for trecho in partes[:-1]:
        pendente = f"{pendente} {trecho}".strip() if pendente else trecho
        if len(pendente) >= min_chars:
            fechadas.append(pendente)
            pendente = ""
    resto = f"{pendente} {partes[-1]}".strip() if pendente else partes[-1]
    return fechadas, resto


# ══════════════════════════════════════════════════════════════════
# PONTE: generator SÍNCRONO/bloqueante → async generator
# ══════════════════════════════════════════════════════════════════
#
# Pocket TTS (e outros motores locais baseados em PyTorch) expõem geração
# incremental como um generator Python comum (`for chunk in model.generate_audio_stream(...)`),
# não como um async generator. Rodar esse generator direto dentro de uma
# coroutine bloquearia o event loop inteiro a cada chunk.
#
# `iterate_blocking_generator` roda o generator síncrono numa thread separada
# e repassa os itens pro lado assíncrono através de uma queue thread-safe,
# preservando o streaming incremental sem travar o loop.

_SENTINEL_OK = object()


async def iterate_blocking_generator(func: Callable[..., Iterable[T]], *args, **kwargs) -> AsyncIterator[T]:
    """
    Executa `func(*args, **kwargs)` (que deve retornar/ser um generator
    síncrono) em uma thread dedicada, e expõe os valores como um async
    generator, sem bloquear o event loop enquanto espera o próximo item.
    """
    loop = asyncio.get_event_loop()
    q: "queue.Queue" = queue.Queue()

    def _worker():
        try:
            for item in func(*args, **kwargs):
                q.put((True, item))
            q.put((False, None))
        except Exception as e:  # repassa a exceção pro lado async
            q.put((False, e))

    thread = threading.Thread(target=_worker, daemon=True, name="pocket-tts-stream")
    thread.start()

    try:
        while True:
            ok, item = await loop.run_in_executor(None, q.get)
            if not ok:
                if isinstance(item, Exception):
                    raise item
                break
            yield item
    finally:
        # Não há como interromper o generator síncrono no meio de forma limpa;
        # a thread termina sozinha quando o generator interno se esgota.
        pass