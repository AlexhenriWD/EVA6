"""
Motor TTS local com Kyutai Pocket TTS.

Baseado no pocket_tts_engine.py do projeto EVA-V5, com os achados de
produção preservados -- cada um deles corrige um bug real:

1. SPLIT POR SENTENÇA (generate_for_discord)
   Um parágrafo inteiro numa chamada só pode bater o detector de EOS antes
   do texto acabar, e a fala para no meio. O próprio Kyutai resolve isso
   quebrando em sentenças.

2. NORMALIZAÇÃO ÚNICA, NO FIM
   Cada sentença é gerada em float CRU e só o áudio concatenado é
   normalizado. Normalizar por sentença faria o volume saltar entre elas,
   porque cada trecho tem um pico local diferente.

3. RESAMPLE EM LOTE (stream_for_discord)
   O filtro do resample tem transiente nas duas pontas de qualquer
   entrada. Resamplear pedaços de 5-20ms e colar produz estática audível.
   O acúmulo de ~120ms antes de resamplear elimina isso e ainda mantém
   streaming de verdade.

4. ALINHAMENTO DE FRAME
   Frame parcial no fim é descartado ou estala no encoder Opus do bridge.

5. LOCK
   generate_audio_stream não é thread-safe: uma geração por vez por
   instância.

SOBRE O IDIOMA
   Português usa a variante `portuguese_24l`, que o Kyutai descreve como
   preview -- modelo maior, não destilado, mais lento que os números de
   inglês. Funciona; vale medir a latência real.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Optional, Union

import numpy as np

from .audio import alinhar_frames
from .audio_utils import (
    float32_to_int16_fixed_scale,
    float_audio_to_int16,
    iterate_blocking_generator,
    resample_and_stereo,
    split_into_sentences,
    torch_audio_to_float32,
    torch_audio_to_int16,
)

try:
    from pocket_tts import TTSModel, export_model_state
    POCKET_TTS_DISPONIVEL = True
except ImportError:
    TTSModel = None
    export_model_state = None
    POCKET_TTS_DISPONIVEL = False


VOZ_PADRAO = "alba"
FRAME_DISCORD = 3840  # 20ms em 48kHz estéreo s16le -- igual ao bridge.js

IDIOMAS = {
    "english", "english_2026-01", "english_2026-04",
    "french_24l", "german_24l", "portuguese_24l",
    "italian_24l", "spanish_24l",
}

# Mapeia código curto para o nome do modelo, para quem configura via
# EVA_TTS_IDIOMA=pt não precisar saber do sufixo _24l
ATALHOS = {
    "pt": "portuguese_24l", "pt-br": "portuguese_24l",
    "en": "english", "fr": "french_24l", "de": "german_24l",
    "it": "italian_24l", "es": "spanish_24l",
}


class ErroPocketTTS(Exception):
    pass


class PocketTTSEngine:
    """Pocket TTS: CPU, clonagem zero-shot, streaming nativo."""

    # Abaixo disso, o áudio às vezes sai como ruído em vez de fala -- ver o
    # comentário em gerar_para_discord. Não é limite documentado pelo
    # Kyutai, é observação empírica; ajuste se medir outro número no seu
    # hardware/voz de referência.
    PISO_CARACTERES = 15

    def __init__(
        self,
        idioma: str = "portuguese_24l",
        voz: Optional[Union[str, Path]] = None,
        temp: float = 0.7,
        device: Optional[str] = None,
    ):
        if not POCKET_TTS_DISPONIVEL:
            raise ErroPocketTTS(
                "pacote 'pocket-tts' não instalado. Rode: pip install pocket-tts"
            )

        idioma = ATALHOS.get(idioma.lower(), idioma)
        if idioma not in IDIOMAS:
            print(f"[pocket-tts] aviso: idioma '{idioma}' fora da lista conhecida "
                  f"({sorted(IDIOMAS)}). Tentando mesmo assim.")

        self.idioma = idioma
        self.voz = voz or os.environ.get("EVA_TTS_VOZ") or VOZ_PADRAO
        self.temp = temp
        self.device_pedido = device or os.environ.get("POCKET_TTS_DEVICE") or None

        self.model = None
        self.voice_state = None
        self.sample_rate = 24000

        # Pocket TTS não é thread-safe: uma geração por vez por instância.
        self._lock = asyncio.Lock()
        self._lock_sync = threading.Lock()
        self._carregado = False
        self._lock_carga = asyncio.Lock()

    # ------------------------------------------------------- carregamento

    def _resolver_device(self) -> str:
        if self.device_pedido:
            return self.device_pedido
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _carregar_sync(self) -> None:
        print(f"[pocket-tts] carregando ({self.idioma})... pode demorar na 1ª vez", flush=True)
        t0 = time.time()
        self.model = TTSModel.load_model(language=self.idioma, temp=self.temp)

        alvo = self._resolver_device()
        if alvo != "cpu":
            try:
                self.model = self.model.to(alvo)
                print(f"[pocket-tts] movido para {alvo.upper()}")
            except Exception as e:
                print(f"[pocket-tts] não consegui usar {alvo} ({e}); seguindo em CPU")

        self.sample_rate = getattr(self.model, "sample_rate", 24000)
        self.voice_state = self.model.get_state_for_audio_prompt(str(self.voz))
        print(f"[pocket-tts] pronto em {time.time()-t0:.1f}s "
              f"(sample_rate={self.sample_rate}Hz, voz={self.voz})", flush=True)

    async def inicializar(self) -> None:
        """Carrega o modelo. Idempotente e seguro sob concorrência."""
        if self._carregado:
            return
        async with self._lock_carga:
            if self._carregado:
                return
            await asyncio.to_thread(self._carregar_sync)
            self._carregado = True

    # ----------------------------------------------------------- clonagem

    async def definir_voz(self, referencia: Union[str, Path], truncar: bool = False) -> None:
        """Troca a voz da EVA (clonagem zero-shot).

        Aceita .wav de 5-20s de fala limpa, .safetensors já exportado,
        URL hf://, ou nome de voz nativa (alba, azelma, cosette, javert).
        """
        if not self._carregado:
            await self.inicializar()
        async with self._lock:
            self.voice_state = await asyncio.to_thread(
                self.model.get_state_for_audio_prompt, str(referencia), truncar
            )
            self.voz = referencia
        print(f"[pocket-tts] voz atualizada: {referencia}")

    async def exportar_voz(self, destino: Union[str, Path]) -> None:
        """Salva a voz atual em .safetensors -- recarregar depois é quase
        instantâneo, porque não reprocessa o áudio de referência."""
        if self.voice_state is None:
            raise ErroPocketTTS("nenhuma voz carregada; chame inicializar() antes")
        await asyncio.to_thread(export_model_state, self.voice_state, str(destino))
        print(f"[pocket-tts] voz exportada para {destino}")

    # ------------------------------------------------------------ geração

    async def gerar_nativo(self, texto: str) -> tuple[np.ndarray, int]:
        """Áudio mono int16 na taxa nativa do modelo."""
        if not self._carregado:
            await self.inicializar()
        async with self._lock:
            tensor = await asyncio.to_thread(
                self.model.generate_audio, self.voice_state, texto
            )
        return torch_audio_to_int16(tensor), self.sample_rate

    def gerar_frase_sync(self, texto: str) -> bytes:
        """Sintetiza UMA frase e devolve PCM 48kHz estéreo já alinhado a
        frame. Bloqueante e SÍNCRONA de propósito -- pra ser chamada de
        dentro da thread dedicada do streaming de voz (bridge_client), nunca
        do event loop direto.

        Sem split por sentença aqui dentro: a frase já chega fechada de fora
        (extrair_frases_fechadas, audio_utils.py). Escala FIXA
        (float32_to_int16_fixed_scale), não normalização por pico -- senão o
        volume salta de frase pra frase. Resample por frase, não por
        pedacinho de 5-20ms: uma frase inteira (meio segundo a poucos
        segundos) não sofre o artefato de transiente do filtro FIR que
        motivou o acúmulo de 120ms em stream_para_discord -- aquele bug era
        sobre fragmentos bem menores que isso.
        """
        if not self._carregado:
            raise ErroPocketTTS("motor não inicializado -- chame inicializar() antes")
        texto = texto.strip()
        if not texto:
            return b""
        with self._lock_sync:
            tensor = self.model.generate_audio(self.voice_state, texto)
        amostras = torch_audio_to_float32(tensor)
        int16 = float32_to_int16_fixed_scale(amostras)
        estereo = resample_and_stereo(int16, self.sample_rate)
        return alinhar_frames(estereo.tobytes(), FRAME_DISCORD)

    def gerar_frase_stream_sync(
        self, texto: str, chunk_bytes: int = FRAME_DISCORD * 6, lote_ms: int = 120
    ):
        """Sintetiza UMA frase via generate_audio_stream (streaming real
        do Pocket TTS, ~200ms até o primeiro pedaço), gerando PCM 48kHz
        estéreo pronto pro Discord conforme o modelo produz -- ao
        contrário de gerar_frase_sync, não espera o clipe inteiro da
        frase terminar antes de devolver o primeiro byte.

        Generator SÍNCRONO e bloqueante de propósito, igual
        gerar_frase_sync: generate_audio_stream do próprio modelo já é
        um generator bloqueante comum (não asyncio), então dá pra
        consumir direto aqui dentro da thread dedicada do streaming de
        voz -- sem precisar da ponte iterate_blocking_generator que
        stream_para_discord usa (aquela existe pra quando quem chama
        está no event loop; aqui já estamos numa thread comum).

        Mesmo acúmulo de ~120ms antes de resamplear que
        stream_para_discord usa e pelo mesmo motivo (resamplear pedaço
        de 5-20ms cru gera estática, transiente do filtro FIR), e mesma
        escala fixa em vez de normalização por pico (volume não pode
        saltar entre pedaços vizinhos). chunk_bytes por padrão é ~120ms
        (6 frames de 20ms) em vez de 1 frame -- manda pedaço grande o
        bastante pra não virar uma mensagem WebSocket a cada 20ms, sem
        perder a granularidade que evita esperar a frase inteira.
        """
        if not self._carregado:
            raise ErroPocketTTS("motor não inicializado -- chame inicializar() antes")
        texto = texto.strip()
        if not texto:
            return

        buffer = bytearray()
        acumulado = np.zeros(0, dtype=np.float32)
        min_amostras = max(1, int(self.sample_rate * lote_ms / 1000))

        def processar_lote(amostras: np.ndarray) -> bytes:
            int16 = float32_to_int16_fixed_scale(amostras)
            return resample_and_stereo(int16, self.sample_rate).tobytes()

        with self._lock_sync:
            for pedaco in self.model.generate_audio_stream(self.voice_state, texto):
                acumulado = np.concatenate(
                    [acumulado, torch_audio_to_float32(pedaco)])

                if len(acumulado) >= min_amostras:
                    buffer.extend(processar_lote(acumulado))
                    acumulado = np.zeros(0, dtype=np.float32)
                    while len(buffer) >= chunk_bytes:
                        yield bytes(buffer[:chunk_bytes])
                        del buffer[:chunk_bytes]

        if len(acumulado):
            buffer.extend(processar_lote(acumulado))

        if buffer:
            resto = len(buffer) % FRAME_DISCORD
            if resto:
                buffer.extend(b"\x00" * (FRAME_DISCORD - resto))
            while buffer:
                fim = min(chunk_bytes, len(buffer))
                yield bytes(buffer[:fim])
                del buffer[:fim]

    async def gerar_para_discord(self, texto: str) -> bytes:
        """PCM s16le 48kHz estéreo, pronto para o bridge.

        Gera sentença a sentença (senão a fala para no meio em textos
        longos), mantém tudo em float cru, e só depois de concatenar
        normaliza e resampleia -- uma vez cada. Fazer isso por sentença
        causaria salto de volume e descontinuidade.

        PISO MÍNIMO DE ENTRADA (achado em produção, não documentado pelo
        Kyutai): texto muito curto -- "Bom.", "Sim." -- às vezes produz só
        um som sem a palavra inteira, em vez de fala. Modelo de síntese
        precisa de contexto fonético mínimo pra estabilizar a prosódia; com
        poucos caracteres ele não tem o que prever direito. Isso pega em
        cheio o dataset da EVA: mediana de resposta é 74 chars, mas há
        exemplos de 4 ("Boa."), e em modo voz o teto empurra pra respostas
        ainda mais curtas.

        Não há chunk anterior pra fundir aqui -- diferente do
        split_into_sentences, isto é o TEXTO INTEIRO da resposta, não um
        fragmento de um texto maior. A única saída real sem trocar de
        modelo é aceitar a fala como está (arriscando degradação) ou pedir
        pro modelo tentar de novo -- e mais uma vez não muda o texto de
        entrada, então não resolve nada. Por ora: log de aviso, para você
        saber quando está acontecendo e decidir se vale ajustar o dataset
        (menos respostas ultra-curtas em modo voz) em vez de mascarar aqui.
        """
        if not texto or not texto.strip():
            return b""
        if not self._carregado:
            await self.inicializar()

        texto = texto.strip()
        if len(texto) < self.PISO_CARACTERES:
            print(f"[pocket-tts] aviso: texto curto ({len(texto)} chars) "
                  f"pode sair degradado: {texto!r}", flush=True)

        sentencas = split_into_sentences(texto) or [texto]

        partes: list[np.ndarray] = []
        async with self._lock:
            for s in sentencas:
                tensor = await asyncio.to_thread(
                    self.model.generate_audio, self.voice_state, s
                )
                partes.append(torch_audio_to_float32(tensor))

        if not partes:
            return b""

        inteiro = partes[0] if len(partes) == 1 else np.concatenate(partes)
        int16 = float_audio_to_int16(inteiro)
        estereo = resample_and_stereo(int16, self.sample_rate)
        return alinhar_frames(estereo.tobytes(), FRAME_DISCORD)

    async def stream_para_discord(
        self, texto: str, chunk_bytes: int = FRAME_DISCORD, lote_ms: int = 120
    ) -> AsyncIterator[bytes]:
        """Streaming real, com o primeiro áudio saindo em ~120ms.

        O acúmulo antes de resamplear não é otimização: resamplear pedaços
        de 5-20ms e colar produz estática audível, porque o filtro tem
        transiente nas duas pontas de cada entrada. Com lotes de ~120ms o
        erro desaparece, e a latência continua baixa.

        A conversão usa escala FIXA, não normalização por pedaço -- cada
        trecho tem pico local diferente, e normalizar faria o volume
        saltar entre chunks.
        """
        if not texto or not texto.strip():
            return
        if not self._carregado:
            await self.inicializar()

        buffer = bytearray()
        acumulado = np.zeros(0, dtype=np.float32)
        min_amostras = max(1, int(self.sample_rate * lote_ms / 1000))

        def processar_lote(amostras: np.ndarray) -> bytes:
            int16 = float32_to_int16_fixed_scale(amostras)
            return resample_and_stereo(int16, self.sample_rate).tobytes()

        async with self._lock:
            async for pedaco in iterate_blocking_generator(
                self.model.generate_audio_stream, self.voice_state, texto
            ):
                acumulado = np.concatenate([acumulado, torch_audio_to_float32(pedaco)])

                if len(acumulado) >= min_amostras:
                    buffer.extend(processar_lote(acumulado))
                    acumulado = np.zeros(0, dtype=np.float32)
                    while len(buffer) >= chunk_bytes:
                        yield bytes(buffer[:chunk_bytes])
                        del buffer[:chunk_bytes]

        if len(acumulado):
            buffer.extend(processar_lote(acumulado))

        if buffer:
            resto = len(buffer) % chunk_bytes
            if resto:
                buffer.extend(b"\x00" * (chunk_bytes - resto))
            while len(buffer) >= chunk_bytes:
                yield bytes(buffer[:chunk_bytes])
                del buffer[:chunk_bytes]

    def disponivel(self) -> bool:
        return POCKET_TTS_DISPONIVEL