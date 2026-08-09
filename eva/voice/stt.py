"""
Speech-to-text via Groq (Whisper).

A Groq expõe endpoints compatíveis com a API da OpenAI, então usamos HTTP
direto em vez do SDK -- uma dependência a menos, e o que precisamos é um
POST multipart.

Modelos disponíveis:
    whisper-large-v3-turbo  rápido, multilíngue, ótimo custo/benefício
    whisper-large-v3        mais preciso, um pouco mais lento

Passamos `language="pt"` de propósito: sem isso o Whisper detecta o idioma
sozinho e às vezes erra em áudio curto ou com ruído, transcrevendo
português como espanhol. Fixar o idioma elimina essa classe de erro.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


class ErroSTT(Exception):
    pass


@dataclass
class Transcricao:
    texto: str
    duracao: float | None = None
    idioma: str | None = None
    # Probabilidade média de "não é fala". Útil para descartar ruído que o
    # Whisper transcreve como alucinação (ele tende a inventar texto em
    # silêncio, tipo "Legendas pela comunidade Amara.org").
    prob_sem_fala: float | None = None

    @property
    def vazia(self) -> bool:
        return not self.texto.strip()


def _montar_multipart(campos: dict[str, str], arquivo: tuple[str, bytes]) -> tuple[bytes, str]:
    """Monta um corpo multipart/form-data sem depender de biblioteca externa."""
    limite = f"----eva{uuid.uuid4().hex}"
    linhas: list[bytes] = []

    for nome, valor in campos.items():
        linhas.append(f"--{limite}\r\n".encode())
        linhas.append(f'Content-Disposition: form-data; name="{nome}"\r\n\r\n'.encode())
        linhas.append(f"{valor}\r\n".encode())

    nome_arquivo, conteudo = arquivo
    tipo = mimetypes.guess_type(nome_arquivo)[0] or "application/octet-stream"
    linhas.append(f"--{limite}\r\n".encode())
    linhas.append(
        f'Content-Disposition: form-data; name="file"; filename="{nome_arquivo}"\r\n'.encode()
    )
    linhas.append(f"Content-Type: {tipo}\r\n\r\n".encode())
    linhas.append(conteudo)
    linhas.append(b"\r\n")
    linhas.append(f"--{limite}--\r\n".encode())

    return b"".join(linhas), f"multipart/form-data; boundary={limite}"


class GroqSTT:
    def __init__(
        self,
        api_key: str | None = None,
        modelo: str = "whisper-large-v3-turbo",
        idioma: str | None = "pt",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.modelo = modelo
        self.idioma = idioma
        self.timeout = timeout

    def disponivel(self) -> bool:
        return bool(self.api_key)

    def transcrever_bytes(
        self,
        audio: bytes,
        nome: str = "audio.wav",
        prompt: str | None = None,
    ) -> Transcricao:
        """Transcreve áudio em memória.

        `prompt` guia o vocabulário -- útil para nomes próprios e termos
        técnicos que o Whisper erraria. Ex: "EVA, Alex, Arch Linux, LoRA".
        """
        if not self.api_key:
            raise ErroSTT(
                "GROQ_API_KEY não definida. Pegue uma chave em console.groq.com "
                "e coloque no .env."
            )
        if not audio:
            raise ErroSTT("áudio vazio")

        campos = {
            "model": self.modelo,
            # verbose_json traz metadados que permitem descartar alucinação
            "response_format": "verbose_json",
            "temperature": "0",
        }
        if self.idioma:
            campos["language"] = self.idioma
        if prompt:
            campos["prompt"] = prompt

        corpo, content_type = _montar_multipart(campos, (nome, audio))
        req = urllib.request.Request(
            GROQ_URL,
            data=corpo,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": content_type,
                # Sem isso, urllib manda "User-Agent: Python-urllib/3.x" --
                # e o Cloudflare na frente da API da Groq bloqueia esse UA
                # por padrão como assinatura óbvia de bot/scraper. O erro
                # que aparece não é da Groq: é a página de bloqueio do
                # Cloudflare (HTTP 403, "error code: 1010"), sem JSON, sem
                # relação nenhuma com a chave ou com rate limit. Qualquer
                # string de navegador real resolve; não precisa ser exata.
                "User-Agent": "Mozilla/5.0 (compatible; EVA/1.0)",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                dados = json.loads(r.read())
        except urllib.error.HTTPError as e:
            detalhe = e.read().decode("utf-8", errors="replace")[:300]
            raise ErroSTT(f"Groq HTTP {e.code}: {detalhe}") from e
        except urllib.error.URLError as e:
            raise ErroSTT(f"falha de rede ao falar com a Groq: {e.reason}") from e

        segmentos = dados.get("segments") or []
        prob = None
        if segmentos:
            probs = [s.get("no_speech_prob", 0.0) for s in segmentos]
            prob = sum(probs) / len(probs)

        return Transcricao(
            texto=(dados.get("text") or "").strip(),
            duracao=dados.get("duration"),
            idioma=dados.get("language"),
            prob_sem_fala=prob,
        )

    def transcrever_arquivo(self, caminho: str, prompt: str | None = None) -> Transcricao:
        with open(caminho, "rb") as f:
            return self.transcrever_bytes(f.read(), os.path.basename(caminho), prompt)


# --------------------------------------------------- whisper.cpp local

import subprocess
import tempfile
from pathlib import Path


class WhisperCppSTT:
    """STT local via whisper.cpp (binário compilado com Vulkan).

    Medido contra a Groq no mesmo áudio de 25s: 6.1s aqui vs 11.7s na
    Groq -- quase 2x mais rápido, mesmos pesos (ggml-large-v3-turbo),
    mesma transcrição. Elimina o round-trip de rede que era o maior
    componente da latência de voz.

    IMPORTANTE sobre o backend de aceleração: o binário PRECISA ter sido
    compilado com Vulkan (GGML_VULKAN=1), não HIP/ROCm -- o caminho HIP
    crashou em runtime nesta máquina (access violation, 0xC0000005,
    provavelmente ligação de DLL). Isso é decidido na hora de compilar o
    whisper.cpp, não em tempo de execução aqui -- esta classe não tem
    como detectar ou trocar isso, só usar o binário que você apontar.

    Mesma interface pública do GroqSTT (transcrever_bytes, disponivel)
    de propósito -- bridge_client.py troca de um pro outro sem precisar
    saber qual está por trás.
    """

    def __init__(
        self,
        exe: str,
        modelo: str,
        idioma: str | None = "pt",
        timeout: int = 60,
    ):
        self.exe = exe
        self.modelo = modelo
        self.idioma = idioma
        self.timeout = timeout

    def disponivel(self) -> bool:
        return bool(self.exe) and Path(self.exe).exists() and Path(self.modelo).exists()

    def transcrever_bytes(
        self,
        audio: bytes,
        nome: str = "audio.wav",
        prompt: str | None = None,
    ) -> Transcricao:
        """`prompt` é aceito pela mesma assinatura do GroqSTT mas
        ignorado aqui -- whisper.cpp não tem um parâmetro equivalente de
        vocabulário guiado via linha de comando. O nome do parâmetro é
        mantido só para os dois backends serem intercambiáveis sem
        mudar a chamada em bridge_client.py.
        """
        if not self.disponivel():
            raise ErroSTT(
                f"whisper.cpp não disponível: exe={self.exe!r} "
                f"modelo={self.modelo!r} -- confira os dois caminhos"
            )
        if not audio:
            raise ErroSTT("áudio vazio")

        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / nome
            wav.write_bytes(audio)
            prefixo = Path(tmp) / "saida"

            cmd = [
                self.exe,
                "-m", self.modelo,
                "-f", str(wav),
                "-otxt", "-of", str(prefixo),
                "-nt",  # sem timestamp
            ]
            if self.idioma:
                cmd += ["-l", self.idioma]

            try:
                proc = subprocess.run(
                    cmd, capture_output=True, timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as e:
                raise ErroSTT(f"whisper.cpp excedeu {self.timeout}s") from e

            if proc.returncode != 0:
                erro = proc.stderr.decode("utf-8", errors="replace")[:400]
                raise ErroSTT(f"whisper.cpp saiu com erro {proc.returncode}: {erro}")

            arq = prefixo.with_suffix(".txt")
            texto = arq.read_text(encoding="utf-8").strip() if arq.exists() else ""

        # whisper.cpp não expõe no_speech_prob por linha de comando --
        # prob_sem_fala fica None. parece_ruido() já trata None com
        # segurança (if t.prob_sem_fala is not None), então esse filtro
        # específico simplesmente não dispara com este backend; os
        # outros dois (texto vazio/curto, lista de alucinações) continuam
        # funcionando normalmente. Perda de precisão, não bug.
        return Transcricao(texto=texto)

    def transcrever_arquivo(self, caminho: str, prompt: str | None = None) -> Transcricao:
        with open(caminho, "rb") as f:
            return self.transcrever_bytes(f.read(), Path(caminho).name, prompt)


# --------------------------------------------------------------- fábrica

def criar_stt(voz_cfg) -> tuple[object, str | None]:
    """Cria o backend de STT configurado (`voz_cfg` é EVAConfig.voz).

    Ao contrário do TTS (onde um fallback silencioso trocaria a voz da
    EVA sem avisar, e por isso criar_tts() prefere quebrar), aqui um
    fallback não troca identidade nenhuma -- só perde a vantagem de
    latência do whisper.cpp local. Por isso, se EVA_STT_BACKEND=
    whisper_cpp for pedido mas o binário/modelo não forem encontrados,
    cai pra Groq em vez de travar a EVA de ouvir. O aviso volta como
    segundo item da tupla para quem chama logar -- não é silencioso,
    só não é fatal.

    Devolve (instancia, aviso_ou_None).
    """
    nome = (voz_cfg.stt_backend or "groq").lower()

    if nome == "whisper_cpp":
        inst = WhisperCppSTT(
            exe=voz_cfg.stt_whisper_cpp_exe,
            modelo=voz_cfg.stt_whisper_cpp_modelo,
            idioma=voz_cfg.stt_idioma,
        )
        if inst.disponivel():
            return inst, None
        aviso = (
            f"EVA_STT_BACKEND=whisper_cpp mas não achei o binário/modelo "
            f"(exe={voz_cfg.stt_whisper_cpp_exe!r}, "
            f"modelo={voz_cfg.stt_whisper_cpp_modelo!r}). Caindo para "
            f"Groq -- confira EVA_STT_WHISPER_CPP_EXE e "
            f"EVA_STT_WHISPER_CPP_MODELO no .env."
        )
        return GroqSTT(
            api_key=voz_cfg.stt_chave, modelo=voz_cfg.stt_modelo,
            idioma=voz_cfg.stt_idioma,
        ), aviso

    if nome != "groq":
        print(f"[stt] aviso: backend '{nome}' desconhecido, usando 'groq'.")

    return GroqSTT(
        api_key=voz_cfg.stt_chave, modelo=voz_cfg.stt_modelo,
        idioma=voz_cfg.stt_idioma,
    ), None


# Frases que o Whisper costuma alucinar em silêncio ou ruído. Filtrar isso
# evita a EVA responder a algo que ninguém disse -- o que numa call em
# grupo aconteceria o tempo todo.
ALUCINACOES = {
    "legendas pela comunidade amara.org",
    "amara.org",
    "obrigado por assistir",
    "obrigada por assistir",
    "inscreva-se no canal",
    "tchau",
    "...",
    "thank you.",
    "thanks for watching",
    "subtitles by the amara.org community",
}


# Palavras curtas que são fala legítima. Sem essa lista, o filtro de
# tamanho mínimo descartaria "oi", "sim" e "não" -- que numa conversa por
# voz são justamente as respostas mais frequentes.
CURTAS_VALIDAS = {
    "oi", "ola", "olá", "sim", "não", "nao", "ok", "certo", "claro", "opa",
    "hey", "ei", "aham", "uhum", "beleza", "valeu", "obrigado", "obrigada",
    "para", "pare", "espera", "calma", "eva",
}


def parece_ruido(t: Transcricao, prob_maxima: float = 0.6, min_caracteres: int = 3) -> bool:
    """Decide se uma transcrição deve ser descartada.

    Três filtros, e cada um pega um caso diferente:
      - texto vazio ou curtíssimo (exceto palavras curtas legítimas)
      - probabilidade alta de não ser fala (métrica do próprio Whisper)
      - frase da lista de alucinações conhecidas
    """
    texto = t.texto.strip().lower().rstrip(".!? ")

    if not texto:
        return True
    if texto in ALUCINACOES:
        return True
    if t.prob_sem_fala is not None and t.prob_sem_fala > prob_maxima:
        return True
    if len(texto) < min_caracteres and texto not in CURTAS_VALIDAS:
        return True
    return False