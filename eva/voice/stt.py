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

from pathlib import Path


def _duracao_wav(dados: bytes) -> float | None:
    """Duração em segundos de um WAV em memória, ou None se não der pra
    saber com segurança."""
    import io
    import wave

    try:
        with wave.open(io.BytesIO(dados), "rb") as w:
            taxa = w.getframerate()
            if not taxa:
                return None
            return w.getnframes() / float(taxa)
    except Exception:
        return None


class WhisperCppServerSTT:
    """STT local via SERVIDOR persistente do whisper.cpp (examples/server
    do próprio repositório), não mais via `subprocess.run()` por chamada.

    ACHADO REAL (log de call em produção): o tempo de STT ficava
    praticamente CONSTANTE (~3s) não importa se o áudio tinha 1s ou 25s
    de fala -- sinal claro de custo FIXO dominando, não custo
    proporcional ao áudio. Causa: a versão anterior desta classe rodava
    `subprocess.run([exe, "-m", modelo, ...])` a cada frase, e isso
    recarrega o modelo GGML inteiro (large-v3-turbo, alguns GB) do disco
    em TODA transcrição -- exatamente o mesmo problema que "Just-in-time
    model loading" resolve pro LM Studio, só que aqui não existia
    equivalente.

    Fix: subir o mesmo binário como SERVIDOR (`./server -m modelo.bin
    --port 8090`), que carrega o modelo UMA vez e fica residente. Esta
    classe só faz o HTTP POST por chamada, igual ao GroqSTT (reusa
    _montar_multipart, sem dependência nova).

    IMPORTANTE: o servidor precisa estar rodando ANTES da EVA, num
    terminal separado -- mesmo padrão operacional que já existe pro LM
    Studio (processo próprio, de fora do Python). Não é auto-spawned
    daqui de propósito: evita gerenciar ciclo de vida de subprocesso
    (crash/restart) só pra economizar um comando manual.

        ./server -m F:\\models\\whisper\\ggml-large-v3-turbo.bin ^
            --host 127.0.0.1 --port 8090 -t 8

    Continua exigindo o binário compilado com Vulkan (GGML_VULKAN=1), não
    HIP/ROCm -- o caminho HIP crashou em runtime nesta máquina (access
    violation, 0xC0000005). Confira o nome exato do binário/flags na sua
    versão compilada (`./server --help`) -- muda entre versões do
    whisper.cpp, mesmo espírito de instabilidade de SDK já documentado em
    cartesia.py.

    Mesma interface pública do GroqSTT (transcrever_bytes, disponivel)
    de propósito -- bridge_client.py troca de um pro outro sem precisar
    saber qual está por trás.
    """

    def __init__(self, url: str, idioma: str | None = "pt", timeout: int = 30):
        self.url = url.rstrip("/")
        self.idioma = idioma
        self.timeout = timeout

    def disponivel(self) -> bool:
        # Checagem leve, sem golpear rede na inicialização -- confia na
        # config; erro de conexão real aparece com mensagem clara na
        # primeira chamada de transcrever_bytes().
        return bool(self.url)

    def transcrever_bytes(
        self,
        audio: bytes,
        nome: str = "audio.wav",
        prompt: str | None = None,
    ) -> Transcricao:
        """`prompt` é aceito pela mesma assinatura do GroqSTT mas
        ignorado aqui -- whisper.cpp não tem um parâmetro equivalente de
        vocabulário guiado via este endpoint. O nome do parâmetro é
        mantido só para os dois backends serem intercambiáveis sem
        mudar a chamada em bridge_client.py.
        """
        if not audio:
            raise ErroSTT("áudio vazio")

        campos = {"response_format": "json", "temperature": "0"}
        if self.idioma:
            campos["language"] = self.idioma

        # audio_ctx proporcional à duração do clipe.
        #
        # O encoder do Whisper processa uma janela FIXA de 30s (contexto
        # de áudio 1500) independentemente do tamanho da fala -- é o
        # grosso do custo de inferência, e numa call ele é gasto quase
        # todo em preenchimento. Reduzir esse contexto faz o encoder
        # avaliar proporcionalmente mais rápido.
        #
        # A fórmula é a empírica da comunidade -- (duração/30)*1500 + 128
        # -- arredondada pra múltiplo de 64 (o kernel prefere) e com piso
        # em 1024.
        #
        # O piso era 768 e SUBIU depois de aparecer em call real: um
        # clipe de 7.8s voltou com a frase inteira duplicada ("Eu não sei
        # como te responder disso... / Eu não sei como te responder
        # disso..."). É o sintoma exato de contexto curto demais -- o
        # decoder foi TREINADO em janelas de 30s e repete os últimos
        # tokens quando o encoder entrega menos do que ele espera.
        #
        # Esse defeito NÃO é pego por `parece_ruido()`: não é alucinação
        # conhecida nem texto curto, é uma frase legítima repetida. Ela
        # chega ao LLM como se a pessoa tivesse falado duas vezes.
        #
        # 1024 mantém boa parte do ganho (encoder processa ~20s em vez de
        # 30) com bem mais margem. Se a duplicação voltar, o próximo passo
        # é 1280 -- e se voltar em 1280 também, o certo é remover o campo:
        # encoder mais rápido não vale transcrição errada.
        #
        # Campo desconhecido é ignorado por builds que não suportam, então
        # mandar sempre é seguro; o ganho é que aparece sozinho quando o
        # build suportar.
        #
        # A duração sai do CABEÇALHO, não de uma divisão por constante: o
        # que chega aqui é 48kHz estéreo (pcm_para_wav usa os padrões do
        # Discord), mas `transcrever_arquivo` aceita qualquer WAV de
        # qualquer lugar, e errar o formato aqui vira audio_ctx errado --
        # que é justamente o parâmetro capaz de fazer o decoder entrar em
        # loop. Formato desconhecido = não manda o campo e segue no
        # comportamento padrão.
        segundos = _duracao_wav(audio)
        if segundos:
            alvo = int((segundos / 30.0) * 1500) + 128
            alvo = max(1024, min(1500, (alvo // 64) * 64))
            if alvo < 1500:
                campos["audio_ctx"] = str(alvo)

        corpo, content_type = _montar_multipart(campos, (nome, audio))
        req = urllib.request.Request(
            f"{self.url}/inference", data=corpo,
            headers={"Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                bruto = r.read()
        except urllib.error.HTTPError as e:
            detalhe = e.read().decode("utf-8", errors="replace")[:300]
            raise ErroSTT(f"whisper.cpp server HTTP {e.code}: {detalhe}") from e
        except urllib.error.URLError as e:
            raise ErroSTT(
                f"não consegui falar com o whisper.cpp server em {self.url}: "
                f"{e.reason} -- ele está rodando? Ver docstring de "
                f"WhisperCppServerSTT pro comando de subir ele."
            ) from e

        # Normaliza a resposta -- builds diferentes do server variam
        # entre devolver JSON {"text": "..."} ou texto puro, mesmo
        # espírito defensivo de _extrair_bytes em cartesia.py.
        try:
            texto = json.loads(bruto).get("text", "")
        except (json.JSONDecodeError, AttributeError):
            texto = bruto.decode("utf-8", errors="replace")

        # whisper.cpp não expõe no_speech_prob por este endpoint --
        # prob_sem_fala fica None. parece_ruido() já trata None com
        # segurança (if t.prob_sem_fala is not None), então esse filtro
        # específico simplesmente não dispara com este backend; os
        # outros dois (texto vazio/curto, lista de alucinações) continuam
        # funcionando normalmente. Perda de precisão, não bug.
        return Transcricao(texto=texto.strip())

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
        inst = WhisperCppServerSTT(
            url=voz_cfg.stt_whisper_cpp_url,
            idioma=voz_cfg.stt_idioma,
        )
        if inst.disponivel():
            return inst, None
        aviso = (
            f"EVA_STT_BACKEND=whisper_cpp mas EVA_STT_WHISPER_CPP_URL não "
            f"está definida. Caindo para Groq -- confira essa variável no "
            f".env e se o servidor whisper.cpp (./server -m ...) está "
            f"rodando."
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