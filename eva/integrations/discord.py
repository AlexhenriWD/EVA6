"""
Sobe a EVA no Discord: bridge Node + cliente Python.

O bridge cuida do Discord (voz e texto) e este processo cuida do cérebro.
A separação existe porque a stack de voz do discord.py é problemática --
o discord.js com @discordjs/voice é o caminho maduro.

Uso:
    python -m eva.integrations.discord           # sobe tudo (bridge, whisper,
                                                   # llama-server de conversa/
                                                   # decisão-visão/embeddings)
    python -m eva.integrations.discord --so-python   # bridge já rodando
    python -m eva.integrations.discord --diagnostico
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import subprocess
import sys

# No Windows o console (cmd/PowerShell) roda em cp1252/cp850 por padrão, e o
# stdout do Python herda essa codepage. Como este módulo imprime logs com
# emoji (às vezes vindos do próprio bridge.js repassado), sem isso o texto
# aparece corrompido (ex.: "🔌" vira "ðŸ”Œ") mesmo depois do subprocesso Node
# já estar sendo lido em UTF-8 corretamente. reconfigure() existe desde o
# Python 3.7; getattr cobre o caso raro de stdout já ter sido substituído
# por algo sem esse método.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

from ..config import carregar_config
from .bridge_client import ClienteBridge

AMARELO = "\033[33m"
VERDE = "\033[32m"
FIM = "\033[0m"

# ---------------------------------------------------------------- log

_ANSI = __import__("re").compile(r"\x1b\[[0-9;]*m")


class _Tee:
    """Duplica tudo que sai no console para um arquivo.

    Ponto único de propósito: TODO log deste sistema passa por `print()`
    -- inclusive o do bridge.js, do whisper-server e dos três
    llama-servers, que os supervisores leem do subprocesso e reimprimem
    com prefixo. Envolvendo `sys.stdout` uma vez, o arquivo recebe o
    mesmo fluxo intercalado que você vê no terminal, na mesma ordem.
    Fazer isso supervisor por supervisor daria cinco arquivos separados
    e nenhum deles teria a ordem relativa entre os processos -- que é
    justamente o que permitiu descobrir que o whisper estava rodando em
    cima do prefill do llama.

    Duas diferenças em relação ao console:

    - cada linha do arquivo ganha carimbo de tempo. Os llama-servers já
      carimbam os deles, mas os `print()` do Python não, e sem isso não
      dá pra medir a distância entre "[call] fulano: ..." e o
      "[tempo] até o primeiro som" que veio depois.
    - códigos de cor ANSI são removidos, senão viram lixo tipo `←[33m`
      no meio do texto.

    `isatty()` delega pro stdout original justamente pra `_cor()`
    continuar colorindo o console -- envolver o stdout não deveria
    mudar como ele se parece pra quem está olhando.
    """

    def __init__(self, original, arquivo):
        self._original = original
        self._arquivo = arquivo
        self._inicio_de_linha = True

    def write(self, texto: str) -> int:
        self._original.write(texto)
        try:
            limpo = _ANSI.sub("", texto)
            for parte in limpo.splitlines(keepends=True):
                if self._inicio_de_linha and parte.strip():
                    self._arquivo.write(_agora_hms() + " ")
                self._arquivo.write(parte)
                self._inicio_de_linha = parte.endswith("\n")
        except Exception:
            # Arquivo cheio, disco removido, permissão -- perder o log é
            # ruim, derrubar a EVA no meio de uma call por causa dele é
            # pior. Console continua funcionando de qualquer jeito.
            pass
        return len(texto)

    def flush(self) -> None:
        self._original.flush()
        try:
            self._arquivo.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        return self._original.isatty()

    def __getattr__(self, nome):
        return getattr(self._original, nome)


def _agora_hms() -> str:
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def iniciar_log_em_arquivo(manter: int = 20):
    """Abre o arquivo de log desta execução e redireciona stdout/stderr.

    Sempre ligado -- não é opção. O terminal do Windows tem buffer de
    rolagem limitado e corta o começo das sessões longas, que é
    exatamente a parte que interessa quando alguma coisa demorou. Um
    arquivo por execução, nomeado pela hora de início.

    EVA_LOG_DIR muda o destino. `manter` apaga os mais antigos pra pasta
    não crescer sem fim -- 20 execuções cobrem semanas de
    desenvolvimento e não chegam a alguns MB.

    Devolve o Path do arquivo, ou None se não deu pra criar (o que NÃO
    é fatal: a EVA sobe do mesmo jeito, só sem log em arquivo).
    """
    import datetime

    try:
        destino = Path(os.environ.get("EVA_LOG_DIR", "")
                       or (Path.cwd() / "logs"))
        destino.mkdir(parents=True, exist_ok=True)

        antigos = sorted(destino.glob("eva-*.log"))
        for velho in antigos[:-manter] if manter > 0 else []:
            try:
                velho.unlink()
            except OSError:
                pass

        carimbo = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        caminho = destino / f"eva-{carimbo}.log"
        # line buffering: se a EVA travar ou você matar com Ctrl+C, o que
        # já foi impresso está no disco. Com buffer cheio, justamente o
        # fim -- a parte que explica o travamento -- se perderia.
        arquivo = open(caminho, "w", encoding="utf-8", buffering=1,
                       errors="replace")
    except Exception as e:
        print(f"[log] não consegui criar o arquivo de log: {e}")
        return None

    sys.stdout = _Tee(sys.stdout, arquivo)
    sys.stderr = _Tee(sys.stderr, arquivo)
    return caminho


def _cor(t: str, c: str) -> str:
    return f"{c}{t}{FIM}" if sys.stdout.isatty() else t


def achar_bridge() -> Path | None:
    """Procura o bridge.js nos lugares prováveis."""
    candidatos = [
        Path(os.environ.get("EVA_BRIDGE_JS", "")),
        Path.cwd() / "bridge.js",
        Path.cwd() / "bridge" / "bridge.js",
        Path(__file__).resolve().parent.parent.parent / "bridge.js",
    ]
    for c in candidatos:
        if c and c.name and c.exists():
            return c
    return None


def _host_porta_de_url(url: str) -> tuple[str, int]:
    """Extrai host e porta de uma URL tipo http://127.0.0.1:8090 -- usado
    tanto pra montar a flag --port do processo quanto pro ping de
    prontidão (_whisper_server_respondendo / _llama_server_respondendo).
    """
    from urllib.parse import urlparse
    p = urlparse(url)
    return p.hostname or "127.0.0.1", p.port or 8090


def _whisper_server_respondendo(url: str, timeout: float = 1.5) -> bool:
    """Ping simples pro whisper-server. HTTPError também conta como 'de
    pé' (respondeu alguma coisa, só não tem handler na raiz "/"); só
    URLError/timeout (conexão recusada, ninguém escutando ali) conta
    como 'não está rodando'. Usado tanto pra decidir SE sobe um servidor
    novo (evita subir em cima de um que você já iniciou na mão, o que
    quebraria por porta em uso) quanto pra saber quando ele terminou de
    carregar o modelo (ver SupervisorWhisper.esperar_pronto).
    """
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _llama_server_respondendo(url: str, timeout: float = 1.5) -> bool:
    """Mesmo padrão de _whisper_server_respondendo, mas contra /health do
    llama-server -- as URLs de config apontam pro endpoint /v1 (formato
    OpenAI), então tira o /v1 e bate em /health, que é o endpoint de
    prontidão que o llama.cpp expõe.
    """
    import urllib.error
    import urllib.request
    base = url.rsplit("/v1", 1)[0].rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/health", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


class SupervisorWhisper:
    """Sobe o whisper-server (whisper.cpp) como subprocesso, mesmo padrão
    do Supervisor (bridge.js) logo abaixo.

    ACHADO REAL que motivou isto existir: ver WhisperCppServerSTT em
    stt.py -- o backend antigo (subprocess.run por CHAMADA, um processo
    novo a cada frase) recarregava o modelo GGML inteiro do disco em
    toda transcrição, deixando o tempo de STT praticamente constante
    (~3s) independente da duração do áudio. Servidor residente resolve
    isso; esta classe só cuida de nascer/morrer junto com a EVA em vez
    de exigir uma janela de terminal separada pra isso, mesmo espírito
    do Supervisor do bridge.js.

    Só sobe se: (1) EVA_STT_WHISPER_CPP_SERVER_EXE está configurado, e
    (2) nada já está respondendo em EVA_STT_WHISPER_CPP_URL -- assim, se
    você preferir continuar subindo na mão num terminal separado (mesmo
    padrão que o LM Studio já usa), basta deixar a variável de exe vazia
    ou já ter o servidor rodando antes: o executar() abaixo detecta e
    não tenta subir outro em cima (quebraria por porta já em uso).
    """

    def __init__(self, exe: str, modelo: str, url: str, threads: int | None = None):
        self.exe = exe
        self.modelo = modelo
        self.url = url
        self.threads = threads
        self.proc: subprocess.Popen | None = None

    def iniciar(self) -> None:
        host, porta = _host_porta_de_url(self.url)
        cmd = [self.exe, "-m", self.modelo, "--host", host, "--port", str(porta)]
        if self.threads:
            cmd += ["-t", str(self.threads)]

        print(f"[whisper] iniciando whisper-server em {host}:{porta}...")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # mesmo motivo do Supervisor abaixo: sem isso, log com
            # caractere fora de cp1252/cp850 derruba o processo inteiro
            # no Windows.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    async def esperar_pronto(self, timeout: float = 60.0) -> bool:
        """Espera o servidor começar a responder via polling, em vez de
        um sleep fixo -- carregar large-v3-turbo pode levar bem mais que
        os 2s que bastam pro bridge.js (que é só Node subindo, sem
        modelo nenhum pra carregar pra GPU). Sleep cego arriscaria a
        primeira transcrição real chegar antes do modelo terminar de
        carregar; polling evita essa corrida sem chutar um número fixo.
        """
        loop = asyncio.get_running_loop()
        inicio = loop.time()
        while loop.time() - inicio < timeout:
            if self.proc and self.proc.poll() is not None:
                print(f"[whisper] processo encerrou cedo (código {self.proc.poll()}) "
                      f"-- confira o log [whisper] acima")
                return False
            if await loop.run_in_executor(None, _whisper_server_respondendo, self.url):
                return True
            await asyncio.sleep(0.5)
        return False

    async def repassar_saida(self) -> None:
        """Mostra o log do whisper-server com prefixo -- mesmo padrão do
        Supervisor.repassar_saida abaixo, pra distinguir dos outros
        processos (Python principal, bridge.js, llama-servers) no mesmo
        console.
        """
        if not self.proc or not self.proc.stdout:
            return
        loop = asyncio.get_running_loop()
        while True:
            linha = await loop.run_in_executor(None, self.proc.stdout.readline)
            if not linha:
                break
            print(f"[whisper] {linha.rstrip()}")
        codigo = self.proc.poll()
        print(f"[whisper] processo encerrou (código {codigo})")

    def parar(self) -> None:
        if not self.proc:
            return
        print("[whisper] encerrando servidor...")
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


class SupervisorLlama:
    """Sobe um llama-server como subprocesso -- mesmo padrão de
    SupervisorWhisper. Reutilizável pra qualquer modelo (conversa,
    decisão/visão, embeddings): só muda exe/modelo/url/flags/mmproj.

    `nome` dá um prefixo de log diferente pra cada instância -- sem isso,
    as três (conversa/decisão/embeddings) ficariam indistinguíveis no
    console, todas dizendo "[llama]".

    Só sobe se: (1) o `server_exe` correspondente está configurado, e
    (2) nada já está respondendo na URL -- mesma dupla de condições do
    SupervisorWhisper, mesmo motivo (não subir em cima de um servidor que
    você já deixou rodando manualmente pra testar outro modelo/flags).
    """

    def __init__(self, exe: str, modelo: str, url: str, flags: list[str] | None = None,
                 mmproj: str = "", nome: str = "llama"):
        self.exe = exe
        self.modelo = modelo
        self.url = url
        self.flags = flags or []
        self.mmproj = mmproj
        self.nome = nome
        self.proc: subprocess.Popen | None = None

    def iniciar(self) -> None:
        host, porta = _host_porta_de_url(self.url)
        cmd = [self.exe, "-m", self.modelo, "--host", host, "--port", str(porta), *self.flags]
        if self.mmproj:
            cmd += ["--mmproj", self.mmproj]

        print(f"[{self.nome}] iniciando llama-server em {host}:{porta}...")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    async def esperar_pronto(self, timeout: float = 120.0) -> bool:
        """Timeout maior que o do whisper (60s): um 12B em Q4 subindo pra
        VRAM pode levar mais que o modelo de STT. Mesma lógica de polling
        em vez de sleep fixo -- ver SupervisorWhisper.esperar_pronto.
        """
        loop = asyncio.get_running_loop()
        inicio = loop.time()
        while loop.time() - inicio < timeout:
            if self.proc and self.proc.poll() is not None:
                print(f"[{self.nome}] processo encerrou cedo (código {self.proc.poll()}) "
                      f"-- confira o log [{self.nome}] acima")
                return False
            if await loop.run_in_executor(None, _llama_server_respondendo, self.url):
                return True
            await asyncio.sleep(0.5)
        return False

    async def repassar_saida(self) -> None:
        if not self.proc or not self.proc.stdout:
            return
        loop = asyncio.get_running_loop()
        while True:
            linha = await loop.run_in_executor(None, self.proc.stdout.readline)
            if not linha:
                break
            print(f"[{self.nome}] {linha.rstrip()}")
        codigo = self.proc.poll()
        print(f"[{self.nome}] processo encerrou (código {codigo})")

    def parar(self) -> None:
        if not self.proc:
            return
        print(f"[{self.nome}] encerrando servidor...")
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


def diagnostico() -> bool:
    """Checa tudo que o modo Discord precisa. Retorna True se está pronto."""
    cfg = carregar_config()
    ok = True

    print(f"\n{_cor('EVA no Discord — diagnóstico', VERDE)}\n")

    # Node
    node = shutil.which("node")
    if node:
        v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        print(f"  node:        {v}")
    else:
        print(f"  node:        {_cor('NÃO ENCONTRADO', AMARELO)}")
        print("               instale em https://nodejs.org")
        ok = False

    # bridge.js
    bridge = achar_bridge()
    if bridge:
        print(f"  bridge.js:   {bridge}")
        node_modules = bridge.parent / "node_modules"
        if node_modules.exists():
            print(f"  dependências: instaladas")
        else:
            print(f"  dependências: {_cor('FALTAM', AMARELO)}")
            print(f"               cd {bridge.parent} && npm install "
                  "discord.js @discordjs/voice ws prism-media libsodium-wrappers")
            ok = False
    else:
        print(f"  bridge.js:   {_cor('NÃO ENCONTRADO', AMARELO)}")
        print("               coloque na raiz do projeto, ou defina EVA_BRIDGE_JS")
        ok = False

    # ffmpeg
    if shutil.which("ffmpeg"):
        print(f"  ffmpeg:      ok")
    else:
        print(f"  ffmpeg:      {_cor('NÃO ENCONTRADO', AMARELO)} "
              "(necessário para converter o áudio do TTS)")
        ok = False

    # token
    if cfg.discord.token:
        print(f"  DISCORD_TOKEN: presente")
    else:
        print(f"  DISCORD_TOKEN: {_cor('AUSENTE', AMARELO)}")
        ok = False

    # STT / TTS
    print(f"  GROQ_API_KEY:  {'presente' if cfg.voz.stt_chave else _cor('AUSENTE', AMARELO)}")

    try:
        from ..voice.tts import criar_tts
        t = criar_tts(cfg.voz.tts_backend or None, cfg.voz.tts_idioma)
        print(f"  TTS:         {t.nome} (idioma {cfg.voz.tts_idioma})")
    except Exception as e:
        print(f"  TTS:         {_cor('INDISPONÍVEL', AMARELO)}")
        print(f"               {str(e).splitlines()[0]}")
        ok = False

    # websockets
    try:
        import websockets  # noqa: F401
        print(f"  websockets:  ok")
    except ImportError:
        print(f"  websockets:  {_cor('FALTA', AMARELO)} — pip install websockets")
        ok = False

    # modelo de conversa
    from ..orchestrator import EVA
    eva = EVA(cfg)
    conectado = eva.llm.disponivel()
    print(f"  modelo:      {'conectado' if conectado else _cor('NÃO CONECTADO', AMARELO)} "
          f"({cfg.llm.base_url})")
    eva.fechar()
    if not conectado:
        ok = False

    # modelo de decisão/visão -- mesmo servidor atende os dois papéis
    print(f"  decisão/visão: "
          f"{'respondendo' if _llama_server_respondendo(cfg.decisao.base_url) else _cor('NÃO RESPONDE', AMARELO)} "
          f"({cfg.decisao.base_url})")

    # embeddings
    if cfg.memoria.usar_embeddings:
        print(f"  embeddings:  "
              f"{'respondendo' if _llama_server_respondendo(cfg.memoria.embeddings_base_url) else _cor('NÃO RESPONDE', AMARELO)} "
              f"({cfg.memoria.embeddings_base_url})")

    print()
    print(_cor("  tudo pronto" if ok else "  faltam itens acima", VERDE if ok else AMARELO))
    print()
    return ok


class Supervisor:
    """Sobe o bridge Node como subprocesso e o cliente Python junto."""

    def __init__(self, caminho_bridge: Path, porta: int = 8765):
        self.caminho = caminho_bridge
        self.porta = porta
        self.proc: subprocess.Popen | None = None

    def iniciar_bridge(self) -> None:
        env = os.environ.copy()
        env["VOICE_BRIDGE_PORT"] = str(self.porta)

        print(f"[node] iniciando {self.caminho.name}...")
        self.proc = subprocess.Popen(
            ["node", str(self.caminho)],
            cwd=str(self.caminho.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # O Node escreve UTF-8 (o bridge.js usa emoji nos logs — 🔌, 🔑,
            # etc). Sem `encoding=` explícito, o Popen usa
            # locale.getpreferredencoding(), que no Windows é a codepage do
            # console (cp1252/cp850, não UTF-8). O primeiro emoji cortava a
            # leitura no meio de um byte multibyte e derrubava o supervisor
            # inteiro com UnicodeDecodeError -- levando o cliente Python
            # junto, porque os dois rodam no mesmo asyncio.gather.
            # errors="replace" é rede de segurança adicional: byte
            # remanescente vira "?" no log em vez de crashar o processo de
            # novo se algo mais escapar no futuro.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    async def repassar_saida(self) -> None:
        """Mostra o log do Node com prefixo, para distinguir dos dois lados."""
        if not self.proc or not self.proc.stdout:
            return
        loop = asyncio.get_running_loop()
        while True:
            linha = await loop.run_in_executor(None, self.proc.stdout.readline)
            if not linha:
                break
            print(f"[node] {linha.rstrip()}")
        codigo = self.proc.poll()
        print(f"[node] processo encerrou (código {codigo})")

    def parar(self) -> None:
        if not self.proc:
            return
        print("[node] encerrando bridge...")
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


async def executar(so_python: bool, porta: int) -> None:
    cfg = carregar_config()
    cliente = ClienteBridge(cfg, url=f"ws://localhost:{porta}")

    # Os llama-server sobem AGORA (chamada não-bloqueante, só cria o
    # subprocesso) e só são ESPERADOS lá embaixo, depois de bridge.js e
    # whisper -- assim os processos carregam em paralelo de verdade
    # (cada um é um subprocesso do SO, continua carregando enquanto este
    # coroutine espera outra coisa), em vez de empilhar o tempo de espera
    # de cada um em sequência. O de conversa sobe primeiro por ser o
    # maior, quem mais se beneficia da folga extra pra carregar.
    supervisor_llama = None
    if cfg.llm.server_exe and not _llama_server_respondendo(cfg.llm.base_url):
        supervisor_llama = SupervisorLlama(
            exe=cfg.llm.server_exe, modelo=cfg.llm.server_modelo,
            url=cfg.llm.base_url, flags=cfg.llm.server_flags,
            nome="llama",
        )
        supervisor_llama.iniciar()

    # Decisão -- puro-texto por padrão desde 02/09/2026 (visão saiu para
    # servidor próprio, ver supervisor_visao abaixo). server_mmproj só é
    # passado se você tiver preenchido EVA_DECISAO_SERVER_MMPROJ no .env
    # (compatibilidade com quem ainda quer o esquema antigo combinado).
    supervisor_decisao = None
    if cfg.decisao.server_exe and not _llama_server_respondendo(cfg.decisao.base_url):
        supervisor_decisao = SupervisorLlama(
            exe=cfg.decisao.server_exe, modelo=cfg.decisao.server_modelo,
            url=cfg.decisao.base_url, flags=cfg.decisao.server_flags,
            mmproj=cfg.decisao.server_mmproj,
            nome="decisao",
        )
        supervisor_decisao.iniciar()

    # Visão -- servidor PRÓPRIO desde 02/09/2026 (ver docstring de
    # VisaoConfig.base_url pro achado real de contenção de GPU que
    # motivou a separação: o tick de fundo da visão disputando a mesma
    # GPU que o modelo conversacional derrubou o throughput dele à
    # metade num turno real). Só sobe se EVA_VISAO_SERVER_EXE estiver
    # preenchido -- sem isso, cfg.visao.base_url precisa já estar
    # respondendo (subido na mão, ou apontando pro mesmo servidor de
    # decisão se você preencher EVA_VISAO_URL=EVA_DECISAO_URL no .env,
    # voltando ao esquema antigo).
    supervisor_visao = None
    if cfg.visao.server_exe and not _llama_server_respondendo(cfg.visao.base_url):
        supervisor_visao = SupervisorLlama(
            exe=cfg.visao.server_exe, modelo=cfg.visao.server_modelo,
            url=cfg.visao.base_url, flags=cfg.visao.server_flags,
            mmproj=cfg.visao.server_mmproj,
            nome="visao",
        )
        supervisor_visao.iniciar()

    # Embeddings -- modelo minúsculo (nomic-embed, ~84MB), mas isolado do
    # resto pra tirar a última dependência do LM Studio.
    supervisor_embeddings = None
    if (cfg.memoria.usar_embeddings and cfg.memoria.embeddings_server_exe
            and not _llama_server_respondendo(cfg.memoria.embeddings_base_url)):
        supervisor_embeddings = SupervisorLlama(
            exe=cfg.memoria.embeddings_server_exe, modelo=cfg.memoria.embeddings_server_modelo,
            url=cfg.memoria.embeddings_base_url, flags=cfg.memoria.embeddings_server_flags,
            nome="embeddings",
        )
        supervisor_embeddings.iniciar()

    supervisor = None
    if not so_python:
        bridge = achar_bridge()
        if not bridge:
            raise SystemExit(
                "bridge.js não encontrado.\n"
                "Coloque-o na raiz do projeto, ou defina EVA_BRIDGE_JS=/caminho/bridge.js\n"
                "Se o bridge já está rodando em outro terminal, use --so-python."
            )
        if not shutil.which("node"):
            raise SystemExit("Node.js não encontrado. Instale em https://nodejs.org")

        supervisor = Supervisor(bridge, porta)
        supervisor.iniciar_bridge()
        # dá um tempo para o WebSocket do Node subir antes de tentar conectar
        await asyncio.sleep(2)

    # whisper-server: só sobe se configurado (exe presente) E nada já
    # estiver respondendo na URL configurada -- ver docstring de
    # SupervisorWhisper pro motivo dessas duas condições.
    supervisor_whisper = None
    if (cfg.voz.stt_backend == "whisper_cpp" and cfg.voz.stt_whisper_cpp_server_exe
            and not _whisper_server_respondendo(cfg.voz.stt_whisper_cpp_url)):
        supervisor_whisper = SupervisorWhisper(
            exe=cfg.voz.stt_whisper_cpp_server_exe,
            modelo=cfg.voz.stt_whisper_cpp_modelo,
            url=cfg.voz.stt_whisper_cpp_url,
            threads=cfg.voz.stt_whisper_cpp_threads,
        )
        supervisor_whisper.iniciar()
        pronto = await supervisor_whisper.esperar_pronto()
        if not pronto:
            print("[whisper] AVISO: servidor não respondeu a tempo -- STT vai cair "
                  "pra Groq (ver criar_stt em stt.py) até você conferir o log acima.")

    # Só agora espera os llama-server ficarem prontos -- por causa da
    # ordem acima, eles já vêm carregando há um tempo (o de bridge.js +
    # o de whisper), então a espera efetiva aqui costuma ser bem menor
    # que os timeouts individuais.
    for sup in (supervisor_llama, supervisor_decisao, supervisor_visao, supervisor_embeddings):
        if sup is None:
            continue
        pronto = await sup.esperar_pronto()
        if not pronto:
            print(f"[{sup.nome}] AVISO: servidor não respondeu a tempo -- "
                  f"confira o log [{sup.nome}] acima antes de usar a EVA.")

    tarefas = [asyncio.create_task(cliente.rodar())]
    if supervisor:
        tarefas.append(asyncio.create_task(supervisor.repassar_saida()))
    if supervisor_whisper:
        tarefas.append(asyncio.create_task(supervisor_whisper.repassar_saida()))
    if supervisor_llama:
        tarefas.append(asyncio.create_task(supervisor_llama.repassar_saida()))
    if supervisor_decisao:
        tarefas.append(asyncio.create_task(supervisor_decisao.repassar_saida()))
    if supervisor_visao:
        tarefas.append(asyncio.create_task(supervisor_visao.repassar_saida()))
    if supervisor_embeddings:
        tarefas.append(asyncio.create_task(supervisor_embeddings.repassar_saida()))

    try:
        await asyncio.gather(*tarefas)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tarefas:
            t.cancel()
        if supervisor:
            supervisor.parar()
        if supervisor_whisper:
            supervisor_whisper.parar()
        if supervisor_llama:
            supervisor_llama.parar()
        if supervisor_decisao:
            supervisor_decisao.parar()
        if supervisor_visao:
            supervisor_visao.parar()
        if supervisor_embeddings:
            supervisor_embeddings.parar()
        cliente.fechar()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eva.discord")
    p.add_argument("--so-python", action="store_true",
                   help="não sobe o bridge (útil quando ele já está rodando)")
    p.add_argument("--porta", type=int,
                   default=int(os.environ.get("VOICE_BRIDGE_PORT", "8765")))
    p.add_argument("--diagnostico", action="store_true")
    args = p.parse_args(argv)

    if args.diagnostico:
        return 0 if diagnostico() else 1

    # Antes de qualquer coisa imprimir: a linha de erro mais útil de
    # todas costuma ser a primeira (modelo que não carregou, porta em
    # uso, .env faltando), e ela sai antes dos supervisores subirem.
    caminho_log = iniciar_log_em_arquivo()
    if caminho_log:
        print(f"[log] gravando esta execução em {caminho_log}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def encerrar(*_):
        for t in asyncio.all_tasks(loop):
            t.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, encerrar)
    except (NotImplementedError, AttributeError):
        pass  # Windows não suporta add_signal_handler

    try:
        loop.run_until_complete(executar(args.so_python, args.porta))
    except KeyboardInterrupt:
        print("\nencerrando...")
    finally:
        loop.close()
        # O `finally` também roda quando a saída é abrupta (Ctrl+C no meio
        # de uma call, que é como a maioria das sessões termina aqui) --
        # sem este flush, as últimas linhas ficariam no buffer e sumiriam,
        # e são justamente as que dizem por que você saiu.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if caminho_log:
            print(f"[log] execução salva em {caminho_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())