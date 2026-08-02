"""
Sobe a EVA no Discord: bridge Node + cliente Python.

O bridge cuida do Discord (voz e texto) e este processo cuida do cérebro.
A separação existe porque a stack de voz do discord.py é problemática --
o discord.js com @discordjs/voice é o caminho maduro.

Uso:
    python -m eva.integrations.discord           # sobe os dois
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

    # modelo
    from ..orchestrator import EVA
    eva = EVA(cfg)
    conectado = eva.llm.disponivel()
    print(f"  modelo:      {'conectado' if conectado else _cor('NÃO CONECTADO', AMARELO)} "
          f"({cfg.llm.base_url})")
    eva.fechar()
    if not conectado:
        ok = False

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

    tarefas = [asyncio.create_task(cliente.rodar())]
    if supervisor:
        tarefas.append(asyncio.create_task(supervisor.repassar_saida()))

    try:
        await asyncio.gather(*tarefas)
    except asyncio.CancelledError:
        pass
    finally:
        for t in tarefas:
            t.cancel()
        if supervisor:
            supervisor.parar()
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
    return 0


if __name__ == "__main__":
    sys.exit(main())