"""
launcher.py -- liga/desliga a EVA com um clique, sem precisar digitar
`python -m eva.integrations.discord` no terminal toda vez.

Fica FORA do pacote eva/ de propósito (mesmo lugar do seed_historia_eva.py)
-- este processo existe pra SUBIR a EVA, então não pode viver dentro do
processo que ele mesmo sobe.

Não reimplementa nada: sobe exatamente `python -m eva.integrations.discord`
como subprocesso -- o mesmo comando que você já digitava, só que a partir
de um botão. Esse módulo já sabe subir o bridge.js sozinho e limpar os
dois processos direito no fechamento (ver Supervisor.parar() em
eva/integrations/discord.py); o launcher não duplica essa lógica, só
aciona ela.

DESLIGAR NO WINDOWS -- por que não é só proc.terminate():
Popen.terminate() no Windows manda TerminateProcess, que mata o processo
Python na hora, sem deixar o bloco `finally` de discord.py rodar -- e é
esse finally que mata o bridge.js (Node) junto. Sem isso, desligar pelo
launcher deixaria um bridge.js orfão rodando escondido.
O caminho certo é CTRL_BREAK_EVENT: só funciona se o processo foi criado
com CREATE_NEW_PROCESS_GROUP (por isso o creationflags ao subir), e o
Python recebe isso como se fosse um Ctrl+Break de verdade -- vira
KeyboardInterrupt do lado de lá, que o próprio discord.py já trata (`except
KeyboardInterrupt`), e o finally roda, matando o bridge.js. Só cai pro
taskkill /T /F (mata a árvore de processos inteira, à força) se isso não
responder a tempo -- rede de segurança, não o caminho normal.

Uso: python launcher.py
Requisito: nenhum além do Python padrão -- tkinter já vem no instalador
oficial do python.org no Windows.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext

# CREATE_NEW_PROCESS_GROUP e CTRL_BREAK_EVENT só existem no Windows --
# getattr com padrão evita erro de import em outro SO, mesmo que este
# launcher seja pensado pro fluxo do Alexandre (Windows + PowerShell).
CRIAR_GRUPO_PROCESSO = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CTRL_BREAK = getattr(signal, "CTRL_BREAK_EVENT", None)

TIMEOUT_DESLIGAMENTO_GRACIOSO = 6  # segundos antes de forçar


class Launcher:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None

        self.janela = tk.Tk()
        self.janela.title("EVA")
        self.janela.geometry("560x380")
        self.janela.protocol("WM_DELETE_WINDOW", self._ao_fechar)

        self.status = tk.StringVar(value="● desligada")
        rotulo_status = tk.Label(
            self.janela, textvariable=self.status,
            font=("Segoe UI", 13, "bold"), fg="#a33",
        )
        rotulo_status.pack(pady=(10, 4))
        self._rotulo_status = rotulo_status

        frame_botoes = tk.Frame(self.janela)
        frame_botoes.pack(pady=4)
        self.btn_ligar = tk.Button(
            frame_botoes, text="Ligar", width=16, command=self.ligar)
        self.btn_ligar.grid(row=0, column=0, padx=6)
        self.btn_desligar = tk.Button(
            frame_botoes, text="Desligar", width=16, command=self.desligar,
            state="disabled")
        self.btn_desligar.grid(row=0, column=1, padx=6)

        self.log = scrolledtext.ScrolledText(
            self.janela, height=18, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    # ------------------------------------------------------------- log

    def _escrever_log(self, linha: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", linha + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _atualizar_status(self, ligada: bool) -> None:
        if ligada:
            self.status.set("● ligada")
            self._rotulo_status.configure(fg="#2a2")
            self.btn_ligar.configure(state="disabled")
            self.btn_desligar.configure(state="normal")
        else:
            self.status.set("● desligada")
            self._rotulo_status.configure(fg="#a33")
            self.btn_ligar.configure(state="normal")
            self.btn_desligar.configure(state="disabled")

    # --------------------------------------------------------- ligar

    def ligar(self) -> None:
        if self.proc is not None:
            return
        self._escrever_log("[launcher] iniciando EVA (python -m eva.integrations.discord)...")
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "eva.integrations.discord"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CRIAR_GRUPO_PROCESSO,
            )
        except Exception as e:
            self._escrever_log(f"[launcher] falha ao iniciar: {e}")
            return

        self._atualizar_status(ligada=True)
        threading.Thread(target=self._ler_saida, daemon=True).start()

    def _ler_saida(self) -> None:
        """Roda em thread separada -- ler stdout do subprocesso é
        bloqueante, e travar o event loop do Tkinter travaria a janela
        inteira até a EVA desligar."""
        proc = self.proc
        if not proc or not proc.stdout:
            return
        for linha in proc.stdout:
            self.janela.after(0, self._escrever_log, linha.rstrip())
        codigo = proc.wait()
        self.janela.after(0, self._ao_processo_terminar, codigo)

    def _ao_processo_terminar(self, codigo: int) -> None:
        self._escrever_log(f"[launcher] EVA encerrou (código {codigo})")
        self.proc = None
        self._atualizar_status(ligada=False)

    # ------------------------------------------------------- desligar

    def desligar(self) -> None:
        if self.proc is None:
            return
        proc = self.proc
        pid = proc.pid
        self._escrever_log("[launcher] desligando EVA...")

        desligou_gracioso = False
        if CTRL_BREAK is not None:
            try:
                proc.send_signal(CTRL_BREAK)
                proc.wait(timeout=TIMEOUT_DESLIGAMENTO_GRACIOSO)
                desligou_gracioso = True
            except subprocess.TimeoutExpired:
                self._escrever_log(
                    "[launcher] não respondeu a tempo, forçando "
                    "(mata a árvore de processos, incluindo o bridge.js)")
            except Exception as e:
                self._escrever_log(f"[launcher] Ctrl+Break falhou ({e}), forçando")

        if not desligou_gracioso:
            try:
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(pid)],
                    capture_output=True, text=True,
                )
            except Exception as e:
                self._escrever_log(f"[launcher] taskkill falhou: {e}")

        # Não atualiza status aqui -- a thread de _ler_saida detecta o
        # processo morto sozinha (proc.wait() retorna) e chama
        # _ao_processo_terminar, seja o desligamento gracioso ou forçado.

    def _ao_fechar(self) -> None:
        if self.proc is not None:
            self.desligar()
        self.janela.destroy()

    def rodar(self) -> None:
        self.janela.mainloop()


if __name__ == "__main__":
    Launcher().rodar()
