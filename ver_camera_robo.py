#!/usr/bin/env python3
"""
ver_camera_robo.py -- visualizador ao vivo do stream de vídeo do robô.

Requer opencv e numpy no ambiente que rodar este script (mesmas
bibliotecas que o lado do robô já usa pra codificar os frames). Se não
tiver: pip install opencv-python numpy (sem --break-system-packages se
estiver num venv).

Conecta em DUAS portas, com propósitos diferentes:
- porta de VÍDEO (padrão 8000): recebe o stream de frames JPEG. Segundo
  server.py/tcp_server.py, esta porta é write-only do lado do
  servidor -- ele nunca lê nada que o cliente mandar aqui, só escreve.
- porta de COMANDO (padrão 5000): usada só pra perguntar qual câmera
  está ativa (pra legendar a janela) e pra trocar de câmera (tecla V).
  Trocar câmera NUNCA poderia acontecer pela porta de vídeo, mesmo que
  quiséssemos -- ela é write-only por desenho.

Dentro da janela de vídeo:
  V         trocar câmera (USB <-> PICAM)
  Q ou ESC  sair

Uso:
    python3 ver_camera_robo.py --host 192.168.100.30
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERRO: Precisa de opencv e numpy neste ambiente:")
    print("   pip install opencv-python numpy")
    sys.exit(1)


# ---------------------------------------------------------------- comando

class ClienteComando:
    """Réplica mínima do protocolo de comando -- só o suficiente pra
    perguntar o estado e trocar câmera. De propósito NÃO reaproveita
    ClienteManual (controle_manual_robo.py) pra este visualizador poder
    rodar sozinho, sem depender de outro arquivo do projeto."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self.arquivo = None
        self._seq = 0
        self._lock = threading.Lock()

    def _garantir_conectado(self) -> bool:
        if self.sock is not None:
            return True
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=3)
            self.arquivo = self.sock.makefile("rwb")
            return True
        except OSError:
            self.sock = None
            self.arquivo = None
            return False

    def enviar(self, cmd: str, params: dict | None = None) -> dict:
        with self._lock:
            if not self._garantir_conectado():
                return {"ok": False, "erro": "sem_conexao"}
            self._seq += 1
            envelope = {
                "type": "command", "source": "manual", "priority": 0,
                "seq": self._seq, "ttl_ms": 2000, "cmd": cmd,
                "params": params or {}, "sent_ts": time.time(),
            }
            try:
                self.arquivo.write((json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8"))
                self.arquivo.flush()
                linha = self.arquivo.readline()
                if not linha:
                    raise ConnectionError("conexão fechada pelo servidor")
                return json.loads(linha.decode("utf-8"))
            except (OSError, ConnectionError, json.JSONDecodeError) as e:
                self.sock = None
                self.arquivo = None
                return {"ok": False, "erro": "falha_conexao", "detalhe": str(e)[:150]}


# ---------------------------------------------------------------- vídeo

def _recv_exato(sock: socket.socket, n: int) -> bytes | None:
    """Lê exatamente n bytes, ou None se a conexão fechar no meio."""
    dados = b""
    while len(dados) < n:
        pedaco = sock.recv(n - len(dados))
        if not pedaco:
            return None
        dados += pedaco
    return dados


def ler_frame(sock: socket.socket) -> bytes | None:
    """Protocolo da porta de vídeo (ver eva_command_server._video_loop):
    4 bytes little-endian com o tamanho do frame, depois os bytes JPEG."""
    cabecalho = _recv_exato(sock, 4)
    if cabecalho is None:
        return None
    (tamanho,) = struct.unpack('<L', cabecalho)
    if tamanho <= 0 or tamanho > 10_000_000:  # sanity check -- nunca deveria vir isto
        return None
    return _recv_exato(sock, tamanho)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("EVA_ROBOT_HOST"))
    parser.add_argument("--video-port", type=int,
                         default=int(os.environ.get("EVA_ROBOT_VIDEO_PORT", "8000") or 8000))
    parser.add_argument("--command-port", type=int,
                         default=int(os.environ.get("EVA_ROBOT_PORT", "5000") or 5000))
    args = parser.parse_args()

    if not args.host:
        print("ERRO: Host não informado. Use --host <ip> ou defina EVA_ROBOT_HOST no ambiente.")
        print("   Lembrete: 127.0.0.1/localhost não funciona -- use o IP real do Pi na rede.")
        return 1

    print(f"Conectando no vídeo em {args.host}:{args.video_port}...")
    try:
        sock_video = socket.create_connection((args.host, args.video_port), timeout=5)
        sock_video.settimeout(5.0)  # detecta stream travado em vez de bloquear pra sempre
    except OSError as e:
        print(f"ERRO: Não consegui conectar na porta de vídeo: {e}")
        print("   eva_command_server.py está rodando? Alguém mais já está vendo o vídeo "
              "(max_clients pode estar ocupado)?")
        return 1
    print("Conectado. Janela abrindo -- V pra trocar câmera, Q/ESC pra sair.")

    comando = ClienteComando(args.host, args.command_port)

    janela = "EVA Robot -- camera"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    camera_atual = "?"
    ultima_checagem_estado = 0.0

    try:
        while True:
            try:
                frame_bytes = ler_frame(sock_video)
            except socket.timeout:
                print("Sem frame novo em 5s -- stream pode ter travado. Reconectando...")
                frame_bytes = None

            if frame_bytes is None:
                sock_video.close()
                time.sleep(1)
                try:
                    sock_video = socket.create_connection((args.host, args.video_port), timeout=5)
                    sock_video.settimeout(5.0)
                    print("Reconectado.")
                except OSError as e:
                    print(f"ERRO: Falha ao reconectar: {e} -- tentando de novo em 2s")
                    time.sleep(2)
                continue

            frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue  # frame corrompido pontual -- só pula, não derruba o visualizador

            agora = time.time()
            if agora - ultima_checagem_estado > 2.0:
                ultima_checagem_estado = agora
                r = comando.enviar("get_state")
                if r.get("ok"):
                    camera_atual = (r.get("estado", {}).get("camera", {})
                                     .get("active_camera") or "?").upper()

            cv2.putText(frame, f"camera: {camera_atual}   V=trocar  Q=sair",
                        (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(janela, frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord('q'), 27):  # 27 = ESC
                break
            elif tecla == ord('v'):
                print("[trocar câmera]")
                r = comando.enviar("camera_switch")
                if r.get("ok"):
                    camera_atual = (r.get("camera_ativa") or "?").upper()
                    print(f"  câmera ativa agora: {camera_atual}")
                    ultima_checagem_estado = agora  # evita reconsultar de novo já no próximo frame
                else:
                    print(f"  falhou: {r.get('erro')} -- {r.get('detalhe')}")

    finally:
        sock_video.close()
        cv2.destroyAllWindows()
        print("Encerrado.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
