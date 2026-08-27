"""
robot_video_client.py -- cliente do stream de vídeo do eva_command_server.py.

Porta SEPARADA da de comando (robot_client.py fala com a porta 5000; isto
fala com a 8000, video_port). Protocolo binário simples, não JSON: o
servidor (eva_command_server._video_loop) manda, continuamente enquanto
houver cliente conectado, um quadro JPEG por vez, cada um prefixado por 4
bytes little-endian com o tamanho (struct.pack('<L', len(frame_data)) +
frame_data) -- sem delimitador entre quadros, o tamanho É o delimitador.

SÓ O QUADRO MAIS RECENTE IMPORTA -- este cliente não bufferiza histórico
nenhum, só guarda o último quadro lido em `ultimo_frame`. Quem consome
isto (robot_tools.robo_ver/robo_olhar) quer "o que a câmera está vendo
agora", nunca um replay do que passou.

LEITURA CROSS-THREAD SEM LOCK: `ultimo_frame` é escrito pela task de
leitura (rodando no event loop dedicado do robô, ver
robot_tools._iniciar_thread_robo) e lido de QUALQUER outra thread (a
ferramenta síncrona que quer o quadro atual). Isso é seguro sem lock pelo
mesmo motivo que `_em_call` (robot_tools.py) é: é sempre uma troca de
REFERÊNCIA a um objeto imutável (bytes), nunca uma mutação in-place --
CPython garante atomicidade nisso. Quem lê no meio de uma troca vê o
quadro velho ou o novo, nunca um estado parcial.

ONDE ISSO DEVE FICAR: ao lado de robot_client.py, em
eva/integrations/robot_video_client.py -- ajuste o import em
robot_tools.py se o layout real for outro.
"""

from __future__ import annotations

import asyncio
import struct
import time


class ClienteVideoRobo:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000,
                 timeout_conexao: float = 3.0, espera_entre_tentativas: float = 2.0):
        self.host = host
        self.port = port
        self.timeout_conexao = timeout_conexao
        self.espera_entre_tentativas = espera_entre_tentativas

        self.ultimo_frame: bytes | None = None
        # Sem timestamp, quem lê não tem como distinguir "quadro de
        # agora" de "quadro de três minutos atrás, guardado depois da
        # conexão cair" -- as duas coisas são bytes JPEG perfeitamente
        # válidos. Ver robot_tools.obter_quadro_camera, que usa isto pra
        # recusar quadro velho em vez de deixar o modelo de visão
        # descrever com confiança uma cena que não existe mais.
        self.ultimo_frame_ts: float = 0.0
        self.conectado = False
        self._falhas_conexao_consecutivas = 0

    async def rodar(self) -> None:
        """Loop pra sempre: conecta, lê quadros até a conexão cair,
        reconecta. Pensado pra rodar como task de fundo (ver
        _loop.create_task em robot_tools._iniciar_thread_robo), no mesmo
        espírito de _ciclo_heartbeat -- nunca deve parar sozinho, e
        nunca deve derrubar a thread por causa de um soluço de rede."""
        while True:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port),
                    timeout=self.timeout_conexao,
                )
                self.conectado = True
                self._falhas_conexao_consecutivas = 0
                print(f"[robo-video] conectado em {self.host}:{self.port}")
                try:
                    await self._ler_quadros(reader)
                finally:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass
            except (OSError, ConnectionError, asyncio.TimeoutError, asyncio.IncompleteReadError) as e:
                self.conectado = False
                self._falhas_conexao_consecutivas += 1
                # Só os primeiros -- depois disso, sem robô ligado, isto
                # tentaria pra sempre a cada espera_entre_tentativas
                # segundos, poluindo o terminal com a mesma linha. O
                # comando (robot_client.ClienteRobo) já tem o mesmo
                # padrão de "desiste de logar, não de tentar".
                if self._falhas_conexao_consecutivas <= 3:
                    print(f"[robo-video] sem conexão com {self.host}:{self.port} ({e})")
            except Exception as e:
                self.conectado = False
                print(f"[robo-video] erro inesperado: {e}")

            # Primeira falha reconecta quase na hora; só depois recua pro
            # intervalo normal. A queda mais comum aqui é NO MEIO de um
            # quadro (IncompleteReadError, "N bytes read on a total of M
            # expected") e se recupera na tentativa seguinte -- esperar
            # os 2s fixos deixava o último quadro envelhecer até cair na
            # checagem de idade de obter_quadro_camera, e robo_ver
            # respondia "video_parado" por uma falha que já tinha passado.
            espera = (0.2 if self._falhas_conexao_consecutivas <= 1
                      else self.espera_entre_tentativas)
            await asyncio.sleep(espera)

    async def _ler_quadros(self, reader: asyncio.StreamReader) -> None:
        while True:
            cabecalho = await reader.readexactly(4)
            (tamanho,) = struct.unpack("<L", cabecalho)
            dados = await reader.readexactly(tamanho)
            self.ultimo_frame = dados
            self.ultimo_frame_ts = time.monotonic()