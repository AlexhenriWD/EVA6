"""
robot_client.py -- cliente Python mínimo pro eva_command_server.py do robô.

Mesmo espírito de minecraft_client.py (ver integrations/minecraft_client.py):
ponte fina, sem estado de personalidade/decisão, só garante que o
protocolo funciona contra o servidor de verdade.

Transporte é TCP puro com um objeto JSON por linha, não WebSocket --
porque o lado do robô já tinha server.py/tcp_server.py rodando e testado,
processando uma mensagem de cada vez, em ordem. Por isso aqui basta
send -> recv correlacionados por ORDEM de chegada (um lock garante isso
do lado do cliente), sem precisar de id/Future por mensagem como o
Minecraft precisa (lá o bridge podia responder fora de ordem porque tem
mensagens assíncronas -- chat, evento, snapshot periódico -- chegando
misturadas com resultado de ação; aqui não).

ONDE ISSO DEVE FICAR: ao lado de minecraft_client.py, em
eva/integrations/robot_client.py (mesmo pacote `integrations`) -- ajuste
o import em robot_tools.py se o layout real for outro.
"""

from __future__ import annotations

import asyncio
import json
import time


class ClienteRobo:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000, fonte: str = "eva"):
        self.host = host
        self.port = port
        self.fonte = fonte  # vai no campo "source" de todo CommandEnvelope

        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.conectado = False

        self._seq = 0
        # O servidor processa a fila de comando em ordem, sequencialmente
        # -- então só um comando em voo por vez também do lado do
        # cliente evita qualquer ambiguidade de qual resposta é de qual
        # request.
        self._lock = asyncio.Lock()

    async def conectar(self, timeout: float = 3.0) -> bool:
        """Timeout curto e explícito: sem isso, um robô desligado/
        inacessível na rede faz `open_connection` ficar pendurado por
        bem mais que o razoável (varia por SO), o que é especialmente
        ruim quando quem chama isso é um painel de debug fazendo poll
        periódico -- o painel não pode travar porque o robô está
        desligado."""
        try:
            self.reader, self.writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=timeout)
            self.conectado = True
            print(f"[robo] conectado em {self.host}:{self.port} (fonte={self.fonte})")
            return True
        except (OSError, ConnectionError, asyncio.TimeoutError) as e:
            self.conectado = False
            print(f"[robo] sem conexão com {self.host}:{self.port} ({e})")
            return False

    async def desconectar(self) -> None:
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
        self.conectado = False

    async def _enviar(self, cmd: str, params: dict | None = None, *,
                       ttl_ms: int = 300, timeout: float = 5.0) -> dict:
        """Manda um CommandEnvelope e espera UMA linha de resposta.
        Nunca levanta exceção -- erro de conexão/timeout vira
        {"ok": False, "erro": ...}, mesma regra de toda ferramenta do
        projeto (ver registry.py)."""
        if not self.conectado and not await self.conectar():
            return {"ok": False, "erro": "robo_desconectado",
                     "detalhe": "eva_command_server não está acessível"}

        self._seq += 1
        envelope = {
            "type": "command",
            "source": self.fonte,
            "priority": 0,
            "seq": self._seq,
            "ttl_ms": ttl_ms,
            "cmd": cmd,
            "params": params or {},
            "sent_ts": time.time(),
        }

        async with self._lock:
            try:
                linha_envio = json.dumps(envelope, ensure_ascii=False) + "\n"
                self.writer.write(linha_envio.encode("utf-8"))
                await self.writer.drain()
                linha_resposta = await asyncio.wait_for(self.reader.readline(), timeout=timeout)
            except (asyncio.TimeoutError, OSError, ConnectionError) as e:
                self.conectado = False
                return {"ok": False, "erro": "robo_falha", "detalhe": str(e)[:150]}

        if not linha_resposta:
            self.conectado = False
            return {"ok": False, "erro": "robo_desconectado",
                    "detalhe": "conexão fechada pelo servidor"}

        try:
            return json.loads(linha_resposta.decode("utf-8"))
        except json.JSONDecodeError:
            return {"ok": False, "erro": "resposta_invalida",
                    "detalhe": linha_resposta[:150].decode("utf-8", "replace")}

    # ------------------------------------------------------------ ações

    async def estado(self) -> dict:
        return await self._enviar("get_state")

    async def mover(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                     speed_scale: float | None = None, ttl_ms: int = 500) -> dict:
        params = {"vx": vx, "vy": vy, "vz": vz}
        if speed_scale is not None:
            params["speed_scale"] = speed_scale
        return await self._enviar("drive", params, ttl_ms=ttl_ms)

    async def olhar(self, yaw: int | None = None, pitch: int | None = None,
                     smooth: bool = True) -> dict:
        params: dict = {"smooth": smooth}
        if yaw is not None:
            params["yaw"] = yaw
        if pitch is not None:
            params["pitch"] = pitch
        return await self._enviar("head", params)

    async def parar(self) -> dict:
        return await self._enviar("stop")

    async def heartbeat(self) -> dict:
        return await self._enviar("heartbeat")

    async def estop(self, motivo: str = "eva solicitou parada de emergência") -> dict:
        return await self._enviar("estop", {"motivo": motivo})

    async def reset_estop(self) -> dict:
        return await self._enviar("reset_estop")


# ============================================================================
# TESTE MANUAL
# ============================================================================

async def _loop_comandos(cliente: ClienteRobo) -> None:
    print("Comandos: mover VX VY VZ | olhar YAW PITCH | parar | estado | "
          "estop | heartbeat | sair")
    loop = asyncio.get_event_loop()
    while True:
        linha = await loop.run_in_executor(None, input, "> ")
        partes = linha.strip().split()
        if not partes:
            continue
        cmd = partes[0].lower()

        if cmd == "sair":
            return
        elif cmd == "mover":
            vx, vy, vz = (float(v) for v in partes[1:4]) if len(partes) >= 4 else (0.0, 0.0, 0.0)
            r = await cliente.mover(vx=vx, vy=vy, vz=vz)
        elif cmd == "olhar":
            yaw = int(partes[1]) if len(partes) > 1 else None
            pitch = int(partes[2]) if len(partes) > 2 else None
            r = await cliente.olhar(yaw=yaw, pitch=pitch)
        elif cmd == "parar":
            r = await cliente.parar()
        elif cmd == "estado":
            r = await cliente.estado()
        elif cmd == "estop":
            r = await cliente.estop()
        elif cmd == "heartbeat":
            r = await cliente.heartbeat()
        else:
            print(f"comando desconhecido: {cmd}")
            continue

        print(json.dumps(r, ensure_ascii=False, indent=2))


async def _main() -> None:
    cliente = ClienteRobo()
    await _loop_comandos(cliente)
    await cliente.desconectar()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nencerrando...")
