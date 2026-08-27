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

        # Reconexão automática em BACKGROUND (heartbeat, checagem de
        # segurança periódica -- ninguém está esperando o resultado na
        # hora) para depois de algumas falhas seguidas, em vez de
        # martelar pra sempre contra um servidor genuinamente fora do
        # ar. Isso é só pra reduzir ruído de log quando o robô fica
        # desligado por um tempo -- com o botão "Conectar agora" no
        # dashboard (robot_tools.conectar_dashboard) e qualquer comando
        # de verdade (voz/decisão) sempre tentando de novo (ver _enviar,
        # parâmetro `background`), a reconexão nunca fica de fato presa:
        # só o retry silencioso em segundo plano é que desiste.
        self._falhas_conexao_consecutivas = 0
        self._max_falhas_antes_de_desistir_em_background = 5
        self._desistiu_reconexao_background = False

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
            self._falhas_conexao_consecutivas = 0
            self._desistiu_reconexao_background = False
            print(f"[robo] conectado em {self.host}:{self.port} (fonte={self.fonte})")
            return True
        except (OSError, ConnectionError, asyncio.TimeoutError) as e:
            self.conectado = False
            self._falhas_conexao_consecutivas += 1
            # Só imprime até desistir em background -- depois disso,
            # continua contando a falha (silenciosamente) mas não repete
            # a mesma linha pra sempre. Uma tentativa manual (botão do
            # dashboard, ou qualquer comando real que ignore o flag de
            # desistência) sempre volta a imprimir, porque reseta o
            # contador em caso de sucesso e é a via que "quebra" o
            # silêncio de propósito.
            if not self._desistiu_reconexao_background:
                print(f"[robo] sem conexão com {self.host}:{self.port} ({e})")
            if self._falhas_conexao_consecutivas >= self._max_falhas_antes_de_desistir_em_background \
                    and not self._desistiu_reconexao_background:
                self._desistiu_reconexao_background = True
                print(f"[robo] desistindo de tentar reconectar sozinho em segundo plano "
                      f"após {self._falhas_conexao_consecutivas} falhas seguidas -- "
                      f"use o botão 'Conectar agora' no dashboard, ou peça pra ela usar o corpo, "
                      f"quando o robô estiver acessível de novo.")
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
                       ttl_ms: int = 300, timeout: float = 5.0,
                       background: bool = False) -> dict:
        """Manda um CommandEnvelope e espera UMA linha de resposta.
        Nunca levanta exceção -- erro de conexão/timeout vira
        {"ok": False, "erro": ...}, mesma regra de toda ferramenta do
        projeto (ver registry.py).

        `background=True` é só pra chamadas automáticas onde ninguém
        está esperando o resultado na hora (heartbeat, checagem de
        segurança periódica) -- essas respeitam
        `_desistiu_reconexao_background` e não tentam `conectar()` de
        novo depois de várias falhas seguidas (ver conectar()), pra não
        martelar sozinhas contra um servidor genuinamente fora do ar.
        `background=False` (default, usado por toda ação real: drive,
        head, stop, estop, reset_estop, e get_state quando chamado por
        ferramenta de verdade ou pelo dashboard) SEMPRE tenta conectar,
        ignorando esse flag -- com o botão 'Conectar agora' existindo,
        não faz sentido nenhum comando de verdade desistir sem tentar só
        porque o retry silencioso em segundo plano já tinha desistido."""
        if not self.conectado:
            if background and self._desistiu_reconexao_background:
                return {"ok": False, "erro": "robo_desconectado",
                        "detalhe": "eva_command_server não está acessível"}
            if not await self.conectar():
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

    async def estado(self, *, background: bool = False) -> dict:
        return await self._enviar("get_state", background=background)

    async def mover(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
                     speed_scale: float | None = None, ttl_ms: int = 500) -> dict:
        params = {"vx": vx, "vy": vy, "vz": vz}
        if speed_scale is not None:
            params["speed_scale"] = speed_scale
        return await self._enviar("drive", params, ttl_ms=ttl_ms)

    async def olhar(self, yaw: int | None = None, pitch: int | None = None,
                     cabeca: int | None = None, cotovelo: int | None = None,
                     smooth: bool = True) -> dict:
        """Eixos confirmados por imagem (ver ferramentas/testar_cabeca.py):

            yaw       canal 0, 0..90    gira a BASE       -- horizontal
            pitch     canal 1, 40..110  levanta/abaixa    -- VERTICAL (o único)
            cabeca    canal 3, 0..117   gira só a CÂMERA  -- horizontal (pan)
            cotovelo  canal 2, 90..180  aproxima/afasta

        Não existe eixo de roll -- nada inclina a imagem de lado. E há
        DOIS eixos horizontais (yaw + cabeça, ~207° somados) contra um
        vertical só.

        `cabeca` e `cotovelo` são novos aqui: o servidor
        (eva_command_server._cmd_head) já aceitava os dois, só não havia
        nada deste lado mandando. O canal 3, apesar do nome, é o servo
        que originalmente movia a GARRA deste braço acrílico; com a
        PiCam parafusada no lugar dela, virou o segundo eixo de pan.

        timeout maior que o padrão de _enviar porque `smooth=True` faz o
        movimento em passos de 2° com 20ms entre eles
        (ArmController._move_smooth): um curso de 90° leva quase um
        segundo POR SERVO, e o _command_loop do Pi processa uma mensagem
        de cada vez."""
        params: dict = {"smooth": smooth}
        for nome, valor in (("yaw", yaw), ("pitch", pitch),
                            ("cabeca", cabeca), ("cotovelo", cotovelo)):
            if valor is not None:
                params[nome] = int(valor)
        return await self._enviar("head", params, timeout=8.0)

    async def trocar_camera(self, tipo: str | None = None) -> dict:
        """tipo: "usb", "picam", ou None pra alternar.

        ttl_ms=0 (TTL desligado) de propósito: a troca fecha e reabre um
        device de vídeo do lado do Pi (camera_manager.switch_camera, com
        sleep de 0.2s mais aquecimento) e pode segurar o _command_loop
        por segundos. Com o TTL padrão de 300ms, o próprio comando
        expiraria na fila se chegasse atrás de qualquer coisa lenta."""
        params = {"tipo": tipo} if tipo else {}
        return await self._enviar("camera_switch", params, ttl_ms=0, timeout=15.0)

    async def parar(self) -> dict:
        return await self._enviar("stop")

    async def heartbeat(self) -> dict:
        """ttl_ms=0 -- heartbeat NUNCA deve expirar na fila.

        Comando expirado não alimenta o watchdog (is_expired retorna
        antes de qualquer processamento), e um `head` com smooth=True
        bloqueia o _command_loop do Pi por ~0.9s POR SERVO. Os
        heartbeats que chegam durante um movimento longo enfileiram e
        voltam todos "comando_expirado" -- uma sequência de gestos
        conseguia estourar o WATCHDOG_TIMEOUT de 5s sozinha e derrubar o
        robô em emergency stop no meio do próprio gesto.

        Heartbeat que ficou na fila continua sendo prova de que o
        cliente estava vivo quando mandou, que é exatamente o que o
        watchdog quer saber. TTL aqui nunca protegeu de nada."""
        return await self._enviar("heartbeat", ttl_ms=0, background=True)

    async def estop(self, motivo: str = "eva solicitou parada de emergência") -> dict:
        return await self._enviar("estop", {"motivo": motivo})

    async def reset_estop(self) -> dict:
        return await self._enviar("reset_estop")


# ============================================================================
# TESTE MANUAL
# ============================================================================

async def _loop_comandos(cliente: ClienteRobo) -> None:
    print("Comandos: mover VX VY VZ | olhar YAW PITCH CABECA COTOVELO | "
          "camera [usb|picam] | parar | estado | estop | heartbeat | sair")
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
            # partes[3] (cabeça) e partes[4] (cotovelo) são lidos agora --
            # antes eram silenciosamente descartados, e "olhar 90 90 115"
            # mandava só yaw e pitch, respondia "ok", e não havia nada na
            # saída indicando que a cabeça tinha sido ignorada.
            def _arg(i):
                return int(partes[i]) if len(partes) > i else None
            r = await cliente.olhar(yaw=_arg(1), pitch=_arg(2),
                                     cabeca=_arg(3), cotovelo=_arg(4))
        elif cmd == "camera":
            r = await cliente.trocar_camera(partes[1] if len(partes) > 1 else None)
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
    import os

    host = os.environ.get("EVA_ROBOT_HOST", "127.0.0.1")
    port = int(os.environ.get("EVA_ROBOT_PORT", "5000"))
    cliente = ClienteRobo(host=host, port=port)
    await _loop_comandos(cliente)
    await cliente.desconectar()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\nencerrando...")