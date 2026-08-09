"""
minecraft_client.py -- cliente Python mínimo pro minecraft_bridge.js.

DE PROPÓSITO não integra com EVA/memória/decisão ainda -- isso é só pra
confirmar que a ponte em si funciona contra o seu servidor de verdade
antes de construir mais nada em cima. Roda sozinho, mostra tudo que chega
(conectado, snapshot, chat, evento) e deixa mandar ação manual pra testar
cada uma isolada.

Uso:
    python minecraft_client.py
    (dentro, comandos: "mover X Y Z", "minerar BLOCO", "craftar ITEM",
     "falar TEXTO", "snapshot", "sair")

Depois de confirmar que isso funciona contra o seu servidor -- bot conecta,
snapshot faz sentido, ações executam -- a integração de verdade (memória,
estado de tarefa, agente roteado pelo Cérebro) entra por cima disso, não
substituindo.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

try:
    import websockets
except ImportError:
    websockets = None


class ClienteMinecraft:
    def __init__(self, url: str = "ws://localhost:8766"):
        self.url = url
        self.ws = None
        # Correlaciona ação enviada com o acao_resultado que volta --
        # mesmo id, resolve a mesma Future.
        self._pendentes: dict[str, asyncio.Future] = {}
        self.conectado = False
        # Enquanto True, snapshot automático (a cada 4s, vindo do bridge
        # sozinho) imprime na hora que chega -- ao mesmo tempo que você
        # digita um comando, o que embaralha os dois no terminal. Modo de
        # testar ação (_loop_comandos) desliga isso; "snapshot" continua
        # funcionando sob demanda independente do estado disso aqui.
        self.mostrar_snapshot_automatico = True
        # Último snapshot recebido (periódico ou sob demanda) -- existe
        # porque o acao_resultado de "obter_snapshot" NÃO carrega o dado
        # em si, só confirma que foi mandado; o dado de verdade chega
        # separado, como mensagem type=snapshot. Ferramentas que a EVA usa
        # pra decidir (minecraft_tools.py) leem daqui, sem precisar
        # correlacionar id de ação nenhum -- o periódico já mantém isso
        # atualizado a cada poucos segundos de qualquer jeito.
        self.ultimo_snapshot: dict | None = None

    async def rodar(self) -> None:
        if websockets is None:
            raise SystemExit("websockets não instalado. Rode: pip install websockets")

        tentativa = 0
        while True:
            try:
                async with websockets.connect(self.url, max_size=None) as ws:
                    self.ws = ws
                    tentativa = 0
                    print(f"[cliente] conectado ao bridge em {self.url}")
                    async for msg in ws:
                        self._tratar(json.loads(msg))
            except (OSError, ConnectionError) as e:
                tentativa += 1
                espera = min(30, 2 * tentativa)
                print(f"[cliente] sem conexão ({e}). Nova tentativa em {espera}s...")
                print("          o bridge está rodando? node minecraft_bridge.js")
                await asyncio.sleep(espera)
            finally:
                self.ws = None
                self.conectado = False

    def _tratar(self, msg: dict) -> None:
        tipo = msg.get("type")

        if tipo == "conectado":
            self.conectado = True
            print(f"[minecraft] ✅ conectado: {msg['usuario']}@{msg['servidor']}")
        elif tipo == "desconectado":
            self.conectado = False
            print(f"[minecraft] ❌ desconectado: {msg.get('motivo')}")
        elif tipo == "snapshot":
            self.ultimo_snapshot = msg.get("dados")
            if msg.get("origem") == "periodico" and not self.mostrar_snapshot_automatico:
                return
            self._mostrar_snapshot(msg.get("dados"))
        elif tipo == "chat":
            print(f"[chat] {msg['jogador']}: {msg['texto']}")
        elif tipo == "evento":
            print(f"[evento] {msg['evento']}: {msg.get('dados')}")
        elif tipo == "acao_resultado":
            fut = self._pendentes.pop(msg.get("id"), None)
            if fut and not fut.done():
                fut.set_result(msg)
            else:
                # resultado de ação que ninguém está esperando (ex: você
                # matou o processo que mandou e reiniciou) -- só mostra.
                marca = "✅" if msg.get("sucesso") else "⚠️"
                print(f"[ação] {marca} {msg.get('detalhe')}")
        else:
            print(f"[cliente] mensagem não reconhecida: {msg}")

    def _mostrar_snapshot(self, d: dict | None) -> None:
        if not d:
            print("[snapshot] vazio (bot ainda não entrou no mundo?)")
            return
        p = d["posicao"]
        print(
            f"[snapshot] pos=({p['x']}, {p['y']}, {p['z']}) "
            f"vida={d['vida']} fome={d['fome']} "
            f"{'dia' if d['e_dia'] else 'noite'}"
        )
        if d["inventario"]:
            itens = ", ".join(f"{i['nome']}x{i['quantidade']}" for i in d["inventario"])
            print(f"           inventário: {itens}")
        if d["blocos_proximos"]:
            blocos = ", ".join(f"{b['nome']}@({b['x']},{b['y']},{b['z']})" for b in d["blocos_proximos"])
            print(f"           blocos por perto: {blocos}")
        if d["entidades_proximas"]:
            entidades = ", ".join(f"{e['nome']}({e['distancia']}m)" for e in d["entidades_proximas"])
            print(f"           entidades por perto: {entidades}")

    async def enviar_acao(self, tipo: str, timeout: float = 30.0, **campos) -> dict:
        """Manda uma ação e espera o acao_resultado correspondente vir de
        volta -- correlacionado por id, não por ordem de chegada (várias
        ações podem estar em voo ao mesmo tempo)."""
        if not self.ws:
            return {"sucesso": False, "detalhe": "sem conexão com o bridge"}
        id_acao = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pendentes[id_acao] = fut
        await self.ws.send(json.dumps({"type": tipo, "id": id_acao, **campos}))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pendentes.pop(id_acao, None)
            return {"sucesso": False, "detalhe": f"sem resposta em {timeout}s"}


async def _loop_comandos(cliente: ClienteMinecraft) -> None:
    """Lê comando da entrada padrão numa thread (input() bloqueia) e manda
    a ação correspondente -- só pra testar cada tipo de ação manualmente
    antes de qualquer decisão automática existir.

    Desliga o snapshot automático assim que entra aqui: ele chegava a cada
    4s por conta própria e embaralhava com o que você tava digitando no
    mesmo terminal (comando cortado no meio por print concorrente). O
    comando "snapshot" continua funcionando igual, sob demanda.
    """
    cliente.mostrar_snapshot_automatico = False
    loop = asyncio.get_event_loop()
    print("Comandos: mover X Y Z | minerar BLOCO | craftar ITEM | "
          "equipar ITEM | atacar ALVO | seguir JOGADOR | parar | "
          "falar TEXTO | snapshot | sair")
    while True:
        linha = await loop.run_in_executor(None, input, "> ")
        partes = linha.strip().split(maxsplit=1)
        if not partes:
            continue
        cmd = partes[0].lower()
        resto = partes[1] if len(partes) > 1 else ""

        if cmd == "sair":
            return
        elif cmd == "mover":
            x, y, z = (float(v) for v in resto.split())
            r = await cliente.enviar_acao("mover_para", x=x, y=y, z=z)
        elif cmd == "minerar":
            r = await cliente.enviar_acao("minerar", bloco=resto, quantidade=1)
        elif cmd == "craftar":
            r = await cliente.enviar_acao("craftar", item=resto, quantidade=1)
        elif cmd == "equipar":
            r = await cliente.enviar_acao("equipar", item=resto)
        elif cmd == "atacar":
            r = await cliente.enviar_acao("atacar", alvo=resto)
        elif cmd == "seguir":
            r = await cliente.enviar_acao("seguir", jogador=resto)
        elif cmd == "parar":
            r = await cliente.enviar_acao("parar")
        elif cmd == "falar":
            r = await cliente.enviar_acao("falar_no_jogo", texto=resto)
        elif cmd == "snapshot":
            r = await cliente.enviar_acao("obter_snapshot")
        else:
            print(f"comando desconhecido: {cmd}")
            continue

        marca = "✅" if r.get("sucesso") else "⚠️"
        print(f"{marca} {r.get('detalhe')}")


async def main() -> None:
    cliente = ClienteMinecraft()
    tarefa_conexao = asyncio.create_task(cliente.rodar())
    try:
        await _loop_comandos(cliente)
    finally:
        tarefa_conexao.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nencerrando...")
        sys.exit(0)
