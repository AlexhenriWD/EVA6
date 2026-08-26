"""
minecraft_client.py -- cliente Python do minecraft_bridge.js.

Duas formas de uso, no mesmo arquivo:

1. INTEGRADO (minecraft_tools.py): a conexão vive numa thread dedicada e
   TUDO que chega do jogo -- chat, dano, morte, desconexão -- é entregue
   por callback pra quem integrou. Ver `ao_chat` / `ao_evento` /
   `ao_conexao` abaixo.

2. SOLTO (python minecraft_client.py): terminal interativo pra testar
   cada ação isolada contra o servidor de verdade, sem EVA nenhuma no
   meio. Sem callback registrado, tudo que chega é impresso -- é o modo
   que existia antes e continua igual.

MUDANÇA IMPORTANTE EM RELAÇÃO À VERSÃO ANTERIOR: antes, `_tratar` tratava
`chat` e `evento` imprimindo no console e mais nada. O bridge em Node tem
cuidado real com isso (filtra o eco da própria fala, só dispara evento em
QUEDA de vida, manda snapshot inteiro na morte) e o lado Python jogava
tudo fora. Resultado: a EVA falava no jogo mas não ouvia nada, e nunca
sabia que tinha tomado dano ou morrido. Os callbacks aqui são o caminho
que faltava -- quem consome é minecraft_tools, que transforma isso em
impulso de Consciência (mesmo padrão de robot_tools._emitir_evento_corpo).
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from typing import Callable

try:
    import websockets
except ImportError:
    websockets = None


class ClienteMinecraft:
    def __init__(self, url: str = "ws://localhost:8766", verboso: bool = True):
        self.url = url
        # Modo solto imprime; modo integrado (dentro da EVA) não tem
        # terminal pra imprimir e o console já é do processo principal.
        self.verboso = verboso
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

        # ------------------------------------------------ percepção
        # Callbacks opcionais. Nenhum é obrigatório: sem eles, o cliente
        # se comporta exatamente como a versão solta de teste. Todos são
        # chamados DE DENTRO do event loop dedicado, então precisam ser
        # baratos e nunca bloquear -- quem registra deve só enfileirar e
        # sair (ver minecraft_tools._emitir_evento_jogo).
        self.ao_chat: Callable[[str, str], None] | None = None
        self.ao_evento: Callable[[str, dict], None] | None = None
        self.ao_conexao: Callable[[bool, str], None] | None = None

    # ------------------------------------------------------------ conexão

    async def rodar(self) -> None:
        if websockets is None:
            raise SystemExit("websockets não instalado. Rode: pip install websockets")

        tentativa = 0
        while True:
            try:
                async with websockets.connect(self.url, max_size=None) as ws:
                    self.ws = ws
                    tentativa = 0
                    self._log(f"[cliente] conectado ao bridge em {self.url}")
                    async for msg in ws:
                        try:
                            self._tratar(json.loads(msg))
                        except Exception as e:
                            # Uma mensagem problemática não pode derrubar
                            # o laço inteiro -- mesma regra do bridge do
                            # Discord (bridge_client._tratar_mensagem).
                            self._log(f"[cliente] mensagem ignorada ({e})")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # ANTES: `except (OSError, ConnectionError)`. Isso cobria
                # o caso testado (bridge não rodando = connection refused,
                # que é OSError) e SÓ ele. websockets.exceptions.
                # ConnectionClosedError -- que é o que acontece quando o
                # processo do bridge morre no meio (fechamento 1006) --
                # não herda de nenhum dos dois. A exceção subia, matava
                # run_until_complete, matava a thread eva-minecraft, e
                # como `_thread` continuava não sendo None do lado de
                # minecraft_tools, nada nunca reiniciava: Minecraft morto
                # até reiniciar a EVA inteira, em silêncio. Captura ampla
                # aqui é o certo -- este laço é literalmente o dono da
                # política de reconexão.
                tentativa += 1
                espera = min(30, 2 * tentativa)
                self._log(f"[cliente] sem conexão ({type(e).__name__}: {e}). "
                          f"Nova tentativa em {espera}s...")
                self._log("          o bridge está rodando? node minecraft_bridge.js")
                await asyncio.sleep(espera)
            finally:
                self.ws = None
                if self.conectado:
                    self.conectado = False
                    self._notificar_conexao(False, "conexão com o bridge caiu")
                # Ação em voo que nunca vai receber resposta: resolve
                # agora como falha em vez de deixar o chamador esperando
                # o timeout inteiro à toa.
                self._cancelar_pendentes("conexão com o bridge caiu")

    def _cancelar_pendentes(self, motivo: str) -> None:
        for id_acao, fut in list(self._pendentes.items()):
            self._pendentes.pop(id_acao, None)
            if not fut.done():
                fut.set_result({"sucesso": False, "detalhe": motivo})

    # ---------------------------------------------------------- recepção

    def _tratar(self, msg: dict) -> None:
        tipo = msg.get("type")

        if tipo == "conectado":
            ja_estava = self.conectado
            self.conectado = True
            self._log(f"[minecraft] ✅ conectado: {msg.get('usuario')}@{msg.get('servidor')}")
            if not ja_estava:
                self._notificar_conexao(True, f"entrei no servidor {msg.get('servidor')}")

        elif tipo == "desconectado":
            self.conectado = False
            self._log(f"[minecraft] ❌ desconectado: {msg.get('motivo')}")
            self._notificar_conexao(False, str(msg.get("motivo") or "desconectada do servidor"))

        elif tipo == "snapshot":
            # Guardado como veio. Quem lê de outra thread deve COPIAR
            # antes de devolver adiante -- ver minecraft_estado: sem a
            # cópia, registry.executar carimbava "_ms" direto neste dict
            # compartilhado a cada chamada.
            self.ultimo_snapshot = msg.get("dados")
            if msg.get("origem") == "periodico" and not self.mostrar_snapshot_automatico:
                return
            self._mostrar_snapshot(msg.get("dados"))

        elif tipo == "chat":
            jogador = str(msg.get("jogador") or "alguém")
            texto = str(msg.get("texto") or "")
            self._log(f"[chat] {jogador}: {texto}")
            if self.ao_chat and texto.strip():
                self._seguro(lambda: self.ao_chat(jogador, texto))

        elif tipo == "evento":
            evento = str(msg.get("evento") or "")
            dados = msg.get("dados") or {}
            self._log(f"[evento] {evento}: {dados}")
            if self.ao_evento:
                self._seguro(lambda: self.ao_evento(evento, dados))

        elif tipo == "acao_resultado":
            fut = self._pendentes.pop(msg.get("id"), None)
            if fut and not fut.done():
                fut.set_result(msg)
            else:
                # resultado de ação que ninguém está esperando (ex: você
                # matou o processo que mandou e reiniciou) -- só mostra.
                marca = "✅" if msg.get("sucesso") else "⚠️"
                self._log(f"[ação] {marca} {msg.get('detalhe')}")

        else:
            self._log(f"[cliente] mensagem não reconhecida: {msg}")

    def _notificar_conexao(self, conectado: bool, motivo: str) -> None:
        if self.ao_conexao:
            self._seguro(lambda: self.ao_conexao(conectado, motivo))

    def _seguro(self, fn) -> None:
        """Callback de integração nunca pode derrubar o laço de recepção.
        Mesma regra de todo callback deste projeto: falha vira log, não
        exceção que sobe."""
        try:
            fn()
        except Exception as e:
            self._log(f"[cliente] callback falhou ({type(e).__name__}: {e})")

    def _log(self, texto: str) -> None:
        if self.verboso:
            print(texto)

    def _mostrar_snapshot(self, d: dict | None) -> None:
        if not self.verboso:
            return
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

    # ------------------------------------------------------------- envio

    async def enviar_acao(self, tipo: str, timeout: float = 30.0, **campos) -> dict:
        """Manda uma ação e espera o acao_resultado correspondente vir de
        volta -- correlacionado por id, não por ordem de chegada (várias
        ações podem estar em voo ao mesmo tempo)."""
        if not self.ws:
            return {"sucesso": False, "detalhe": "sem conexão com o bridge"}
        id_acao = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pendentes[id_acao] = fut
        try:
            await self.ws.send(json.dumps({"type": tipo, "id": id_acao, **campos}))
        except Exception as e:
            self._pendentes.pop(id_acao, None)
            return {"sucesso": False, "detalhe": f"falha ao enviar: {e}"}
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pendentes.pop(id_acao, None)
            return {"sucesso": False, "detalhe": f"sem resposta em {timeout}s"}


# ---------------------------------------------------------- modo solto


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
    loop = asyncio.get_running_loop()
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

        try:
            if cmd == "sair":
                return
            elif cmd == "mover":
                x, y, z = (float(v) for v in resto.split())
                r = await cliente.enviar_acao("mover_para", timeout=60, x=x, y=y, z=z)
            elif cmd == "minerar":
                r = await cliente.enviar_acao("minerar", timeout=60, bloco=resto, quantidade=1)
            elif cmd == "craftar":
                r = await cliente.enviar_acao("craftar", item=resto, quantidade=1)
            elif cmd == "equipar":
                r = await cliente.enviar_acao("equipar", item=resto)
            elif cmd == "atacar":
                r = await cliente.enviar_acao("atacar", timeout=60, alvo=resto)
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
        except ValueError:
            print("argumento inválido (mover espera três números: mover X Y Z)")
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