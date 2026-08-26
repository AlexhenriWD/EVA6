"""
Dashboard local -- painel de controle e debug da EVA.

Servidor HTTP da stdlib (http.server.ThreadingHTTPServer), sem
dependência nova. Roda numa thread separada do event loop principal.

POR QUE STDLIB EM VEZ DE aiohttp/FastAPI
------------------------------------------
Investiguei antes de escolher: quase toda peça que o dashboard precisa
tocar (decisão, busca de memória, montagem de contexto -- ver
EVA.pre_visualizar em orchestrator.py) já é síncrona por desenho. Os
únicos objetos que o dashboard mexe (EVAConfig e seus sub-configs,
BancoMemoria, SistemaVisual, Consciencia) já são acessados de threads
diferentes em outros lugares do sistema (BancoMemoria usa RLock
explicitamente por causa disso). Rodar o dashboard numa thread com
http.server evita a complexidade de rodar um segundo framework async
dentro do mesmo processo, ponte com asyncio.run_coroutine_threadsafe,
etc. -- sem precisar de nada disso, escrita simples de atributo (bool
gate) já é segura o bastante sob o GIL para um painel de controle/debug.

POR QUE OS CAMPOS DE CONFIG SÃO "AO VIVO" SEM MECANISMO ESPECIAL
--------------------------------------------------------------------
`EVAConfig.consciencia`, `.visao`, `.memoria` etc. são sub-objetos
guardados por REFERÊNCIA em toda parte do sistema que os usa (Consciencia
guarda `cfg.consciencia`, SistemaVisual guarda `cfg.visao`, ...). Mudar um
campo aqui, no mesmo objeto que o ClienteBridge já tem, propaga sozinho --
o próximo lugar que ler aquele campo (próximo tick, próximo turno) já vê
o valor novo. Não há "aplicar mudança"; a mudança já é o objeto vivo.

O QUE NÃO É AO VIVO -- E POR QUÊ
------------------------------------
`usar_embeddings` só é lido UMA VEZ, na construção do BancoMemoria (decide
se cria o ClienteEmbeddings). Depois disso, `self.embeddings is not None`
é o que importa, e nada re-checa o campo do config. Mesma coisa para
`visao.ativa` no momento em que ClienteBridge decide criar (ou não) o
SistemaVisual. Esses dois exigem reiniciar para mudar de OFF para ON (ou
vice-versa "de verdade", destruindo o objeto). O dashboard deixa isso
explícito na UI em vez de fingir que o toggle funcionaria.

SEÇÃO "ROBÔ FÍSICO" (NOVA)
------------------------------
Mesmo padrão da seção Minecraft (módulo próprio com conexão/thread
dedicada, lido por função pública -- eva/tools/robot_tools.py), com uma
diferença de propósito: robot_tools.status_dashboard() FAZ I/O de rede
de verdade (busca o estado atual do robô, com timeout curto de 2s) em
vez de só ler atributo -- porque o corpo aqui é físico, e saber "qual
câmera está ativa agora" (não qual estava ativa há 15s) importa mais do
que custa. Também expõe dois botões de ação (parar / EMERGENCY STOP) que
NENHUMA outra seção deste painel tem -- as outras seções controlam
software; esta controla hardware que pode bater em alguma coisa.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# Minecraft e Robô vivem em módulos próprios (eva/tools/minecraft_tools.py,
# eva/tools/robot_tools.py), cada um com conexão/thread dedicada -- não são
# atributo de ClienteBridge como visão/consciência são. Import de módulo
# inteiro, não símbolo solto, porque o dashboard só lê funções públicas
# (status_dashboard, definir_iniciativa, ...), nunca toca estado interno
# (_cliente, _tarefa_atual) direto.
from eva.tools import minecraft_tools
from eva.tools import robot_tools
from eva.voice.tts import BACKENDS as TTS_BACKENDS, ErroTTS

# Chaves aceitas pelo endpoint /api/toggle, mapeadas para uma função que
# aplica o valor no cfg. Whitelist explícita -- uma chave não listada aqui
# é rejeitada com 400, nunca ignorada em silêncio (silêncio seria pior:
# pareceria que o toggle funcionou quando não fez nada).
def _aplicar_toggles(cfg):
    def set_debug(v):
        # EVA_DEBUG existe em DOIS lugares (EVAConfig.debug e
        # VisaoConfig.debug, lidos independentemente na inicialização) --
        # tratados aqui como UM toggle lógico só, escrevendo nos dois.
        cfg.debug = v
        cfg.visao.debug = v

    return {
        "debug": set_debug,
        "consciencia_ativa": lambda v: setattr(cfg.consciencia, "ativa", v),
        "carisma": lambda v: setattr(cfg.llm, "carisma", v),
        "memoria_llm": lambda v: setattr(cfg.memoria, "extrair_com_llm", v),
        "consolidar": lambda v: setattr(cfg.memoria, "consolidar_com_llm", v),
        # Diferente de visão -- essa é de verdade ao vivo, sem precisar
        # reiniciar. Ver definir_iniciativa em minecraft_tools.py: o ciclo
        # já está sempre rodando, só checa essa flag a cada volta.
        "minecraft_iniciativa": lambda v: minecraft_tools.definir_iniciativa(v),
        # Mesmo padrão, ver definir_iniciativa em robot_tools.py.
        "robo_iniciativa": lambda v: robot_tools.definir_iniciativa(v),
    }


# Rótulo amigável + descrição, na ordem em que aparecem na UI.
_TOGGLES_INFO = [
    ("debug", "Logs de debug",
     "[memoria-llm], [consolidacao], [lacuna], [consciencia], [visao] no console."),
    ("consciencia_ativa", "Consciência (fala espontânea)",
     "Ela pode puxar assunto sozinha no silêncio. Desligar = só responde quando falada com."),
    ("carisma", "Humor / carisma no prompt",
     "Linha extra pedindo mais emoção e brincadeira. Aditivo, não mexe na âncora treinada."),
    ("memoria_llm", "Extração de memória por LLM",
     "Cobre fato dito de forma indireta, além das regras. Roda em segundo plano."),
    ("consolidar", "Consolidação periódica de memória",
     "Resume memórias antigas parecidas a cada N turnos. Requer embeddings ligado."),
    ("minecraft_iniciativa", "Minecraft: iniciativa própria",
     "Ela decide sozinha, periodicamente, se quer começar uma tarefa no jogo sem ninguém pedir."),
    ("robo_iniciativa", "Robô: iniciativa própria",
     "Ela decide sozinha, periodicamente, se quer se mexer (olhar em volta ou andar um pouco). "
     "DESLIGADO por padrão por bom motivo -- ligue só depois de confirmar que a direção das "
     "rodas e a identificação de câmera estão certas na seção Robô Físico abaixo."),
]


class _Handler(BaseHTTPRequestHandler):
    # atribuído pelo ServidorDashboard antes de servir
    cliente = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):
        pass  # silencioso -- o log da EVA já tem debug suficiente por conta própria

    # ------------------------------------------------------------ util

    def _json(self, status: int, dados: dict) -> None:
        corpo = json.dumps(dados, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def _html(self, corpo: str) -> None:
        dados = corpo.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def _corpo_json(self) -> dict:
        tam = int(self.headers.get("Content-Length", 0))
        if tam == 0:
            return {}
        return json.loads(self.rfile.read(tam))

    # ------------------------------------------------------------ GET

    def do_GET(self):
        caminho = urlparse(self.path).path
        if caminho == "/":
            self._html(PAGINA_HTML)
        elif caminho == "/api/estado":
            self._json(200, self.cliente.montar_estado())
        else:
            self._json(404, {"erro": "rota não encontrada"})

    # ------------------------------------------------------------ POST

    def do_POST(self):
        caminho = urlparse(self.path).path
        try:
            corpo = self._corpo_json()
        except (json.JSONDecodeError, ValueError):
            self._json(400, {"erro": "corpo não é JSON válido"})
            return

        if caminho == "/api/toggle":
            self._post_toggle(corpo)
        elif caminho == "/api/acao":
            self._post_acao(corpo)
        elif caminho == "/api/prompt-preview":
            self._post_prompt_preview(corpo)
        elif caminho == "/api/tts":
            self._post_tts(corpo)
        elif caminho == "/api/desligar":
            self._post_desligar()
        else:
            self._json(404, {"erro": "rota não encontrada"})

    def _post_toggle(self, corpo: dict) -> None:
        chave = corpo.get("chave")
        valor = corpo.get("valor")
        if not isinstance(chave, str) or not isinstance(valor, bool):
            self._json(400, {"erro": "corpo precisa de {chave: str, valor: bool}"})
            return
        aplicadores = _aplicar_toggles(self.cliente.cfg)
        if chave not in aplicadores:
            self._json(400, {"erro": f"chave desconhecida: {chave!r}. "
                                     f"Válidas: {list(aplicadores)}"})
            return
        aplicadores[chave](valor)
        self._json(200, {"ok": True, "chave": chave, "valor": valor})

    def _post_acao(self, corpo: dict) -> None:
        acao = corpo.get("acao")
        try:
            resultado = self.cliente.executar_acao(acao, corpo)
            self._json(200, {"ok": True, "resultado": resultado})
        except ValueError as e:
            self._json(400, {"erro": str(e)})
        except Exception as e:
            self._json(500, {"erro": f"falha ao executar ação: {e}"})

    def _post_tts(self, corpo: dict) -> None:
        backend = corpo.get("backend")
        if backend not in TTS_BACKENDS:
            self._json(400, {"erro": f"backend desconhecido: {backend!r}. "
                                     f"Válidos: {sorted(TTS_BACKENDS)}"})
            return
        try:
            resultado = self.cliente.cliente_bridge.trocar_tts(backend)
            self._json(200, {"ok": True, **resultado})
        except ErroTTS as e:
            # A mensagem de erro de criar_tts() já é específica (chave
            # faltando, pacote não instalado, voice_id vazio, etc.) --
            # devolve ela crua pro dashboard mostrar, em vez de só "falhou".
            # Isso é o que resolve "troquei e não vi nada acontecer": agora
            # o motivo aparece na hora, na tela, sem precisar de console.
            self._json(400, {"erro": str(e)})

    def _post_desligar(self) -> None:
        """Fecha tudo e mata o processo Python -- ver desligar_tudo()
        no ClienteBridge pra entender por que é assim, bruto."""
        self._json(200, {"ok": True, "mensagem": "encerrando..."})
        threading.Timer(0.3, self.cliente.cliente_bridge.desligar_tudo).start()

    def _post_prompt_preview(self, corpo: dict) -> None:
        mensagem = corpo.get("mensagem", "")
        if not mensagem.strip():
            self._json(400, {"erro": "mensagem vazia"})
            return
        try:
            r = self.cliente.eva.pre_visualizar(
                mensagem,
                usuario=corpo.get("usuario") or None,
                modo_voz=bool(corpo.get("modo_voz", False)),
                contexto_visual=corpo.get("contexto_visual") or None,
            )
            self._json(200, r)
        except Exception as e:
            self._json(500, {"erro": f"falha ao montar prompt: {e}"})


class ServidorDashboard:
    """Integra com ClienteBridge -- ver eva/integrations/bridge_client.py.

    `cliente` é o ClienteBridge inteiro (não só EVA), porque o dashboard
    precisa enxergar coisas que só existem nessa camada: consciências por
    guild, sistema visual, config compartilhada.
    """

    def __init__(self, cliente_bridge):
        self.cliente_bridge = cliente_bridge
        self.cfg = cliente_bridge.cfg
        self.eva = cliente_bridge.eva
        self._servidor: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._inicio = time.time()
        # Cache curto de "modelo disponível": montar_estado() é chamado a
        # cada poll do frontend, e checar isso faz um GET /v1/models de
        # verdade no LM Studio (ver ClienteLLM.disponivel em llm.py) --
        # sem cache, isso rodava a cada 2s (intervalo antigo do polling),
        # enchendo o log do LM Studio de ruído sem nenhum ganho: se o
        # modelo estava disponível 2s atrás, quase certamente ainda está.
        self._cache_modelo: tuple[float, bool] | None = None
        self._cache_modelo_ttl = 10.0
        # Mesmo padrão do cache de modelo acima, pro SearXNG: sem isso,
        # cada poll do dashboard faria uma requisição HTTP extra pro
        # container. Existe porque a instabilidade do SearXNG (ver log de
        # docker) até agora só aparecia quando uma busca falhava NO MEIO
        # de uma conversa -- não tinha jeito de checar "tá no ar?" sem
        # esperar isso acontecer de novo.
        self._cache_searxng: tuple[float, tuple[bool, str]] | None = None
        self._cache_searxng_ttl = 15.0

    def _modelo_disponivel_cacheado(self) -> bool:
        agora = time.time()
        if self._cache_modelo is not None:
            quando, valor = self._cache_modelo
            if agora - quando < self._cache_modelo_ttl:
                return valor
        valor = self.eva.llm.disponivel()
        self._cache_modelo = (agora, valor)
        return valor

    def _searxng_disponivel_cacheado(self) -> tuple[bool, str]:
        """(disponivel, detalhe_do_erro). Timeout curto (2s) de propósito:
        isto roda a cada poll do dashboard, não pode travar a UI esperando
        um container que pode estar fora do ar.
        """
        agora = time.time()
        if self._cache_searxng is not None:
            quando, valor = self._cache_searxng
            if agora - quando < self._cache_searxng_ttl:
                return valor
        import os
        import urllib.error
        import urllib.request
        url = os.environ.get("EVA_SEARXNG_URL", "http://127.0.0.1:8080").rstrip("/") + "/healthz"
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                valor = (r.status == 200, "")
        except urllib.error.URLError as e:
            valor = (False, str(e.reason)[:100])
        except Exception as e:
            valor = (False, str(e)[:100])
        self._cache_searxng = (agora, valor)
        return valor

    def montar_estado(self) -> dict:
        cfg = self.cfg
        cb = self.cliente_bridge

        toggles = []
        for chave, rotulo, descricao in _TOGGLES_INFO:
            valor = {
                "debug": cfg.debug,
                "consciencia_ativa": cfg.consciencia.ativa,
                "carisma": cfg.llm.carisma,
                "memoria_llm": cfg.memoria.extrair_com_llm,
                "consolidar": cfg.memoria.consolidar_com_llm,
                "minecraft_iniciativa": minecraft_tools.status_dashboard()["iniciativa_ativa"],
                "robo_iniciativa": robot_tools.status_dashboard()["iniciativa_ativa"],
            }[chave]
            toggles.append({"chave": chave, "rotulo": rotulo,
                            "descricao": descricao, "valor": valor})

        visao = cb.visao
        if visao is None:
            info_visao = {
                "instanciada": False,
                "motivo": cb.erro_visao or (
                    "EVA_VISAO=0 -- para ligar, mude no .env e reinicie "
                    "(não dá pra ligar em tempo real: a captura de tela só "
                    "é preparada na inicialização)."
                ),
            }
        else:
            cena = visao.cena
            info_visao = {
                "instanciada": True,
                "ativo": visao.ativo,
                "cena_atual": cena.descricao if cena else None,
                "cena_idade_s": round(cena.idade_segundos(), 1) if cena else None,
            }

        consciencias = {
            gid: c.situacao() for gid, c in cb.consciencias.items()
        }

        try:
            memoria_total = self.eva.memoria.contar()
            usuarios = self.eva.memoria.usuarios()
        except Exception:
            memoria_total, usuarios = {}, []

        try:
            modelo_disponivel = self._modelo_disponivel_cacheado()
        except Exception:
            modelo_disponivel = False

        try:
            searxng_ok, searxng_erro = self._searxng_disponivel_cacheado()
        except Exception as e:
            searxng_ok, searxng_erro = False, str(e)[:100]

        return {
            "uptime_s": round(time.time() - self._inicio, 1),
            "toggles": toggles,
            "somente_reinicio": {
                "embeddings_ativo": cfg.memoria.usar_embeddings,
                "visao_configurada": cfg.visao.ativa,
            },
            "tts": {
                "backend_pedido": cfg.voz.tts_backend,
                "backend_ativo": cb.tts.nome if cb.tts else None,
                "disponiveis": sorted(TTS_BACKENDS),
                "erro": cb.erro_tts,
            },
            "searxng": {
                "disponivel": searxng_ok,
                "erro": searxng_erro or None,
            },
            "modelo": {
                "nome": cfg.llm.modelo,
                "url": cfg.llm.base_url,
                "disponivel": modelo_disponivel,
            },
            "memoria": {"por_tipo": memoria_total, "usuarios": usuarios},
            "visao": info_visao,
            "consciencia_por_guild": consciencias,
            "minecraft": minecraft_tools.status_dashboard(),
            "robo": robot_tools.status_dashboard(),
        }

    def executar_acao(self, acao: str, corpo: dict) -> dict:
        """Ações manuais úteis para debug. Só toca método que já é
        sincrono e seguro de chamar fora do event loop (ver docstring do
        módulo) -- nada aqui usa await."""
        cb = self.cliente_bridge

        if acao == "visao_ligar":
            if cb.visao is None:
                raise ValueError("visão não instanciada (EVA_VISAO=0 na inicialização)")
            cb.visao.ligar()
            return {"visao_ativo": cb.visao.ativo}

        if acao == "visao_desligar":
            if cb.visao is None:
                raise ValueError("visão não instanciada")
            cb.visao.desligar()
            return {"visao_ativo": cb.visao.ativo}

        if acao == "visao_tick_agora":
            if cb.visao is None:
                raise ValueError("visão não instanciada")
            if not cb.visao.ativo:
                raise ValueError("visão instanciada mas desligada -- ligue primeiro")
            evento = cb.visao.tick()
            return {"evento_gerado": evento}

        if acao == "esquecer_memoria":
            usuario = corpo.get("usuario")
            termo = corpo.get("termo")
            if not usuario or not termo:
                raise ValueError("ação 'esquecer_memoria' precisa de usuario e termo")
            n = self.eva.esquecer(termo, usuario=usuario)
            return {"removidas": n}

        if acao == "minecraft_cancelar_tarefa":
            return minecraft_tools.minecraft_cancelar_tarefa()

        if acao == "robo_conectar":
            return robot_tools.conectar_dashboard()

        if acao == "robo_parar":
            return robot_tools.parar_dashboard()

        if acao == "robo_estop":
            motivo = corpo.get("motivo") or "acionado pelo painel"
            return robot_tools.estop_dashboard(motivo)

        if acao == "robo_reset_estop":
            return robot_tools.reset_estop_dashboard()

        raise ValueError(
            f"ação desconhecida: {acao!r}. Válidas: visao_ligar, "
            f"visao_desligar, visao_tick_agora, esquecer_memoria, "
            f"minecraft_cancelar_tarefa, robo_conectar, robo_parar, "
            f"robo_estop, robo_reset_estop"
        )

    # ------------------------------------------------------------ ciclo

    def iniciar(self) -> None:
        _Handler.cliente = self
        self._servidor = ThreadingHTTPServer(
            (self.cfg.dashboard.host, self.cfg.dashboard.porta), _Handler)
        self._thread = threading.Thread(
            target=self._servidor.serve_forever, daemon=True,
            name="eva-dashboard")
        self._thread.start()
        print(f"[dashboard] http://{self.cfg.dashboard.host}:{self.cfg.dashboard.porta}")

    def parar(self) -> None:
        if self._servidor is not None:
            self._servidor.shutdown()
            self._servidor.server_close()
            self._servidor = None


PAGINA_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>EVA -- painel</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", sans-serif;
    background: #14161b; color: #e4e6eb; margin: 0; padding: 24px;
    max-width: 920px; margin-inline: auto;
  }
  h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
  .sub { color: #8a8f98; font-size: 13px; margin-bottom: 24px; }
  section {
    background: #1c1f26; border: 1px solid #2a2e37; border-radius: 10px;
    padding: 16px 18px; margin-bottom: 16px;
  }
  section h2 { font-size: 14px; text-transform: uppercase; letter-spacing: .04em;
    color: #8a8f98; margin: 0 0 12px; }
  .linha { display: flex; align-items: center; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid #262a33; gap: 16px; }
  .linha:last-child { border-bottom: none; }
  .rotulo { font-size: 14px; font-weight: 500; }
  .desc { font-size: 12px; color: #8a8f98; margin-top: 2px; }
  .switch { position: relative; width: 40px; height: 22px; flex-shrink: 0; }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider { position: absolute; inset: 0; background: #3a3f4b; border-radius: 22px;
    cursor: pointer; transition: .15s; }
  .slider::before { content: ""; position: absolute; width: 16px; height: 16px;
    left: 3px; top: 3px; background: #e4e6eb; border-radius: 50%; transition: .15s; }
  input:checked + .slider { background: #4c7cf0; }
  input:checked + .slider::before { transform: translateX(18px); }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 20px;
    font-size: 11px; font-weight: 600; }
  .pill.on { background: #1e3a2a; color: #5fd88a; }
  .pill.off { background: #3a2020; color: #e88; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .kv { font-size: 13px; color: #c4c8d0; }
  .kv b { color: #e4e6eb; }
  textarea, input[type=text] {
    width: 100%; background: #14161b; border: 1px solid #2a2e37; color: #e4e6eb;
    border-radius: 6px; padding: 8px 10px; font-family: inherit; font-size: 13px;
  }
  textarea { min-height: 60px; resize: vertical; }
  button {
    background: #2a2e37; color: #e4e6eb; border: 1px solid #3a3f4b; border-radius: 6px;
    padding: 7px 14px; font-size: 13px; cursor: pointer; margin-top: 8px;
  }
  button:hover { background: #343947; }
  button.perigo { background: #3a2020; border-color: #5a2a2a; color: #ff8a8a; }
  button.perigo:hover { background: #4a2626; }
  pre { background: #0f1115; border: 1px solid #2a2e37; border-radius: 6px;
    padding: 12px; font-size: 12px; overflow-x: auto; white-space: pre-wrap;
    word-break: break-word; max-height: 400px; overflow-y: auto; }
  .impulso { font-size: 12px; color: #c4c8d0; padding: 2px 0; }
  .vazio { color: #565b66; font-style: italic; font-size: 13px; }
</style>
</head>
<body>
  <h1>EVA -- painel de controle</h1>
  <div class="sub" id="uptime">carregando...</div>

  <section>
    <h2>Controle</h2>
    <button style="background:#3a2020;border-color:#5a2a2a;color:#ff8a8a"
            onclick="fecharTudo()">Fechar tudo</button>
    <div class="desc" style="margin-top:6px">
      Encerra o processo Python (bridge, visão, dashboard, banco) na hora.
      Não fecha o bridge.js (Node) -- isso é outro terminal, feche à parte.
    </div>
  </section>

  <section>
    <h2>Interruptores</h2>
    <div id="toggles"></div>
  </section>

  <section>
    <h2>Só com reinício</h2>
    <div id="somente_reinicio"></div>
  </section>

  <section>
    <h2>Modelo</h2>
    <div id="modelo" class="grid2"></div>
  </section>

  <section>
    <h2>Voz (TTS)</h2>
    <div id="tts_status"></div>
    <div class="grid2" style="margin-top:8px">
      <select id="tts_select"></select>
      <button onclick="trocarTts()" style="margin-top:0">Trocar agora</button>
    </div>
    <div class="desc" style="margin-top:6px">
      Troca em tempo real -- não precisa editar .env nem reiniciar nada.
      Se der erro (chave faltando, backend não instalado), aparece aqui na hora.
    </div>
  </section>

  <section>
    <h2>Busca (SearXNG)</h2>
    <div id="searxng_status"></div>
    <div class="desc" style="margin-top:6px">
      Checa http://.../healthz a cada poll (cache de 15s) -- é a ferramenta
      "buscar" que a EVA usa. Se aparecer indisponível aqui, ela vai
      alucinar em vez de pesquisar quando pedirem uma busca.
    </div>
  </section>

  <section>
    <h2>Memória</h2>
    <div id="memoria"></div>
  </section>

  <section>
    <h2>Visão</h2>
    <div id="visao"></div>
    <div id="visao_acoes"></div>
  </section>

  <section>
    <h2>Consciência (por canal de voz)</h2>
    <div id="consciencia"></div>
  </section>

  <section>
    <h2>Minecraft</h2>
    <div id="minecraft"></div>
  </section>

  <section>
    <h2>Robô Físico</h2>
    <div id="robo"></div>
    <div class="desc" style="margin-top:10px; padding-top:10px; border-top:1px solid #262a33">
      <b>USB</b> = webcam de navegação (o "olho" que vê pra onde o carro anda).
      <b>PICAM</b> = câmera na garra/cabeça (o "rosto" que se move com o braço).
      São hardwares completamente diferentes (USB via OpenCV, PICAM via
      Picamera2) -- não dá pra uma virar a outra por engano no código, mas o
      <i>índice</i> da USB (qual /dev/videoN) pode mudar se o cabo for
      replugado. Se "câmera ativa" abaixo não bater com o que você espera
      ver, é o primeiro lugar a olhar.
    </div>
  </section>

  <section>
    <h2>Pré-visualizar prompt (não chama o modelo)</h2>
    <textarea id="pv_mensagem" placeholder="Digite uma mensagem de teste..."></textarea>
    <div class="grid2" style="margin-top:8px">
      <input type="text" id="pv_usuario" placeholder="usuario (opcional)">
      <label style="font-size:13px;display:flex;align-items:center;gap:6px">
        <input type="checkbox" id="pv_modo_voz"> modo voz
      </label>
    </div>
    <button onclick="preverPrompt()">Montar prompt</button>
    <pre id="pv_resultado" style="display:none"></pre>
  </section>

<script>
async function api(caminho, opcoes) {
  const r = await fetch(caminho, opcoes);
  return r.json();
}

function pill(v) {
  return `<span class="pill ${v ? 'on' : 'off'}">${v ? 'ligado' : 'desligado'}</span>`;
}

async function fecharTudo() {
  if (!confirm('Fechar tudo? Isso encerra o processo da EVA agora.')) return;
  await api('/api/desligar', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: '{}',
  });
  document.body.innerHTML = '<h1>EVA encerrada.</h1>';
}

async function alternar(chave, valor) {
  await api('/api/toggle', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({chave, valor}),
  });
  carregar();
}

async function acao(nome, extra) {
  const r = await api('/api/acao', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({acao: nome, ...extra}),
  });
  carregar();
  return r;
}

async function roboConectar() {
  const r = await acao('robo_conectar');
  if (r.resultado && r.resultado.erro) {
    alert('Não consegui conectar: ' + JSON.stringify(r.resultado));
  }
}

async function roboEstop() {
  if (!confirm('EMERGENCY STOP no robô agora?')) return;
  await acao('robo_estop');
}

async function roboResetEstop() {
  const r = await acao('robo_reset_estop');
  if (r.resultado && r.resultado.erro) {
    alert('Não consegui liberar: ' + JSON.stringify(r.resultado));
  }
}

async function trocarTts() {
  const backend = document.getElementById('tts_select').value;
  const r = await api('/api/tts', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({backend}),
  });
  const status = document.getElementById('tts_status');
  if (r.erro) {
    status.innerHTML = `<div class="kv" style="color:#e88">Falhou: ${r.erro}</div>`;
  } else {
    carregar();
  }
}

async function preverPrompt() {
  const mensagem = document.getElementById('pv_mensagem').value;
  const usuario = document.getElementById('pv_usuario').value;
  const modo_voz = document.getElementById('pv_modo_voz').checked;
  const r = await api('/api/prompt-preview', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mensagem, usuario, modo_voz}),
  });
  const out = document.getElementById('pv_resultado');
  out.style.display = 'block';
  out.textContent = JSON.stringify(r, null, 2);
}

async function carregar() {
  const e = await api('/api/estado');

  document.getElementById('uptime').textContent =
    `rodando há ${Math.round(e.uptime_s)}s`;

  document.getElementById('toggles').innerHTML = e.toggles.map(t => `
    <div class="linha">
      <div>
        <div class="rotulo">${t.rotulo}</div>
        <div class="desc">${t.descricao}</div>
      </div>
      <label class="switch">
        <input type="checkbox" ${t.valor ? 'checked' : ''}
               onchange="alternar('${t.chave}', this.checked)">
        <span class="slider"></span>
      </label>
    </div>`).join('');

  const sr = e.somente_reinicio;
  document.getElementById('somente_reinicio').innerHTML = `
    <div class="linha"><div class="rotulo">Embeddings</div>${pill(sr.embeddings_ativo)}</div>
    <div class="linha"><div class="rotulo">Visão configurada (EVA_VISAO)</div>${pill(sr.visao_configurada)}</div>`;

  document.getElementById('modelo').innerHTML = `
    <div class="kv"><b>${e.modelo.nome}</b></div>
    <div class="kv">${pill(e.modelo.disponivel)}</div>
    <div class="kv" style="grid-column:1/-1">${e.modelo.url}</div>`;

  const tts = e.tts;
  document.getElementById('tts_status').innerHTML = tts.erro
    ? `<div class="kv" style="color:#e88"><b>sem voz -- ${tts.backend_pedido} falhou:</b> ${tts.erro}</div>`
    : `<div class="kv">ativo agora: <b>${tts.backend_ativo || '(nenhum)'}</b></div>`;
  const selectTts = document.getElementById('tts_select');
  if (selectTts.dataset.preenchido !== '1') {
    selectTts.innerHTML = tts.disponiveis.map(b => `<option value="${b}">${b}</option>`).join('');
    selectTts.dataset.preenchido = '1';
  }
  if (tts.backend_ativo) selectTts.value = tts.backend_ativo;

  const sx = e.searxng;
  document.getElementById('searxng_status').innerHTML = sx.disponivel
    ? `<div class="kv">${pill(true)} respondendo</div>`
    : `<div class="kv">${pill(false)} ${sx.erro || 'sem resposta'}</div>`;

  const porTipo = Object.entries(e.memoria.por_tipo)
    .map(([k, v]) => `${k}: <b>${v}</b>`).join(' &nbsp;·&nbsp; ') || 'vazio';
  document.getElementById('memoria').innerHTML = `
    <div class="kv">${porTipo}</div>
    <div class="kv" style="margin-top:6px">usuários: ${e.memoria.usuarios.join(', ') || '-'}</div>`;

  const v = e.visao;
  if (!v.instanciada) {
    document.getElementById('visao').innerHTML = `<div class="vazio">${v.motivo}</div>`;
    document.getElementById('visao_acoes').innerHTML = '';
  } else {
    document.getElementById('visao').innerHTML = `
      <div class="linha"><div class="rotulo">Ativa agora</div>${pill(v.ativo)}</div>
      <div class="kv" style="margin-top:8px"><b>Cena atual:</b> ${v.cena_atual || '(nenhuma ainda)'}</div>
      ${v.cena_idade_s !== null ? `<div class="kv">há ${v.cena_idade_s}s</div>` : ''}`;
    document.getElementById('visao_acoes').innerHTML = `
      <button onclick="acao('visao_ligar')">Ligar agora</button>
      <button onclick="acao('visao_desligar')">Desligar agora</button>
      <button onclick="acao('visao_tick_agora')">Forçar captura agora</button>`;
  }

  const guilds = Object.entries(e.consciencia_por_guild);
  document.getElementById('consciencia').innerHTML = guilds.length ? guilds.map(([gid, s]) => `
    <div style="margin-bottom:10px">
      <div class="kv"><b>canal ${gid}</b> -- silêncio ${s.silencio}s, ${s.ocupada ? 'ocupada' : 'livre'}</div>
      ${s.ultimo_veredito ? `<div class="kv" style="margin-top:2px">último veredito: ${s.ultimo_veredito}</div>` : ''}
      ${s.impulsos.length ? s.impulsos.map(i => `<div class="impulso">${i}</div>`).join('')
                          : '<div class="vazio">sem impulsos na fila</div>'}
      ${s.fios.length ? `<div class="kv" style="margin-top:4px">fios: ${s.fios.join('; ')}</div>` : ''}
    </div>`).join('') : '<div class="vazio">nenhum canal de voz ativo agora</div>';

  const mc = e.minecraft;
  if (!mc.conectado) {
    document.getElementById('minecraft').innerHTML = `
      <div class="linha"><div class="rotulo">Conectado</div>${pill(false)}</div>
      <div class="vazio" style="margin-top:6px">Sem conexão ainda -- conecta sozinho na primeira vez que
        alguma ferramenta de Minecraft for usada (node minecraft_bridge.js precisa estar rodando).</div>`;
  } else {
    const t = mc.tarefa;
    document.getElementById('minecraft').innerHTML = `
      <div class="linha"><div class="rotulo">Conectado</div>${pill(true)}</div>
      <div class="kv" style="margin-top:6px">vida ${mc.vida ?? '-'} &nbsp;·&nbsp; fome ${mc.fome ?? '-'}</div>
      ${mc.posicao ? `<div class="kv">posição (${mc.posicao.x}, ${mc.posicao.y}, ${mc.posicao.z})</div>` : ''}
      <div class="kv" style="margin-top:8px"><b>Tarefa:</b> ${t ? `${t.objetivo} -- ${t.status} (${t.passos} passo(s))` : '(nenhuma ainda)'}</div>
      <div class="kv">eventos na fila: ${mc.eventos_na_fila ?? 0}</div>
      ${t && t.status === 'ativa' ? `<button onclick="acao('minecraft_cancelar_tarefa')">Cancelar tarefa</button>` : ''}`;
  }

  const ro = e.robo;
  const roboDiv = document.getElementById('robo');
  if (!ro.conectado) {
    roboDiv.innerHTML = `
      <div class="linha"><div class="rotulo">Conectado</div>${pill(false)}</div>
      <div class="vazio" style="margin-top:6px">Sem conexão ainda -- conecta sozinho na primeira vez que
        alguma ferramenta de robô for usada, ou ao entrar numa call (eva_command_server.py precisa
        estar rodando no robô).</div>
      <button style="margin-top:8px" onclick="roboConectar()">Conectar agora</button>`;
  } else if (ro.erro_estado) {
    roboDiv.innerHTML = `
      <div class="linha"><div class="rotulo">Conectado</div>${pill(true)}</div>
      <div class="kv" style="color:#e88; margin-top:6px">Não consegui buscar o estado agora: ${ro.erro_estado}</div>
      <button onclick="acao('robo_parar')">Parar mesmo assim</button>
      <button class="perigo" onclick="roboEstop()">EMERGENCY STOP</button>`;
  } else {
    const est = ro.estado || {};
    const cam = est.camera || {};
    const seg = est.safety || {};
    const sensores = seg.last_sensor_data || {};
    const camAtiva = (cam.active_camera || '?').toUpperCase();
    const camDetalhe = camAtiva === 'USB'
      ? `USB (navegação) -- device index ${cam.usb_id ?? '?'}`
      : camAtiva === 'PICAM'
      ? `PICAM (garra/cabeça, "rosto") -- picam_id ${cam.picam_id ?? '?'}`
      : `desconhecida (${camAtiva})`;
    roboDiv.innerHTML = `
      <div class="linha"><div class="rotulo">Conectado</div>${pill(true)}</div>
      <div class="linha"><div class="rotulo">Emergency stop</div>${pill(!seg.emergency_stop)}</div>
      <div class="linha"><div class="rotulo">Watchdog OK</div>${pill(seg.watchdog_ok !== false)}</div>
      <div class="kv" style="margin-top:8px"><b>Câmera ativa:</b> ${camDetalhe}</div>
      <div class="kv">modo: ${est.mode ?? '-'} &nbsp;·&nbsp; trocando câmera: ${cam.switching ? 'sim' : 'não'}</div>
      <div class="kv" style="margin-top:6px">bateria: ${sensores.battery_v != null ? sensores.battery_v + 'V' : 'sem leitura'}
        &nbsp;·&nbsp; obstáculo: ${sensores.ultrasonic_cm != null ? sensores.ultrasonic_cm + 'cm' : 'sem leitura'}</div>
      <div style="margin-top:10px">
        <button onclick="acao('robo_parar')">Parar</button>
        <button class="perigo" onclick="roboEstop()">EMERGENCY STOP</button>
        ${seg.emergency_stop ? `<button onclick="roboResetEstop()">Liberar emergency stop</button>` : ''}
      </div>`;
  }
}

carregar();
setInterval(carregar, 15000);
</script>
</body>
</html>
"""