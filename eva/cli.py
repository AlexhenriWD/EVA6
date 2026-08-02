"""
Interface de conversa com a EVA.

Uso:
    python -m eva                       conversa
    python -m eva --debug               mostra o que o sistema fez em cada turno
    python -m eva --diagnostico         checa conexão, memórias, estado
    python -m eva --lembrar "fato"      adiciona memória manualmente
    python -m eva --memorias            lista o que a EVA sabe

Comandos durante a conversa:
    /debug      liga/desliga a visão interna
    /memoria    o que a EVA sabe sobre você
    /estado     mundo interno
    /lembrar X  guarda um fato
    /esquecer X remove memórias que casem com X
    /limpar     apaga o histórico da conversa (memórias ficam)
    /sair
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import carregar_config
from .orchestrator import EVA

VERDE = "\033[32m"
CINZA = "\033[90m"
AMARELO = "\033[33m"
NEGRITO = "\033[1m"
FIM = "\033[0m"


def _cor(texto: str, cor: str) -> str:
    return f"{cor}{texto}{FIM}" if sys.stdout.isatty() else texto


def mostrar_debug(r) -> None:
    p = r.plano
    print(_cor(f"  ├ intenção: {p.intencao} ({p.motivo})", CINZA))
    print(_cor(f"  ├ sinais: emoção={p.carga_emocional:.1f} "
               f"novidade={p.novidade:.1f} complexidade={p.complexidade:.1f}", CINZA))
    if r.memorias_usadas:
        for tipo, itens in r.memorias_usadas.items():
            print(_cor(f"  ├ memória[{tipo}]: {'; '.join(itens)}", CINZA))
    if r.ferramentas:
        for nome, res in r.ferramentas.items():
            limpo = {k: v for k, v in res.items() if not k.startswith("_")}
            print(_cor(f"  ├ ferramenta[{nome}]: {json.dumps(limpo, ensure_ascii=False)}", CINZA))
    if r.memorias_novas:
        for m in r.memorias_novas:
            print(_cor(f"  ├ aprendeu[{m['tipo']}]: {m['conteudo']}", AMARELO))
    print(_cor(f"  └ {r.ms}ms", CINZA))


def mostrar_diagnostico(eva: EVA) -> None:
    d = eva.diagnostico()
    print(f"\n{_cor('EVA — diagnóstico', NEGRITO)}\n")
    ok = d["llm_disponivel"]
    print(f"  modelo:     {'conectado' if ok else 'NÃO CONECTADO'} em {d['url']}")
    if ok:
        print(f"  disponíveis: {', '.join(d['modelos']) or '(nenhum)'}")
        if d["modelo_configurado"] not in d["modelos"] and d["modelos"]:
            print(_cor(f"  AVISO: '{d['modelo_configurado']}' não está na lista.", AMARELO))
            print(_cor(f"         Defina EVA_LLM_MODEL com um dos nomes acima.", AMARELO))
    else:
        print(_cor("\n  Abra o LM Studio, carregue o modelo da EVA e inicie", AMARELO))
        print(_cor("  o Local Server (aba Developer). Ou defina EVA_LLM_URL.", AMARELO))

    print(f"\n  decisor:    {d['decisor']}")
    print(f"  ferramentas: {', '.join(d['ferramentas'])}")
    print(f"  banco:      {d['banco']}")
    mem = d["memorias"]
    print(f"  memórias:   {sum(mem.values())} " +
          (f"({', '.join(f'{k}={v}' for k, v in mem.items())})" if mem else "(vazio)"))
    print(f"  interações: {d['interacoes']}")
    e = d["estado"]
    print(f"  estado:     energia={e['energia']} curiosidade={e['curiosidade']} "
          f"confiança={e['confianca']} estresse={e['estresse']}")

    # --- voz ---
    import os as _os
    print(f"\n  {_cor('voz', NEGRITO)}")
    tem_groq = bool(_os.environ.get("GROQ_API_KEY"))
    print(f"  STT (Groq): {'chave presente' if tem_groq else 'SEM GROQ_API_KEY'}")
    if not tem_groq:
        print(_cor("              pegue em console.groq.com e coloque no .env", CINZA))

    try:
        from .voice.tts import diagnostico as diag_tts
        idioma = _os.environ.get("EVA_TTS_IDIOMA", "pt")
        algum = False
        for nome, info in diag_tts().items():
            if info.get("instalado"):
                algum = True
                ok_idioma = "" if info["suporta_pt"] or idioma != "pt" else \
                    _cor(f"  (NÃO suporta {idioma})", AMARELO)
                print(f"  TTS {nome:<8}: instalado{ok_idioma}")
        if not algum:
            print(f"  TTS:        nenhum backend instalado")
            print(_cor("              pip install pocket-tts", CINZA))
    except ImportError:
        pass

    tem_discord = bool(_os.environ.get("DISCORD_TOKEN"))
    print(f"  Discord:    {'token presente' if tem_discord else 'sem DISCORD_TOKEN'}")
    # A voz não passa mais por discord.py: quem ouve a call é o bridge.js
    # (Node + @discordjs/voice). O que importa checar aqui é o Node, não
    # uma biblioteca Python que o sistema não usa mais.
    import shutil as _sh
    if _sh.which("node"):
        print("  ouvir call: sim  " + _cor("(node bridge.js)", CINZA))
    else:
        print("  ouvir call: não" + _cor("  (instale o Node.js)", CINZA))
    print()


def loop_conversa(eva: EVA, debug: bool) -> None:
    print(f"\n{_cor('EVA', NEGRITO)} — /ajuda para comandos, /sair para encerrar\n")

    if not eva.llm.disponivel():
        print(_cor("Modelo não está respondendo. Rode --diagnostico para detalhes.\n", AMARELO))

    while True:
        try:
            msg = input(_cor("você> ", NEGRITO)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté.")
            break

        if not msg:
            continue

        if msg.startswith("/"):
            if _comando(eva, msg):
                break
            debug = eva._debug_cli if hasattr(eva, "_debug_cli") else debug
            continue

        r = eva.responder(msg)
        if r.erro:
            print(_cor(f"\n[erro] {r.erro}\n", AMARELO))
            continue

        print(f"\n{_cor('EVA>', VERDE)} {r.resposta}\n")
        if debug or getattr(eva, "_debug_cli", False):
            mostrar_debug(r)
            print()


def _comando(eva: EVA, linha: str) -> bool:
    """Processa um comando. Retorna True se for para sair."""
    partes = linha.split(maxsplit=1)
    cmd = partes[0].lower()
    arg = partes[1].strip() if len(partes) > 1 else ""

    if cmd in ("/sair", "/quit", "/exit"):
        print("Até.")
        return True

    if cmd == "/ajuda":
        print("""
  /debug        liga/desliga a visão do que o sistema fez
  /memoria      o que a EVA sabe sobre você
  /estado       mundo interno (energia, curiosidade...)
  /lembrar X    guarda um fato
  /esquecer X   remove memórias que casem com X
  /limpar       apaga o histórico da conversa (memórias ficam)
  /sair
""")
    elif cmd == "/debug":
        eva._debug_cli = not getattr(eva, "_debug_cli", False)
        print(f"debug {'ligado' if eva._debug_cli else 'desligado'}")
    elif cmd == "/memoria":
        contagem = eva.memoria.contar()
        if not contagem:
            print("  (nada guardado ainda)")
        for tipo in ("semantica", "episodica", "procedural", "personalidade"):
            itens = eva.memoria.listar(usuario=eva.cfg.usuario, tipo=tipo, limite=20)
            if itens:
                print(f"\n  {tipo}:")
                for m in itens:
                    print(f"    - {m.conteudo}  {_cor(f'(conf {m.confianca:.2f})', CINZA)}")
        print()
    elif cmd == "/estado":
        e = eva.estado.estado
        print(f"\n  energia:     {e.energia:.2f}")
        print(f"  curiosidade: {e.curiosidade:.2f}")
        print(f"  confiança:   {e.confianca:.2f}")
        print(f"  estresse:    {e.estresse:.2f}")
        if e.foco:
            print(f"  foco:        {e.foco}")
        print(f"  interações:  {e.total_interacoes}\n")
    elif cmd == "/lembrar":
        if not arg:
            print("uso: /lembrar <fato>")
        else:
            eva.lembrar(arg)
            print(f"guardado: {arg}")
    elif cmd == "/esquecer":
        if not arg:
            print("uso: /esquecer <termo>")
        else:
            n = eva.esquecer(arg)
            print(f"{n} memória(s) removida(s)")
    elif cmd == "/limpar":
        eva.memoria.con.execute("DELETE FROM conversas WHERE usuario=?",
                                (eva.cfg.usuario,))
        eva.memoria.con.commit()
        print("histórico apagado (memórias mantidas)")
    else:
        print(f"comando desconhecido: {cmd} — /ajuda para a lista")
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="eva", description="EVA — IA conversacional")
    p.add_argument("--debug", action="store_true", help="mostra o processo interno a cada turno")
    p.add_argument("--diagnostico", action="store_true", help="checa a instalação e sai")
    p.add_argument("--lembrar", metavar="FATO", help="adiciona uma memória e sai")
    p.add_argument("--tipo", default="semantica",
                   choices=["semantica", "episodica", "procedural", "personalidade"])
    p.add_argument("--memorias", action="store_true", help="lista as memórias e sai")
    p.add_argument("--mensagem", "-m", help="envia uma mensagem e sai (sem loop)")
    args = p.parse_args(argv)

    cfg = carregar_config()
    eva = EVA(cfg)

    try:
        if args.diagnostico:
            mostrar_diagnostico(eva)
            return 0

        if args.lembrar:
            eva.lembrar(args.lembrar, args.tipo)
            print(f"guardado [{args.tipo}]: {args.lembrar}")
            return 0

        if args.memorias:
            for tipo in ("semantica", "episodica", "procedural", "personalidade"):
                itens = eva.memoria.listar(usuario=eva.cfg.usuario, tipo=tipo, limite=100)
                if itens:
                    print(f"\n{tipo}:")
                    for m in itens:
                        print(f"  - {m.conteudo}")
            print()
            return 0

        if args.mensagem:
            r = eva.responder(args.mensagem)
            if r.erro:
                print(f"[erro] {r.erro}", file=sys.stderr)
                return 1
            print(r.resposta)
            if args.debug:
                mostrar_debug(r)
            return 0

        loop_conversa(eva, args.debug)
        return 0
    finally:
        eva.fechar()


if __name__ == "__main__":
    sys.exit(main())