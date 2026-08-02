"""
Simulador do portão de fala.

Roda cenários de call em tempo virtual: nada de asyncio, nada de GPU, nada
de Discord. Serve para você calibrar os limiares vendo o comportamento, em
vez de descobrir numa call que ela fala demais.

    python ferramentas/simular_portao.py
    python ferramentas/simular_portao.py --curiosidade 0.95 --silencio 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eva.config import carregar_config
from eva.consciousness import Consciencia
from eva.state import EstadoInterno

VERDE = "\033[92m"
CINZA = "\033[90m"
AMARELO = "\033[93m"
FIM = "\033[0m"


def cenario(nome, cfg, estado, roteiro, passo=5.0, duracao=300.0):
    """`roteiro` é uma lista de (segundo, usuario, mensagem)."""
    print(f"\n{AMARELO}━━ {nome} ━━{FIM}")
    c = Consciencia(cfg, canal="sim")
    t = 0.0
    c.ultima_fala_alguem = t
    c.ultima_fala_dela = t
    falas = 0
    pendentes = sorted(roteiro)

    while t < duracao:
        while pendentes and pendentes[0][0] <= t:
            _, usuario, msg = pendentes.pop(0)
            c.alguem_falou(usuario, msg)
            c.ultima_fala_alguem = t
            print(f"  {CINZA}{t:6.0f}s{FIM} {usuario}: {msg}")
            fios = [f.assunto for f in c.fios if not f.usado]
            if fios:
                print(f"  {CINZA}{'':6} └ fio: {fios[-1]}{FIM}")

        v = c.tick(estado, agora=t)
        if v.passou:
            falas += 1
            print(f"  {VERDE}{t:6.0f}s EVA →{FIM} {v.impulso.conteudo}")
            print(f"  {CINZA}{'':6} └ {v}{FIM}")
            c.ela_falou(espontanea=True)
            c.ultima_fala_dela = t

        t += passo

    print(f"  {CINZA}total: {falas} fala(s) espontânea(s) em {duracao:.0f}s"
          f" | limiar final {c.portao.limiar(estado, c.falas_sem_resposta):.2f}{FIM}")
    return falas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--curiosidade", type=float, default=0.8)
    p.add_argument("--energia", type=float, default=0.7)
    p.add_argument("--estresse", type=float, default=0.1)
    p.add_argument("--silencio", type=float, default=None,
                   help="sobrescreve silencio_minimo")
    p.add_argument("--cooldown", type=float, default=None)
    p.add_argument("--limiar", type=float, default=None)
    args = p.parse_args()

    cfg = carregar_config()
    if args.silencio is not None:
        cfg.consciencia.silencio_minimo = args.silencio
    if args.cooldown is not None:
        cfg.consciencia.cooldown_fala = args.cooldown
    if args.limiar is not None:
        cfg.consciencia.limiar_base = args.limiar

    estado = EstadoInterno(curiosidade=args.curiosidade, energia=args.energia,
                           estresse=args.estresse)

    print(f"{CINZA}silêncio mín {cfg.consciencia.silencio_minimo:.0f}s | "
          f"cooldown {cfg.consciencia.cooldown_fala:.0f}s | "
          f"limiar base {cfg.consciencia.limiar_base:.2f} | "
          f"curiosidade {estado.curiosidade:.2f} estresse {estado.estresse:.2f}{FIM}")

    # Conversa animada: ninguém para de falar. Ela não deve abrir a boca.
    cenario("conversa animada — esperado: 0 falas", cfg, estado,
            [(t, "alex", "papo") for t in range(0, 300, 20)])

    # Silêncio total do início ao fim, sem nenhum fio.
    cenario("silêncio absoluto, sem fio — esperado: pouquíssimas", cfg, estado, [])

    # Alguém menciona algo e a call esvazia. É o caso que ela deve acertar.
    cenario("menção e depois silêncio — esperado: retoma o fio", cfg, estado,
            [(0, "alex", "comecei um projeto novo de robótica ontem"),
             (10, "joao", "boa"),
             (20, "alex", "vou testar o servo amanhã")])

    # Conversa que morre aos poucos.
    cenario("conversa esfriando", cfg, estado,
            [(0, "alex", "descobri um problema no motor"),
             (15, "joao", "hm"),
             (30, "alex", "é"),
             (120, "joao", "voltei")])

    # Estresse alto: o portão deve praticamente fechar.
    estressada = EstadoInterno(curiosidade=0.9, energia=0.7, estresse=0.85)
    cenario("estresse alto — esperado: cala a boca", cfg, estressada,
            [(0, "alex", "comecei um tratamento novo essa semana")])


if __name__ == "__main__":
    main()
