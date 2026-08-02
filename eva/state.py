"""
Mundo interno da EVA.

A ideia (do documento original do projeto): a EVA nao tem so memoria, tem
ESTADO. Curiosidade, energia, confianca, estresse -- valores que mudam
devagar e influenciam como ela responde.

Duas decisoes de design importantes:

1. O estado muda com INERCIA alta. Se ele reagisse imediatamente ao ultimo
   turno, viraria humor volatil: eufórica numa mensagem, apática na
   seguinte. Com inercia, ele se comporta como disposicao acumulada.

2. O estado NAO e verbalizado como numero. Ele entra no contexto como
   dado estruturado, e o modelo foi treinado para deixar isso aparecer no
   comportamento -- nao para dizer "minha curiosidade esta em 0.91".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EstadoInterno:
    energia: float = 0.7
    curiosidade: float = 0.8
    confianca: float = 0.6
    estresse: float = 0.1

    foco: str | None = None          # assunto que vem dominando as conversas
    ultima_reflexao: str | None = None
    atualizado_em: float = field(default_factory=time.time)
    total_interacoes: int = 0

    def clamp(self) -> None:
        """Mantem tudo no intervalo [0, 1]."""
        for campo in ("energia", "curiosidade", "confianca", "estresse"):
            v = getattr(self, campo)
            setattr(self, campo, max(0.0, min(1.0, float(v))))

    def para_contexto(self) -> dict:
        """Versao enxuta para o Context Builder.

        Arredonda para 2 casas: precisao maior nao muda comportamento e so
        gasta token. Campos vazios sao omitidos.
        """
        d = {
            "energia": round(self.energia, 2),
            "curiosidade": round(self.curiosidade, 2),
            "confianca": round(self.confianca, 2),
            "estresse": round(self.estresse, 2),
        }
        if self.foco:
            d["foco"] = self.foco
        return d


class GerenciadorEstado:
    """Carrega, atualiza e persiste o estado interno."""

    def __init__(self, caminho: Path, inercia: float = 0.92):
        self.caminho = Path(caminho)
        self.inercia = inercia
        self.estado = self._carregar()

    def _carregar(self) -> EstadoInterno:
        if not self.caminho.exists():
            return EstadoInterno()
        try:
            with open(self.caminho, encoding="utf-8") as f:
                d = json.load(f)
            campos = {k: v for k, v in d.items() if k in EstadoInterno.__annotations__}
            return EstadoInterno(**campos)
        except (json.JSONDecodeError, TypeError, OSError):
            # estado corrompido nao deve derrubar a EVA -- recomeca do padrao
            return EstadoInterno()

    def salvar(self) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.caminho.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(self.estado), f, ensure_ascii=False, indent=2)
        tmp.replace(self.caminho)  # escrita atomica

    def _mover(self, campo: str, alvo: float, forca: float = 1.0) -> None:
        """Move um valor na direcao do alvo, respeitando a inercia."""
        atual = getattr(self.estado, campo)
        peso = (1 - self.inercia) * forca
        setattr(self.estado, campo, atual * (1 - peso) + alvo * peso)

    def registrar_interacao(self, sinais: dict) -> EstadoInterno:
        """Atualiza o estado a partir de sinais da interacao.

        Sinais esperados (todos opcionais):
            novidade: 0-1   -- quanto o assunto e novo/inesperado
            complexidade: 0-1
            carga_emocional: 0-1
            sucesso: bool   -- a EVA conseguiu ajudar/responder bem
            assunto: str
        """
        e = self.estado
        e.total_interacoes += 1

        novidade = float(sinais.get("novidade", 0.5))
        complexidade = float(sinais.get("complexidade", 0.5))
        carga = float(sinais.get("carga_emocional", 0.0))

        # Curiosidade sobe com novidade -- assunto repetido nao empolga.
        self._mover("curiosidade", 0.35 + 0.6 * novidade, forca=1.0)

        # Energia cai com uso, e cai mais rapido em conversa densa.
        #
        # Decaimento PROPORCIONAL, nao linear: a queda e uma fracao da
        # distancia ate um piso, entao a energia se aproxima do piso
        # assintoticamente em vez de bater nele. Com subtracao fixa ela
        # zerava por volta do turno 40, o que deixava a EVA "exausta" em
        # qualquer conversa longa -- e cansaco permanente nao e humor, e bug.
        piso = 0.25
        taxa = 0.010 + 0.020 * complexidade + 0.015 * carga
        e.energia = piso + (e.energia - piso) * (1 - taxa) if e.energia > piso else e.energia

        # Confianca sobe quando ela consegue responder bem, cai quando nao.
        if "sucesso" in sinais:
            self._mover("confianca", 0.85 if sinais["sucesso"] else 0.35, forca=0.8)

        # Estresse acompanha carga emocional da conversa, com inercia maior
        # ainda -- ele deve subir devagar e descer devagar.
        self._mover("estresse", carga, forca=0.6)

        if sinais.get("assunto"):
            e.foco = sinais["assunto"]

        e.atualizado_em = time.time()
        e.clamp()
        self.salvar()
        return e

    def descansar(self, horas: float) -> EstadoInterno:
        """Recupera energia e alivia estresse com o tempo parado.

        Chamado quando ha um intervalo grande entre conversas -- e o que
        evita a EVA acumular cansaco para sempre.
        """
        e = self.estado
        recuperacao = min(1.0, horas / 8.0)  # 8h restaura quase tudo
        e.energia = min(1.0, e.energia + 0.9 * recuperacao)
        e.estresse = max(0.0, e.estresse * (1 - 0.7 * recuperacao))
        e.atualizado_em = time.time()
        e.clamp()
        self.salvar()
        return e

    def aplicar_tempo_decorrido(self) -> EstadoInterno:
        """Aplica descanso proporcional ao tempo desde a ultima interacao."""
        horas = (time.time() - self.estado.atualizado_em) / 3600
        if horas > 0.5:
            return self.descansar(horas)
        return self.estado
