"""
Mundo interno da EVA.

A ideia (do documento original do projeto): a EVA nao tem so memoria, tem
ESTADO. Curiosidade, energia, confianca, estresse -- valores que mudam
devagar e influenciam como ela responde.

Duas decisoes de design importantes:

1. O estado muda com INERCIA alta. Se ele reagisse imediatamente ao ultimo
   turno, viraria humor volatil: eufórica numa mensagem, apática na
   seguinte. Com inercia, ele se comporta como disposicao acumulada.

2. O estado NAO e verbalizado como numero cru. Ate a sessao de hoje, isso
   dependia do modelo fine-tunado ter aprendido a tratar esses valores
   como "nao-verbalizaveis" -- premissa que nunca se aplicou a nenhum
   modelo generico (Lumimaid e afins), que nunca viu essa convencao no
   proprio treino. Incidente real confirmado: numero decimal cru
   ("energia 0.94") disparou narracao tipo "estatistica de personagem
   decaindo" ("0.34 -> 0.32 (motivo)"), que nunca existiu no codigo --
   pura alucinacao -- e se auto-reforcou via historico de conversa turno
   a turno. Por isso agora e sempre texto qualitativo (baixa/moderada/
   alta), nao numero, independente de qual modelo estiver rodando.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _descrever_nivel(valor: float) -> str:
    """0-1 -> texto qualitativo. Ver docstring do modulo -- numero cru
    nesse lugar especifico ja causou alucinacao real de narracao de
    estatistica; texto tira o gatilho independente do modelo."""
    if valor < 0.35:
        return "baixa"
    if valor < 0.7:
        return "moderada"
    return "alta"


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

        Descritiva, NAO numerica -- incidente real confirmado: expor
        decimal cru ("energia 0.94, curiosidade 0.57...") fez um modelo de
        RP (Lumimaid) alucinar uma narracao de "estatistica decaindo"
        tipo "0.34 -> 0.32 (motivo)" que nunca existiu no codigo -- nada
        aqui jamais gerou esse formato, e a busca confirma isso. Uma vez
        gerado, entrava no historico como fala real dela e o proprio
        historico ensinava a repetir, degradando turno a turno (visto
        caindo ate 0.00 em sessao real). Numero decimal convida esse tipo
        de leitura "e um jogo com estatistica" em modelo treinado nesse
        genero; texto qualitativo tira o gatilho.
        """
        d = {
            "energia": _descrever_nivel(self.energia),
            "curiosidade": _descrever_nivel(self.curiosidade),
            "confianca": _descrever_nivel(self.confianca),
        }
        # Estresse e diferente dos outros -- 0 e o caso comum, e mencionar
        # "estresse: baixo" toda hora e ruido. So entra quando de fato
        # subiu o bastante pra importar.
        if self.estresse >= 0.35:
            d["estresse"] = _descrever_nivel(self.estresse)
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