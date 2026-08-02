"""
Quem é a pessoa do outro lado.

Existe por causa de um detalhe do dataset: a segunda linha do system prompt
varia por situação, e são três formas apenas.

    criador       "Você está falando com Alex, seu criador."      138 exemplos
    conhecido     "Você está falando com alguém que você conhece bem."  210
    desconhecido  (nada -- só a âncora)                                 784

O terceiro caso ser o mais comum não é acidente: a EVA foi treinada para
funcionar bem sem saber com quem fala. Então na dúvida, não invente linha --
o silêncio é a forma mais treinada de todas.

Este módulo só decide qual das três usar e formata. O armazenamento fica no
banco de memórias (tabela `pessoas`), porque é lá que já mora tudo que é
persistente sobre gente.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context import LINHA_CONHECIDO, LINHA_CRIADOR

CRIADOR = "criador"
CONHECIDO = "conhecido"
DESCONHECIDO = "desconhecido"

RELACOES = (CRIADOR, CONHECIDO, DESCONHECIDO)


@dataclass
class Pessoa:
    user_id: str
    nome: str = ""
    relacao: str = DESCONHECIDO
    turnos: int = 0

    def linha(self) -> str | None:
        """A segunda linha do system prompt, ou None."""
        if self.relacao == CRIADOR:
            return LINHA_CRIADOR.format(nome=self.nome or "Alex")
        if self.relacao == CONHECIDO:
            return LINHA_CONHECIDO
        return None


def promover(pessoa: Pessoa, limiar: int) -> bool:
    """Desconhecido vira conhecido depois de conversa suficiente.

    Devolve True se mudou. O criador nunca é rebaixado nem promovido -- essa
    relação é definida à mão, não conquistada por volume.

    O limiar conta turnos, não dias: alguém que trocou 30 mensagens é mais
    "conhecido" que alguém que disse oi uma vez por mês durante um ano.
    """
    if pessoa.relacao != DESCONHECIDO:
        return False
    if pessoa.turnos < limiar:
        return False
    pessoa.relacao = CONHECIDO
    return True