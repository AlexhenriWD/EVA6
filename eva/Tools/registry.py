"""
Registro de ferramentas.

Regra do projeto: ferramenta NUNCA retorna texto em portugues, so JSON.
Quem escreve portugues e a EVA. Isso separa o que e dado do que e voz --
se a ferramenta devolvesse "A previsão é 25 graus", a resposta final teria
um pedaco escrito por outra pessoa, com outro tom.

Toda ferramenta:
- recebe argumentos nomeados
- devolve dict serializavel
- em caso de falha devolve {"erro": "..."} em vez de levantar excecao

O ultimo ponto importa: uma ferramenta quebrada nao pode derrubar a
conversa. A EVA foi treinada para lidar com {"erro": ...} de forma honesta
("a busca deu timeout, não vou chutar"), entao o erro estruturado e mais
util que uma excecao.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Ferramenta:
    nome: str
    descricao: str
    funcao: Callable[..., dict]
    parametros: dict[str, str]
    # Ferramentas caras (rede, API paga) so rodam quando o Decision Engine
    # pede explicitamente. As baratas podem rodar por heuristica.
    cara: bool = False


class RegistroFerramentas:
    def __init__(self):
        self._ferramentas: dict[str, Ferramenta] = {}

    def registrar(self, ferramenta: Ferramenta) -> None:
        self._ferramentas[ferramenta.nome] = ferramenta

    def adicionar(self, nome: str, descricao: str, parametros: dict[str, str] | None = None,
                  cara: bool = False):
        """Decorador para registrar uma funcao como ferramenta."""
        def wrapper(fn):
            self.registrar(Ferramenta(nome, descricao, fn, parametros or {}, cara))
            return fn
        return wrapper

    def get(self, nome: str) -> Ferramenta | None:
        return self._ferramentas.get(nome)

    def listar(self) -> list[Ferramenta]:
        return list(self._ferramentas.values())

    def nomes(self) -> list[str]:
        return list(self._ferramentas)

    def descrever(self) -> str:
        """Descricao das ferramentas, para o Decision Engine escolher."""
        return "\n".join(
            f"- {f.nome}: {f.descricao}"
            + (f" (parâmetros: {', '.join(f.parametros)})" if f.parametros else "")
            for f in self._ferramentas.values()
        )

    def executar(self, nome: str, **kwargs) -> dict:
        """Executa uma ferramenta. Nunca levanta excecao."""
        f = self.get(nome)
        if not f:
            return {"erro": "ferramenta_desconhecida", "nome": nome}

        inicio = time.time()
        try:
            resultado = f.funcao(**kwargs)
            if not isinstance(resultado, dict):
                resultado = {"valor": resultado}
        except TypeError as e:
            # argumentos errados sao erro de chamada, nao da ferramenta
            return {"erro": "argumentos_invalidos", "detalhe": str(e)}
        except Exception as e:
            return {"erro": type(e).__name__, "detalhe": str(e)[:200]}

        resultado["_ms"] = int((time.time() - inicio) * 1000)
        return resultado


registro = RegistroFerramentas()
