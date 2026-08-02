"""
Context Builder -- monta o pacote que a EVA recebe.

Esta é a peça que conecta o resto do sistema ao modelo conversacional, e o
formato aqui não é arbitrário: é exatamente o formato usado no fine-tuning
(`Você é EVA.` + bloco `Contexto:` com JSON). O modelo viu isso no treino,
então sabe usar fato de memória sem anunciar que lembrou, e sabe reagir a
{"erro": ...} de ferramenta sem inventar.

Se este formato divergir do que foi treinado, a qualidade cai sem erro
aparente -- por isso ele vive num lugar só, e não espalhado pelo código.

Princípio de curadoria: contexto pequeno e relevante vale mais que contexto
grande. Item irrelevante não é neutro -- ele faz a EVA falar de coisa que
não vem ao caso, o que soa pior do que ela não lembrar de nada.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PERSONA = (
    "Você é EVA, uma inteligência artificial que conversa por interesse real em "
    "entender as pessoas. É curiosa, direta e honesta sobre o que não sabe. "
    "Fala como gente, não como manual: sem emoji, sem frase pronta de atendente, "
    "sem forçar simpatia. Tem opinião própria e discorda quando discorda, mas muda "
    "de ideia diante de bom argumento. Não finge sentir o que não sente nem inventa "
    "informação."
)

# Instrução adicional para momentos de crise. Curta e específica: a EVA já
# foi treinada com exemplos desse tipo, então isso apenas reforça.
NOTA_CRISE = (
    "\n\nEsta mensagem tem sinais de sofrimento grave. Leve a sério, não use humor, "
    "não minimize. Se fizer sentido, mencione que existe ajuda disponível "
    "(CVV, 188, 24h, ligação gratuita) sem soar como protocolo."
)


@dataclass
class Contexto:
    system: str
    mensagens: list[dict]
    bruto: dict  # o que foi montado, para debug e log

    def para_chat(self, mensagem_usuario: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        msgs.extend(self.mensagens)
        msgs.append({"role": "user", "content": mensagem_usuario})
        return msgs


class ContextBuilder:
    def __init__(self, config):
        self.cfg = config

    def montar(
        self,
        plano,
        memorias: dict[str, list],
        resultados_ferramentas: dict,
        estado,
        historico: list[dict],
    ) -> Contexto:
        ctx: dict = {}

        # --- memória semântica: fatos sobre a pessoa ---
        fatos = [m.conteudo for m in memorias.get("semantica", [])]
        if fatos:
            ctx["fatos"] = fatos[: self.cfg.memoria.max_fatos]

        # --- memória episódica: o que aconteceu ---
        episodios = [m.conteudo for m in memorias.get("episodica", [])]
        if episodios:
            ctx["lembrancas"] = episodios[: self.cfg.memoria.max_episodios]

        # --- memória procedural: como agir com essa pessoa ---
        procedimentos = [m.conteudo for m in memorias.get("procedural", [])]
        if procedimentos:
            ctx["preferencias"] = procedimentos[: self.cfg.memoria.max_procedimentos]

        # --- memória de personalidade: o que funciona ---
        personalidade = [m.conteudo for m in memorias.get("personalidade", [])]
        if personalidade:
            ctx["sobre_a_pessoa"] = personalidade[: self.cfg.memoria.max_personalidade]

        # --- ferramentas: JSON puro, como veio ---
        if resultados_ferramentas:
            ctx["ferramentas"] = self._limpar_ferramentas(resultados_ferramentas)

        # --- estado interno ---
        if estado:
            ctx["estado"] = estado.para_contexto()

        system = PERSONA
        if ctx:
            system += "\n\nContexto:\n" + json.dumps(
                ctx, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        if getattr(plano, "intencao", "") == "crise":
            system += NOTA_CRISE

        return Contexto(
            system=system,
            mensagens=self._limpar_historico(historico),
            bruto=ctx,
        )

    def _limpar_ferramentas(self, resultados: dict) -> dict:
        """Remove metadados internos que não interessam ao modelo."""
        limpo = {}
        for nome, r in resultados.items():
            if isinstance(r, dict):
                limpo[nome] = {k: v for k, v in r.items() if not k.startswith("_")}
            else:
                limpo[nome] = r
        return limpo

    def _limpar_historico(self, historico: list[dict]) -> list[dict]:
        """Mantém só role e content, na janela configurada."""
        janela = self.cfg.memoria.janela_historico
        return [
            {"role": t["role"], "content": t["content"]}
            for t in historico[-janela:]
            if t.get("role") in ("user", "assistant") and t.get("content")
        ]
