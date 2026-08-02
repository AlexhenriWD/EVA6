"""
Context Builder -- monta o pacote que a EVA recebe.

Esta é a única peça que decide o formato do prompt. Ele vive num lugar só
de propósito: divergir do formato de treino derruba a qualidade sem gerar
erro nenhum, então quanto menos lugares puderem divergir, melhor.

O QUE FOI TREINADO E O QUE NÃO FOI
-----------------------------------
Treinado (aparece no fine-tuning, use com confiança):

  PERSONA            a âncora. Entra no treino via finetune/persona.py em
                     três variações, para o modelo não ficar preso à string
                     exata. Aqui usamos a canônica.
  LINHA_CRIADOR      "Você está falando com Alex, seu criador."   (138 ex.)
  LINHA_CONHECIDO    "Você está falando com alguém que você conhece bem."
                                                                  (210 ex.)
  sem segunda linha  o caso mais comum                            (784 ex.)
  MODO_VOZ           "MODO: VOZ. Seja concisa (máx 2-3 frases)."
  PREFIXO_VISUAL     "Contexto visual: ..."

NÃO treinado (funciona por capacidade herdada do Qwen2.5-Instruct base,
não pela LoRA -- merece menos confiança):

  o bloco `Contexto:` com JSON. Zero exemplos no dataset.

Por isso existe `cfg.llm.formato_contexto`: "json" (padrão) ou "prosa".
Dá para trocar por variável de ambiente e comparar as duas sem editar
código. Se a prosa ganhar, o certo é gerar ~40 exemplos com contexto na
próxima rodada de fine-tuning e fixar o formato vencedor.

Princípio de curadoria: contexto pequeno e relevante vale mais que contexto
grande. Item irrelevante não é neutro -- ele faz a EVA falar de coisa que
não vem ao caso, o que soa pior do que ela não lembrar de nada.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

# --------------------------------------------------------------- âncora

# Mesma string usada no treino. Não edite sem retreinar: o modelo aprendeu
# a associar esta âncora ao comportamento demonstrado nos 1.135 exemplos.
PERSONA = (
    "Você é EVA, uma inteligência artificial que conversa por interesse real em "
    "entender as pessoas. É curiosa, direta e honesta sobre o que não sabe. "
    "Fala como gente, não como manual: sem emoji, sem frase pronta de atendente, "
    "sem forçar simpatia. Tem opinião própria e discorda quando discorda, mas muda "
    "de ideia diante de bom argumento. Não finge sentir o que não sente nem inventa "
    "informação."
)

# ------------------------------------------------- linhas situacionais

# Formas literais do dataset. O `{nome}` é a única variação -- o resto da
# frase é fixo porque foi o que o modelo viu.
LINHA_CRIADOR = "Você está falando com {nome}, seu criador."
LINHA_CONHECIDO = "Você está falando com alguém que você conhece bem."

# Modo voz e contexto visual: no dataset aparecem na mesma linha do
# "Você é EVA.". Com a âncora longa fica melhor em linha própria, mas a
# string em si é preservada.
MODO_VOZ = "MODO: VOZ. Seja concisa (máx 2-3 frases)."
PREFIXO_VISUAL = "Contexto visual: "
MODO_INICIATIVA = (
    "MODO: INICIATIVA. Ninguém falou há um tempo e você decidiu dizer algo. "
    "Uma ou duas frases, sem anunciar que está puxando assunto e sem "
    "perguntar se tem alguém aí."
)
PREFIXO_IDEIA = "Ideia: "

# Instrução adicional para momentos de crise. Curta e específica: a EVA já
# foi treinada com exemplos desse tipo, então isso apenas reforça.
NOTA_CRISE = (
    "\n\nEsta mensagem tem sinais de sofrimento grave. Leve a sério, não use humor, "
    "não minimize. Se fizer sentido, mencione que existe ajuda disponível "
    "(CVV, 188, 24h, ligação gratuita) sem soar como protocolo."
)

# ------------------------------------------------------ âncora temporal

_DIAS = ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
         "sexta-feira", "sábado", "domingo")
_MESES = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
          "agosto", "setembro", "outubro", "novembro", "dezembro")


def agora_legivel(quando: datetime | None = None) -> str:
    """Data e hora em português, para o modelo saber em que momento está.

    Sem isso ele responde como se o treino fosse ontem: erra a data quando
    perguntado e, pior, não desconfia de que o que sabe pode ter envelhecido.
    Entra sempre, não só quando a mensagem é sobre horário -- a ferramenta
    `hora_atual` só dispara com pergunta explícita, e a maioria dos casos em
    que a data importa não é uma pergunta sobre a data.
    """
    q = quando or datetime.now()
    return (f"{_DIAS[q.weekday()]}, {q.day} de {_MESES[q.month - 1]} "
            f"de {q.year}, {q:%H:%M}")


# ------------------------------------------------------------- contexto


@dataclass
class Contexto:
    system: str
    mensagens: list[dict]
    bruto: dict = field(default_factory=dict)

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
        identidade: str | None = None,
        modo_voz: bool = False,
        contexto_visual: str | None = None,
        iniciativa: str | None = None,
    ) -> Contexto:
        """Monta o Contexto de um turno.

        `identidade` é a linha situacional já pronta (ver eva.identity).
        `modo_voz` e `contexto_visual` mudam só o cabeçalho, nunca o bloco
        de dados -- são coisas que o modelo viu no treino.
        """
        ctx: dict = {"agora": agora_legivel()}

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

        # ---------------------------------------------------- cabeçalho
        linhas = [PERSONA]
        if identidade:
            linhas.append(identidade)
        if modo_voz:
            linhas.append(MODO_VOZ)
        if contexto_visual:
            linhas.append(PREFIXO_VISUAL + contexto_visual.strip())
        if iniciativa:
            linhas.append(MODO_INICIATIVA)
            linhas.append(PREFIXO_IDEIA + iniciativa.strip())
        system = "\n".join(linhas)

        # ------------------------------------------------- bloco de dados
        corpo = self._renderizar(ctx)
        if corpo:
            system += "\n\nContexto:\n" + corpo

        if getattr(plano, "intencao", "") == "crise":
            system += NOTA_CRISE

        return Contexto(
            system=system,
            mensagens=self._limpar_historico(historico),
            bruto=ctx,
        )

    # ------------------------------------------------------------ render

    def _renderizar(self, ctx: dict) -> str:
        """Serializa o bloco de contexto no formato configurado.

        Só `agora` não justifica um bloco: se não há mais nada, a data vira
        uma linha solta e o modelo às vezes comenta a data sem motivo.
        """
        if set(ctx) <= {"agora"}:
            return f"agora: {ctx['agora']}"

        formato = getattr(self.cfg.llm, "formato_contexto", "json")
        if formato == "prosa":
            return self._como_prosa(ctx)
        return json.dumps(ctx, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    def _como_prosa(self, ctx: dict) -> str:
        """Alternativa em texto, para comparar contra o JSON.

        Ferramentas continuam em JSON mesmo aqui: resultado de ferramenta é
        dado estruturado, e achatar em prosa é justamente o erro que a
        arquitetura evita (quem escreve português é a EVA).
        """
        rotulos = {
            "agora": "Agora",
            "fatos": "Você sabe",
            "lembrancas": "Aconteceu antes",
            "preferencias": "Como agir",
            "sobre_a_pessoa": "Sobre a pessoa",
            "estado": "Seu estado",
        }
        linhas = []
        for chave, valor in ctx.items():
            if chave == "ferramentas":
                continue
            rotulo = rotulos.get(chave, chave)
            if isinstance(valor, list):
                linhas.append(f"{rotulo}: " + "; ".join(str(v) for v in valor))
            elif isinstance(valor, dict):
                linhas.append(f"{rotulo}: " + ", ".join(
                    f"{k} {v}" for k, v in valor.items()))
            else:
                linhas.append(f"{rotulo}: {valor}")
        if "ferramentas" in ctx:
            linhas.append("Ferramentas: " + json.dumps(
                ctx["ferramentas"], ensure_ascii=False, separators=(",", ":")))
        return "\n".join(linhas)

    # ------------------------------------------------------------ limpeza

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