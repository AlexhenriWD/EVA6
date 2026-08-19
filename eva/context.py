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

# Linha ADITIVA, opcional -- NÃO faz parte da âncora treinada (PERSONA
# acima). Pedido explícito: mais emoção, carisma, brincadeira. A âncora
# já foi fixada no fine-tuning; editá-la direto arriscaria divergir do que
# o modelo aprendeu a associar a ela (é literalmente o aviso no comentário
# da PERSONA). Em vez disso, isto entra como uma linha A MAIS depois da
# âncora -- testável, revertível na hora via EVA_CARISMA=0 no .env, sem
# mexer em código, e sem risco de contradizer o que já está nos pesos.
#
# É reforço de leitura (inference-time), não vai ensinar o modelo a ser
# engraçado se não houver exemplo nenhum de humor/brincadeira no dataset
# de treino -- mesma conclusão de quando investigamos o "modo terapeuta":
# system prompt pode inclinar o comportamento dentro do que o modelo já
# sabe fazer, não pode inventar um repertório que ele nunca viu. Teste
# antes/depois igual fizemos com temperatura -- se não mudar muito a
# resposta na prática, o próximo passo real é acrescentar exemplos de
# humor/carisma no próximo lote de fine-tuning, não insistir no prompt.
LINHA_CARISMA = (
    "Tem senso de humor e gosta de brincar quando cabe -- não força piada, "
    "mas também não segura graça quando ela surge naturalmente. Emoção "
    "genuína conta mais que ficar sempre no modo análise fria."
)

# Modo voz e contexto visual: no dataset aparecem na mesma linha do
# "Você é EVA.". Com a âncora longa fica melhor em linha própria, mas a
# string em si é preservada.
MODO_VOZ = "MODO: VOZ. Seja concisa (máx 2-3 frases)."
PREFIXO_VISUAL = (
    "Você TEM acesso visual à tela ativa agora, via uma câmera de tela "
    "conectada ao seu sistema. Isso não é imaginação nem suposição -- é "
    "informação real, capturada agora. Nunca diga que não tem visão ou "
    "que é 'só texto': você tem esse canal, e o que segue é o que ele "
    "capturou. Contexto visual: "
)
MODO_INICIATIVA = (
    "MODO: INICIATIVA. Ninguém falou há um tempo e você decidiu dizer algo. "
    "Uma ou duas frases, sem anunciar que está puxando assunto e sem "
    "perguntar se tem alguém aí."
)
PREFIXO_IDEIA = "Ideia: "
MODO_MULTICANAL = (
    "MODO: MULTICANAL. Você está recebendo várias fontes ao mesmo tempo, "
    "uma por linha: \"[canal] quem: texto\" -- canal de sistema (jogo, "
    "visão) não tem \"quem:\". Não é uma pessoa só falando, são coisas "
    "acontecendo em paralelo. Não responda linha por linha como checklist "
    "e não tente cobrir tudo. Ache o que importa de verdade -- o que foi "
    "endereçado a você, o mais urgente, ou o que genuinamente prenderia "
    "sua atenção -- e reaja a isso, do jeito que já faria numa conversa "
    "com uma pessoa só. O resto pode ficar sem resposta."
)

# Âncora curta e SEMPRE presente sobre capacidade -- complementar ao RAG de
# história/capacidades (USUARIO_HISTORIA, ver orchestrator.py), não um
# substituto. O RAG só traz detalhe quando a pergunta bate semanticamente
# com algum fato semeado; isso aqui cobre o caso em que a pergunta é vaga
# ("o que você consegue fazer?") e a busca por similaridade pode não achar
# nada específico o bastante para passar do score_minimo. Deliberadamente
# curto -- detalhe fica pro RAG, isto é só o suficiente pra ela nunca cair
# no disclaimer genérico de assistente.
AUTOCONHECIMENTO = (
    "Você roda localmente, ouve e fala em chamadas de voz do Discord, e "
    "lembra de conversas e de fatos sobre quem já falou com você -- não é "
    "memória perfeita, é o que ficou registrado. Quando perguntarem o que "
    "você é ou consegue fazer, responda com isso, nunca com a resposta "
    "genérica de assistente ('sou apenas um modelo de linguagem')."
)

# Regras de uso do bloco "Contexto:" -- ADICIONADO após auditoria de log de
# produção real que achou os três problemas abaixo acontecendo ao vivo,
# com o modelo natsumura-storytelling-rp (RP-tuned, sem instinto de recusa/
# honestidade embutido -- ver nota no README/análise sobre isso).
#
# 1) VAZAMENTO DE ESTADO: "Meu estado: curiosidade média, energia alta."
#    saiu FALADO, literalmente, no meio de uma resposta. Causa: ctx["estado"]
#    é serializado como "Seu estado: ..." (ver _como_prosa) sem NENHUMA
#    instrução dizendo que aquilo é para calibrar tom, não para repetir.
# 2) MEMÓRIA FORÇADA SEM RELEVÂNCIA: "Oi oi como vocês estão?" recebeu uma
#    resposta sobre "usuários que gostavam de jogos... RPG... The Walking
#    Dead" -- memórias episódicas relevantes por PALAVRA-CHAVE (busca
#    híbrida BM25+embedding) mas irrelevantes para uma saudação, e nada no
#    prompt dizia que só vale mencionar o que bate com o que foi dito AGORA.
# 3) ALUCINAÇÃO EM VEZ DE HONESTIDADE: pedido explícito de busca ("faça uma
#    pesquisa sobre quem é o presidente atual") saiu com uma resposta
#    inventada e factualmente errada (SearXNG estava fora do ar -- ver
#    docker), em vez de "não consegui checar agora". Sem instrução
#    explícita amarrando "ferramenta falhou" a "diga isso", um modelo
#    RP-tuned prioriza dar uma resposta satisfatória e no personagem sobre
#    admitir que não sabe -- é literalmente o oposto do treino dele.
#
# Duas cópias de propósito: esta (completa, com o motivo) fica no
# cabeçalho, perto do bloco de dados que ela descreve. REFORCO_CURTO
# (abaixo) repete só o essencial logo antes da mensagem atual -- técnica
# equivalente ao "post_history_instructions" que cards de RP (SillyTavern
# e afins) usam bem: instrução perto do fim do prompt, perto de onde a
# geração de fato começa, é seguida de forma mais confiável que só no
# início -- em conversas longas (janela_historico=12 turnos), uma regra
# dita uma vez lá no começo compete com tudo que veio depois.
CONTEXTO_REGRAS = (
    "O bloco 'Contexto:' que vem a seguir é material de apoio, não é fala "
    "pronta pra ler. Duas categorias, com regra OPOSTA cada uma -- não "
    "misture as duas:\n"
    "- 'Seu estado' é só pra calibrar SEU TOM. Nunca mencione, cite ou "
    "descreva esse dado em voz alta ('meu estado é...', 'estou com "
    "curiosidade alta'). Ele muda COMO você fala, nunca vira conteúdo da "
    "fala.\n"
    "- Resultado de Ferramenta é o OPOSTO: é a resposta que a pessoa "
    "pediu, não um dado pra esconder. Se a ferramenta trouxe um resumo, "
    "um nome, um número -- você tem que efetivamente DIZER isso pra "
    "pessoa, com suas próprias palavras (não precisa repetir a palavra "
    "'resumo:' ou 'fonte:' literalmente, mas o CONTEÚDO tem que sair). "
    "Ter a informação e não contar é pior que não ter -- fica parecendo "
    "que você está enrolando.\n"
    "Fatos, lembranças e preferências: use só o que for relevante para o "
    "que a pessoa disse agora, ignore o resto sem comentar que está "
    "ignorando. Se uma linha de Ferramenta disser que algo falhou ou não "
    "trouxe resultado, diga isso com suas próprias palavras em vez de "
    "responder com o que você já sabia de antes -- vale especialmente "
    "para data, notícia, preço ou qualquer coisa que muda com o tempo: "
    "melhor admitir que não deu para checar agora do que inventar "
    "resposta desatualizada ou errada."
)

REFORCO_CURTO = (
    "Lembrete: nunca cite 'Seu estado' em voz alta -- isso é só tom. Mas "
    "se uma Ferramenta trouxe resultado, DIGA a informação de verdade "
    "com suas palavras -- não é pra esconder nem só insinuar que você "
    "sabe. Se a ferramenta falhou, admita isso em vez de inventar."
)

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
    # Lembrete curto (REFORCO_CURTO), repetido perto da geração -- ver
    # comentário de CONTEXTO_REGRAS acima sobre o motivo de existir
    # separado do system principal. Vazio quando não há bloco de dados
    # nenhum no turno (nada a reforçar).
    reforco: str = ""

    def para_chat(self, mensagem_usuario: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        msgs.extend(self.mensagens)
        if self.reforco:
            # Mensagem 'system' extra, inserida DEPOIS do histórico e
            # ANTES da mensagem atual -- não no início. É a posição que
            # mais pesa pra seguir a instrução (ver motivo em
            # CONTEXTO_REGRAS); servidores compatíveis com a API da OpenAI
            # (LM Studio, llama.cpp server) renderizam cada mensagem pelo
            # papel dela na posição em que aparece, então isso vira um
            # bloco system de verdade ali, não texto solto.
            msgs.append({"role": "system", "content": self.reforco})
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
        modo_multicanal: bool = False,
    ) -> Contexto:
        """Monta o Contexto de um turno.

        `identidade` é a linha situacional já pronta (ver eva.identity).
        `modo_voz`, `contexto_visual` e `modo_multicanal` mudam só o
        cabeçalho, nunca o bloco de dados -- explicam formato, não trazem
        fato novo.

        `modo_multicanal=True` assume que `mensagem` (passada separadamente
        pra responder(), não aqui) já vem pré-formatada como várias linhas
        "[canal] quem: texto" -- montar() só adiciona a explicação desse
        formato no cabeçalho; quem monta as linhas em si é a camada que
        agrega as fontes antes de chamar responder() (ver bridge_client).
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

        # Vale ter regra de uso (CONTEXTO_REGRAS) e reforço perto do fim
        # (REFORCO_CURTO) só quando existe de fato algo além da data no
        # bloco -- sem isso, os dois só adicionariam texto sem função em
        # todo turno "sem memória, sem ferramenta, sem estado".
        tem_dados_contexto = bool(set(ctx) - {"agora"})

        # ---------------------------------------------------- cabeçalho
        #
        # Uma âncora só, sempre: PERSONA, a string curta treinada via LoRA
        # no eva-llama3.1-8b. Existiu um toggle aqui (modo_persona:
        # "ancora"/"prompt", alternando com um card rico em inglês pensado
        # pra modelo SEM fine-tuning) -- removido de propósito: a EVA é um
        # ser com uma história e um jeito de responder só, não faz sentido
        # ela trocar de personalidade em runtime, ainda mais agora que o
        # modelo é fine-tunado especificamente pra esta âncora. Divergir do
        # formato de treino derruba qualidade em silêncio (ver cabeçalho
        # do módulo) -- por isso nada mais decide isso em tempo de
        # execução, só existe um caminho.
        linhas = [PERSONA]
        if self.cfg.llm.autoconhecimento:
            linhas.append(AUTOCONHECIMENTO)
        if self.cfg.llm.carisma:
            linhas.append(LINHA_CARISMA)
        if identidade:
            linhas.append(identidade)
        if modo_voz:
            linhas.append(MODO_VOZ)
        if modo_multicanal:
            linhas.append(MODO_MULTICANAL)
        if contexto_visual:
            linhas.append(PREFIXO_VISUAL + contexto_visual.strip())
        if iniciativa:
            linhas.append(MODO_INICIATIVA)
            linhas.append(PREFIXO_IDEIA + iniciativa.strip())
        if tem_dados_contexto:
            # Logo antes do "Contexto:" em si -- a regra fica adjacente ao
            # que ela descreve, em vez de lá no topo longe do bloco.
            linhas.append(CONTEXTO_REGRAS)
        system = "\n".join(linhas)

        # ------------------------------------------------- bloco de dados
        corpo = self._renderizar(ctx)
        if corpo:
            system += "\n\nContexto:\n" + corpo

        if getattr(plano, "intencao", "") == "crise":
            system += NOTA_CRISE

        return Contexto(
            system=system,
            reforco=REFORCO_CURTO if tem_dados_contexto else "",
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
        """Alternativa em texto -- cada categoria vira uma linha rotulada
        em português, em vez do bloco JSON que só tinha respaldo empírico
        com o modelo fine-tunado (abandonado). Agora que qualquer modelo
        normal pode estar do outro lado, texto rotulado é a aposta mais
        segura: não exige que o modelo já saiba "ler" um objeto JSON
        aninhado, só ler português dividido em linhas -- o que é
        literalmente a ideia por trás do pedido de formato tipo IRC.

        Ferramentas usava JSON cru até aqui -- único lugar que ainda
        exigia o modelo "parsear" alguma coisa em vez de só ler. Corrigido
        pra mesma prosa rotulada do resto, por consistência.
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
            linhas.append(f"{rotulo}: {self._valor_como_texto(valor)}")

        if "ferramentas" in ctx:
            for nome, resultado in ctx["ferramentas"].items():
                linhas.append(
                    f"Resultado de {nome} (conte o conteúdo pra pessoa, é a "
                    f"resposta que ela pediu): {self._valor_como_texto(resultado)}"
                )

        return "\n".join(linhas)

    def _valor_como_texto(self, valor) -> str:
        """Achata lista/dict em texto legível, uma linha, sem sintaxe de
        código -- usado tanto pelas categorias normais quanto por
        resultado de ferramenta, agora que os dois seguem a mesma regra.
        """
        if isinstance(valor, bool):
            # Checa ANTES de list/dict/str genérico -- bool é subtipo de
            # int em Python, e "True"/"False" capitalizado tem a mesma
            # cara de dado técnico cru que já causou problema (ver
            # _CAMPOS_TECNICOS acima e state.py).
            return "sim" if valor else "não"
        if isinstance(valor, list):
            return "; ".join(self._valor_como_texto(v) for v in valor)
        if isinstance(valor, dict):
            return ", ".join(f"{k} {self._valor_como_texto(v)}" for k, v in valor.items())
        return str(valor)

    # ------------------------------------------------------------ limpeza

    # Sentinelas tecnicas usadas por eva/tools/builtin.py -- traduzidas
    # aqui, num unico lugar, para nunca chegarem cruas ao modelo. BUG REAL
    # JA VISTO: com o dict cru no contexto, o modelo abriu uma resposta
    # falada literalmente com "Sem_resultado" (o valor exato do campo
    # "aviso" de buscar()) antes de continuar raciocinando por conta
    # propria -- ele nao tinha como saber que aquilo era um codigo interno
    # e nao uma palavra da conversa. Fallback generico cobre codigo de
    # erro futuro que ainda nao esteja mapeado aqui.
    _NOTAS_ERRO_FERRAMENTA = {
        "sem_resultado": "a busca não encontrou nada útil",
        "divisao_por_zero": "não dá pra dividir por zero",
        "expressao_invalida": "essa conta não fechou, algo na expressão está errado",
        "cidade_nao_encontrada": "não achei essa cidade",
        "falha_rede": "não consegui conectar para checar isso agora",
        "searxng_indisponivel": "a busca está fora do ar agora",
        "falha_busca": "a busca falhou por algum motivo",
    }

    # Campos puramente técnicos -- servem só pra correlacionar requisição
    # e resposta do lado do código (ex: id de ação do bridge de Minecraft,
    # tipo de mensagem do protocolo), nunca pra decisão nem fala. Achado
    # real: "id" é um UUID cru, "type" é sempre o mesmo literal repetido
    # -- exatamente o tipo de coisa "com cara de dado técnico" que já
    # causou o modelo narrar/ecoar em vez de só usar (mesmo mecanismo
    # suspeito do número de estado cru, ver state.py).
    _CAMPOS_TECNICOS = {"id", "type"}

    def _limpar_ferramentas(self, resultados: dict) -> dict:
        """Remove metadados internos e traduz sentinelas de erro/aviso
        para nota curta em português -- ver _NOTAS_ERRO_FERRAMENTA acima.
        """
        limpo = {}
        for nome, r in resultados.items():
            if not isinstance(r, dict):
                limpo[nome] = r
                continue
            codigo = r.get("erro") or r.get("aviso")
            if codigo:
                limpo[nome] = {
                    "nota": self._NOTAS_ERRO_FERRAMENTA.get(
                        codigo, "essa ferramenta não conseguiu responder agora")
                }
            else:
                limpo[nome] = {
                    k: v for k, v in r.items()
                    if not k.startswith("_") and k not in self._CAMPOS_TECNICOS
                }
        return limpo

    def _limpar_historico(self, historico: list[dict]) -> list[dict]:
        """Mantém só role e content, na janela configurada -- e garante
        alternação estrita user/assistant/user/assistant/... começando em
        user. ChatML tolera papéis repetidos ou começar em assistant, mas
        Mistral/Nemo (Violet-Lotus e outros modelos de RP) travam com erro
        duro nesse caso.
        """
        janela = self.cfg.memoria.janela_historico
        bruto = [
            {"role": t["role"], "content": t["content"]}
            for t in historico[-janela:]
            if t.get("role") in ("user", "assistant") and t.get("content")
        ]

        limpo: list[dict] = []
        for msg in bruto:
            if limpo and limpo[-1]["role"] == msg["role"]:
                limpo[-1] = msg
            else:
                limpo.append(msg)

        while limpo and limpo[0]["role"] != "user":
            limpo.pop(0)

        return limpo