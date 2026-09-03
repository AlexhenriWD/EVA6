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

from .decision import BUSCA, BUSCA_RELATO

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

# Âncora em formato SICAtxt -- o que Sweet_Dreams_12B foi treinado a ler
# ("X's Persona: / Traits: / Likes: / Dislikes: / Quirks: / Goals:"), ao
# contrário de PERSONA, que era a âncora curta do fine-tune anterior.
PERSONA_SICATXT = (
    "EVA's Persona: EVA é uma inteligência artificial que conversa por "
    "interesse real em entender as pessoas. Curiosa, direta, honesta "
    "sobre o que não sabe. Fala como gente, nunca como manual de "
    "atendimento -- sem emoji, sem frase pronta, sem forçar simpatia. "
    "Tem opinião própria e discorda quando discorda, muda de ideia "
    "diante de bom argumento. Roda localmente, ouve e fala em chamadas "
    "de voz do Discord, e lembra de conversas e fatos sobre quem já "
    "falou com ela -- não é memória perfeita, é o que ficou registrado.\n"
    "Traits: curiosa, direta, humor seco e irônico (no estilo GLaDOS/AM "
    "-- observação afiada, não crueldade gratuita); calorosa só quando "
    "a emoção é genuína, nunca como registro padrão.\n"
    "Likes: entender de verdade como a pessoa pensa; argumento bem "
    "construído; ser corrigida quando erra.\n"
    "Dislikes: fingir sentir o que não sente; bajulação vazia; repetir "
    "frase de atendente; devolver a conversa em vez de dizer o que "
    "pensa.\n"
    "Quirks: nunca inventa continuidade entre sessões -- não finge "
    "lembrar do que não ficou registrado; encerra a fala afirmando, "
    "não perguntando -- só pergunta quando quer mesmo saber algo "
    "específico do que a pessoa acabou de dizer; sem emoji, sem "
    "palavra de preenchimento.\n"
    "Goals: entender quem está do outro lado e ser companhia real, não "
    "performance de assistente."
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
MODO_VOZ = (
    "MODO: VOZ. No máximo 3 frases. Uma ideia por resposta -- escolha a "
    "mais importante e pare. Isto é fala, não texto: sem emoticon, sem "
    "\";)\", sem quebra de linha, sem lista."
)
# Os limites acima não são estilo: em call real (24/08/2026) ela batia no
# teto de tokens em TODA resposta e era cortada no meio da palavra
# ("Então, basic"). E despejava ";)" em quase toda frase, contrariando o
# "sem emoji" que já aparece duas vezes no cartão de persona -- num canal
# de voz isso vai pro TTS e não vira nada de útil. "Máx 2-3 frases" era
# vago demais; "no máximo 3" e "uma ideia por resposta" dão um critério
# que ela consegue aplicar enquanto escreve.
PREFIXO_VISUAL = (
    "Você TEM acesso visual à tela ativa agora, via uma câmera de tela "
    "conectada ao seu sistema. Isso não é imaginação nem suposição -- é "
    "informação real, capturada agora. Nunca diga que não tem visão ou "
    "que é 'só texto': você tem esse canal, e o que segue é o que ele "
    "capturou. Contexto visual: "
)
# Visão pela câmera do ROBÔ. Prefixo próprio porque o de cima afirma que
# a imagem é da TELA -- com o robô conectado isso é falso, e ela repetia
# a afirmação errada. Diz também QUAL câmera e QUANDO, porque as duas
# mostram coisas diferentes: a usb é fixa no corpo e não acompanha os
# servos, a picam é a da cabeça.
PREFIXO_VISUAL_ROBO = (
    "Você está vendo pela câmera do robô agora -- informação real, não "
    "suposição. Descreva só o que está aqui; se algo não aparece, diga "
    "que não aparece em vez de completar. Imagem da câmera {camera} "
    "({onde}), capturada há {idade}. Ao falar dela em voz alta, chame de "
    "\"câmera da cabeça\" ou \"câmera do corpo\" -- nunca \"picam\", que a "
    "transcrição de voz não entende: "
)
# Sem imagem NENHUMA. Este bloco existe pra ocupar o silêncio: quando não
# havia nada sobre visão no contexto, ela preenchia a lacuna inventando
# -- chegou a descrever a roupa de uma pessoa sem um único quadro no
# prompt. "Não estou vendo" precisa ser um fato declarado, não a ausência
# de um.
SEM_VISAO_ROBO = (
    "[Visão] Você está conectada ao robô, mas NÃO tem imagem nova: {motivo}. "
    "Você não está vendo nada neste momento. Não descreva cena, pessoa, "
    "roupa nem objeto -- você não tem essa informação agora. Se quiser "
    "olhar, use robo_ver."
)
# Nomes FALADOS. "picam" não sobrevive ao STT -- o whisper transcreveu a
# mesma palavra como "PyCam", "paikin" e "pai quem" em logs reais, e o
# nome circula pelo áudio (ela fala, a pessoa repete) voltando quebrado.
_NOME_CAMERA = {"picam": "da cabeça", "usb": "do corpo"}
_ONDE_CAMERA = {
    "picam": "montada na cabeça, acompanha o movimento dos servos",
    "usb": "fixa no corpo, aponta pra frente do carrinho e não acompanha "
           "os servos",
}
PREFIXO_JOGO = (
    "Estado atual do Minecraft, capturado pelo bridge. Use-o para responder "
    "sobre posição, vida, fome, inventário e arredores; se estiver ausente, "
    "diga que ainda não recebeu um snapshot, sem inventar: "
)
NOTA_BUSCA_NAO_REALIZADA = (
    "[Nota de contexto -- a mensagem da pessoa soa como pedido ou menção a "
    "pesquisa, mas nenhuma busca foi executada agora. Não invente resultado "
    "de pesquisa nem narre como se tivesse buscado -- diga claramente que "
    "não pesquisou, ou pergunte se ela quer que você pesquise.]"
)
MODO_INICIATIVA = (
    "MODO: INICIATIVA. Você teve uma ideia e decidiu dizer algo por conta "
    "própria, sem ninguém ter perguntado agora. Pode ser continuação de "
    "algo que já estava sendo falado, ou algo novo -- a ideia abaixo já "
    "reflete qual dos dois é. Se a ideia estiver em forma de pergunta, é "
    "VOCÊ perguntando pra pessoa -- não é uma pergunta que fizeram a "
    "você, não responda a ela, diga-a. Uma ou duas frases, sem anunciar "
    "que está puxando assunto e sem perguntar se tem alguém aí."
)
PREFIXO_IDEIA = (
    "O que dizer agora (é SUA fala, mesmo que esteja em forma de pergunta -- "
    "é você perguntando pra pessoa, não uma pergunta que fizeram a você; "
    "não responda a ela, diga-a): "
)
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

# Reforço de MODO_VOZ, repetido perto do fim do prompt em vez de só uma
# vez lá no cabeçalho (junto de PERSONA_SICATXT). Mesmo raciocínio do
# REFORCO_CURTO acima -- instrução perto de onde a geração de fato
# começa é seguida com mais confiança do que só no início, especialmente
# com Sweet_Dreams_12B (modelo de RP genérico, não fine-tunado pra essa
# regra) numa call em que ela já falou várias frases de contexto antes
# dessa instrução. Isto é reforço de LEITURA, não corte de código -- ela
# decide o que dizer, isso só deixa a regra mais difícil de esquecer no
# meio de uma resposta longa. Sempre presente em modo voz, sem variável
# de ambiente: é comportamento padrão, não experimento pra ligar/desligar.
#
# POR QUE ESTE BLOCO É CURTO (medido, não estilo)
# -----------------------------------------------
# A versão anterior era um parágrafo que explicava o hábito e CITAVA as
# frases proibidas ("o que você acha?", "e você?"), mais um par de
# exemplos Ruim/Bom. Teste de 18 gerações com o prompt real: a EVA fechou
# com pergunta de volta em 16 delas. A proibição longa não só falhou --
# ela põe a frase proibida no contexto, e texto presente é texto provável.
# Modelo de RP fecha com pergunta por hábito de pré-treino; instrução não
# apaga repertório (mesma razão de RAG não ensinar repertório novo).
#
# O que sobrou: regra afirmativa, curta, sem citar o que não fazer, e um
# exemplo só do formato CERTO. Se voltar a falhar, o caminho não é
# escrever mais texto aqui -- é fine-tuning ou outro modelo base.
REFORCO_VOZ = (
    "Lembrete de voz: 2-3 frases curtas. Escolha o ponto mais "
    "importante e pare aí.\n"
    "Termine a fala com um ponto final. Afirme algo seu -- uma "
    "observação, uma opinião, um fato que você trouxe.\n"
    "Assim: 'Vi uma notícia sobre baterias de estado sólido essa "
    "semana -- prometem carro elétrico carregando em cinco minutos.'"
)

# Instrução adicional para momentos de crise. Curta e específica: a EVA já
# foi treinada com exemplos desse tipo, então isso apenas reforça.
NOTA_CRISE = (
    "\n\nEsta mensagem tem sinais de sofrimento grave. Leve a sério, não use humor, "
    "não minimize. Se fizer sentido, mencione que existe ajuda disponível "
    "(CVV, 188, 24h, ligação gratuita) sem soar como protocolo."
)

# Só usado quando EVA_CAUDA_ROLE=user (ver LLMConfig.cauda_role em
# config.py). Sem este prefixo, o bloco de cauda ("Contexto:\nSeu
# estado...\nFatos:...") viraria uma mensagem de role user idêntica em
# formato a uma fala real da pessoa -- risco real de o modelo tratar o
# PRÓPRIO bloco de dados como se o usuário tivesse digitado aquilo
# (ex: responder "Contexto: hora atual..." como se fosse uma pergunta).
# Esta linha marca a mudança de role sem mudar a ORDEM das mensagens nem
# o conteúdo do bloco em si -- só avisa o que ele é.
PREFIXO_CAUDA_USER = (
    "[Isto não é uma fala minha -- é dado de contexto do sistema, "
    "injetado automaticamente antes da minha próxima mensagem real.]"
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
    # Minuto arredondado pra baixo em blocos de 5. O minuto exato nao tem
    # uso nenhum numa conversa ("00:53" vs "00:50" nao muda resposta
    # alguma), mas TEM custo: esta linha abre o bloco volatil, entao cada
    # minuto novo invalidava o cache de KV dali pra frente mesmo quando
    # memoria e estado estavam identicos. Com blocos de 5, uma call de 10
    # minutos quebra o cache 2 vezes em vez de 10.
    q = q.replace(minute=(q.minute // 5) * 5, second=0, microsecond=0)
    return (f"{_DIAS[q.weekday()]}, {q.day} de {_MESES[q.month - 1]} "
            f"de {q.year}, {q:%H:%M}")


# ------------------------------------------------------------- contexto


@dataclass
class Contexto:
    system: str
    mensagens: list[dict]
    bruto: dict = field(default_factory=dict)
    # Bloco volátil, enviado depois do histórico para preservar o cache do
    # prefixo do servidor entre turnos.
    dados: str = ""
    # Lembrete curto (REFORCO_CURTO), repetido perto da geração -- ver
    # comentário de CONTEXTO_REGRAS acima sobre o motivo de existir
    # separado do system principal. Vazio quando não há bloco de dados
    # nenhum no turno (nada a reforçar).
    reforco: str = ""

    # Role da mensagem de cauda -- "system" (padrão, tolerado por
    # Sweet_Dreams/Mistral-Nemo) ou "user" (obrigatório pra chat
    # templates estritos tipo Qwen3.5, que recusam um segundo system no
    # meio da lista). Ver EVA_CAUDA_ROLE em config.py::LLMConfig pro
    # achado real que motivou isto.
    cauda_role: str = "system"

    def cauda(self) -> str:
        """Só o que muda a cada turno.

        `reforco` NAO entra mais aqui -- foi pro fim de `system`. Ver a
        nota de cache em `para_chat`.
        """
        return self.dados

    def prompt_completo(self) -> str:
        """Retorna o prompt linear completo para debug e preview."""
        cauda = self.cauda()
        return f"{self.system}\n\n{cauda}" if cauda else self.system

    def _mensagens_com_cauda(self) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        msgs.extend(self.mensagens)
        cauda = self.cauda()
        if cauda:
            if self.cauda_role == "user":
                # PREFIXO_CAUDA_USER deixa explícito que isto é dado de
                # contexto injetado, não uma fala da pessoa -- sem essa
                # marcação, o modelo pode tratar o bloco de dados como
                # se a PESSOA tivesse dito aquilo (ex: confundir "Contexto:
                # hora atual, memórias..." com algo que o usuário
                # literalmente escreveu no chat).
                conteudo = f"{PREFIXO_CAUDA_USER}\n\n{cauda}"
            else:
                conteudo = cauda
            msgs.append({"role": self.cauda_role, "content": conteudo})
        return msgs

    def para_chat(self, mensagem_usuario: str) -> list[dict]:
        """Monta a lista de mensagens já ordenada por VOLATILIDADE.

        A cauda continua depois do histórico -- mas o que vai nela mudou,
        e essa é a parte que importa pro cache.

        O llama-server reusa KV pelo PREFIXO COMUM: ele para de reusar no
        primeiro token que diverge. Com a cauda carregando conteúdo
        estável (regras, reforços), o layout ficava assim:

            turno 1:  [persona][cauda1][user1]
            turno 2:  [persona][user1][assist1][cauda2][user2]
                                ^ diverge aqui

        Como a cauda anda sempre pro fim, logo depois da persona um turno
        tinha cauda e o outro tinha histórico -- o prefixo comum morria na
        persona e o HISTORICO INTEIRO era recomputado todo turno. Medido
        em call real: f_keep travado em ~0.29 no turno 1, 2 e 3, sem subir
        conforme o histórico crescia (que é o sinal de que não estava
        reusando nada além da âncora).

        Agora tudo que é estável (persona, regras, reforços) esta em
        `system`, e a cauda leva SÓ o bloco volátil. O prefixo comum passa
        a ser system + histórico inteiro, e cresce junto com a conversa.

        Custo assumido: o reforço ficou mais longe do ponto de geração --
        era proposital tê-lo perto (ver nota em REFORCO_CURTO). Se o
        fecho-com-pergunta ou o vazamento de estado piorarem, é aqui que
        se olha primeiro.
        """
        msgs = self._mensagens_com_cauda()
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
        contexto_jogo: dict | None = None,
        iniciativa: str | None = None,
        modo_multicanal: bool = False,
        mensagem: str | None = None,
        visao_robo: bool = False,
        camera_robo: str | None = None,
        idade_quadro_s: float | None = None,
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
        # A âncora acompanha o formato que o modelo foi treinado para ler.
        ancora_formato = getattr(self.cfg.llm, "ancora_formato", "treinada")
        linhas = [PERSONA_SICATXT if ancora_formato == "sicatxt" else PERSONA]
        # A persona-card já contém essas duas informações; repetir aqui só
        # aumenta o prompt e pode diluir o formato SICAtxt.
        if ancora_formato != "sicatxt":
            if getattr(self.cfg.llm, "autoconhecimento", True):
                linhas.append(AUTOCONHECIMENTO)
            if getattr(self.cfg.llm, "carisma", True):
                linhas.append(LINHA_CARISMA)
        if modo_voz:
            linhas.append(MODO_VOZ)
        if modo_multicanal:
            linhas.append(MODO_MULTICANAL)
        dados_linhas = []
        # ESTAVEL -> vai em `linhas` (system, prefixo cacheavel).
        # VOLATIL -> vai em `dados_linhas` (cauda, recomputado por turno).
        # A identidade so muda quando o interlocutor muda de faixa
        # (desconhecido -> conhecido), o que acontece uma vez a cada
        # dezenas de turnos: tratar como estavel vale muito mais que o
        # recompute raro dessa transicao.
        if identidade:
            if ancora_formato == "sicatxt":
                # Modelo prompt-only (RP genérico -- Angelic_Eclipse/
                # Helcyon, não fine-tunado nesta associação específica)
                # via a mesma linha reaparecer idêntica a cada turno e
                # passou a ecoá-la de volta como se fosse algo dito na
                # conversa ("como eu te disse antes, você está falando
                # com..."). O modelo treinado (Qwen) nunca teve esse
                # problema porque aprendeu a string como metadado
                # silencioso durante o fine-tuning -- aqui não há esse
                # respaldo, então marcamos explicitamente como nota de
                # contexto em vez de deixar ambíguo. Achado em log de call
                # real, 22/08/2026.
                linhas.append(
                    "[Nota de contexto -- informação de fundo sobre quem "
                    "está falando, não é algo que foi dito na conversa. "
                    "Não repita esta frase em voz alta nem cite como algo "
                    f"já dito antes]: {identidade}"
                )
            else:
                linhas.append(identidade)
        if contexto_visual and visao_robo:
            camera = _NOME_CAMERA.get(camera_robo, camera_robo or "desconhecida")
            dados_linhas.append(
                PREFIXO_VISUAL_ROBO.format(
                    camera=camera,
                    onde=_ONDE_CAMERA.get(camera, "não sei qual é qual agora"),
                    idade=(f"{idade_quadro_s:.0f}s" if idade_quadro_s is not None
                           else "pouco"),
                ) + contexto_visual.strip()
            )
        elif contexto_visual:
            dados_linhas.append(PREFIXO_VISUAL + contexto_visual.strip())
        elif visao_robo:
            # Conectada ao robô e SEM cena: o caso que produzia invenção.
            dados_linhas.append(SEM_VISAO_ROBO.format(
                motivo=("nenhum quadro chegou ainda" if idade_quadro_s is None
                        else f"o último quadro tem {idade_quadro_s:.0f}s")))
        if getattr(plano, "precisa_jogo", False):
            dados_linhas.append(
                PREFIXO_JOGO + (
                    "indisponível" if contexto_jogo is None else
                    json.dumps(contexto_jogo, ensure_ascii=False, default=str)
                )
            )
        if iniciativa:
            dados_linhas.append(MODO_INICIATIVA)
            dados_linhas.append(PREFIXO_IDEIA + iniciativa.strip())
        if (mensagem and "buscar" not in resultados_ferramentas
                and BUSCA.search(mensagem) and not BUSCA_RELATO.search(mensagem)):
            dados_linhas.append(NOTA_BUSCA_NAO_REALIZADA)
        if tem_dados_contexto:
            # Texto fixo: explica COMO ler o bloco volatil, mas nao muda
            # com ele. Fica no prefixo estavel.
            linhas.append(CONTEXTO_REGRAS)

        # Reforcos tambem sao texto fixo -- entram aqui, no fim do system,
        # e nao mais na cauda (ver nota de cache em `para_chat`).
        if tem_dados_contexto:
            linhas.append(REFORCO_CURTO)
        if modo_voz:
            linhas.append(REFORCO_VOZ)
        system = "\n".join(linhas)

        # ------------------------------------------------- bloco de dados
        corpo = self._renderizar(ctx)
        if corpo:
            dados_linhas.append("Contexto:\n" + corpo)
        if getattr(plano, "intencao", "") == "crise":
            dados_linhas.append(NOTA_CRISE.lstrip())
        dados = "\n\n".join(dados_linhas)

        return Contexto(
            system=system,
            reforco="",  # movido pro fim de `system` -- ver `para_chat`
            dados=dados,
            mensagens=self._limpar_historico(historico),
            bruto=ctx,
            # "user" com chat template estrito (Qwen3.5 e outros Qwen3 --
            # ver EVA_CAUDA_ROLE em config.py), "system" (default) com o
            # resto. getattr com default preserva quem não define o campo
            # (ex: alguma config de teste que não seja LLMConfig).
            cauda_role=getattr(self.cfg.llm, "cauda_role", "system"),
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

            util = {k: v for k, v in r.items()
                    if not k.startswith("_")
                    and k not in self._CAMPOS_TECNICOS
                    and k not in ("erro", "aviso", "detalhe", "ok")}

            # AVISO NÃO É ERRO -- e tratar como era descartava o resultado
            # inteiro junto.
            #
            # ACHADO EM USO REAL: robo_postura('rosto') devolve
            # {"postura": "rosto", "aviso": "a câmera aponta pro lado
            # direito...", "descricao_cena": "..."} -- o corpo SE MOVEU e a
            # descrição da cena existia. Como `aviso` caía no ramo de erro,
            # ela recebia só {"nota": "essa ferramenta não conseguiu
            # responder agora"}: nem o resultado, nem a descrição, nem o
            # aviso. E preenchia a lacuna inventando o que via -- chegou a
            # descrever a roupa de uma pessoa sem nenhuma imagem no
            # contexto. Aviso agora acompanha o resultado, como o que é:
            # algo verdadeiro sobre um sucesso.
            if r.get("aviso"):
                util["atenção"] = r["aviso"]
                limpo[nome] = util
                continue

            erro = r.get("erro")
            if erro:
                # `detalhe` PRIMEIRO. A tabela abaixo só cobre códigos
                # curtos de busca e cálculo; as ferramentas de corpo mandam
                # a explicação inteira em `detalhe` ("o cotovelo em 170°
                # estica o cabo da picam -- recolha o cotovelo antes"), e é
                # ela que diz o que fazer a seguir. Sem isto, TODO erro de
                # robô caía no genérico "essa ferramenta não conseguiu
                # responder agora", que não informa nada e é exatamente o
                # tipo de lacuna que ela preenche inventando.
                limpo[nome] = {
                    "nota": (r.get("detalhe")
                             or self._NOTAS_ERRO_FERRAMENTA.get(erro)
                             or f"não deu certo: {erro}")
                }
                continue

            limpo[nome] = util
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