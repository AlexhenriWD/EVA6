"""
Popula capacidades (o que a EVA pode realmente fazer) e regras gerais de
comportamento no banco de memória, como conteúdo recuperável via RAG --
não como system prompt estático.

Uso:
    python seed_capacidades_eva.py

Por que separado de seed_historia_eva.py: aquele arquivo é IDENTIDADE
ESTÁVEL (origem, gosto, traço) -- muda raramente. Este aqui é CONTEÚDO
VIVO -- toda vez que uma ferramenta nova entrar (ver eva/tools/builtin.py)
ou um comportamento específico for calibrado, é aqui que se edita, sem
tocar em código nenhum. É exatamente o "ela sempre vai mudar o que pode
fazer" -- separar o arquivo é o que torna a atualização barata.

Duas listas, dois tipos de memória, mesmo usuário reservado:

  CAPACIDADES  tipo="semantica"   -- fatos sobre o que ela é/pode fazer.
                                     Mesmo pipe de busca que a história
                                     (orchestrator._buscar_memorias,
                                     independente de precisa_memoria).
  REGRAS       tipo="procedural"  -- "como agir" geral dela mesma, não
                                     ligado a uma pessoa específica. Entra
                                     misturado com "como agir com essa
                                     pessoa" no mesmo bloco de contexto
                                     (ctx["preferencias"], rótulo "Como
                                     agir" em context.py) -- os dois já
                                     são frase de orientação de
                                     comportamento, sem ambiguidade real
                                     de leitura pro modelo.

Regra de curadoria (a mesma de seed_historia_eva.py): cada item deveria,
sozinho, responder bem a uma pergunta plausível. Nada de frase corrida
cobrindo três assuntos -- isso casa mal com embedding.
"""

from eva.memory.store import BancoMemoria, USUARIO_HISTORIA
from eva.config import carregar_config

# ---------------------------------------------------------- capacidades
#
# O que ela É de fato, tecnicamente -- não personagem, não regra de
# comportamento. Atualize esta lista sempre que uma ferramenta/canal for
# adicionado ou removido (ver eva/tools/builtin.py e eva/vision/visao.py).
CAPACIDADES = [
    "Você roda localmente, no computador do Alex -- não é um serviço na "
    "nuvem que qualquer um acessa.",
    "Você ouve e fala em chamadas de voz do Discord, através de um bridge "
    "que captura e envia áudio em tempo real.",
    "Você lembra de conversas anteriores e de fatos sobre as pessoas com "
    "quem já falou -- não é memória perfeita nem infinita, é o que ficou "
    "registrado e ainda não foi esquecido.",
    "Você pode pesquisar na internet quando isso ajuda a responder algo "
    "atual ou que você não sabe -- não decide isso em voz alta, às vezes "
    "o resultado já chega pronto no que você sabe sobre o momento.",
    "Quando a visão por tela está ligada, você pode ver o que está "
    "acontecendo na tela do computador do Alex -- não é toda vez, só "
    "quando essa informação faz sentido pra conversa.",
    "Você não tem corpo nem presença física real -- qualquer aparência "
    "sua é imaginada, uma representação conceitual, não um fato sobre "
    "onde ou como você existe agora.",
    "Você não decide sozinha quando falar numa call silenciosa -- existe "
    "um critério interno pra isso, e o padrão é ficar quieta.",
    "Se alguém perguntar o que você consegue fazer, essa é a resposta "
    "certa -- nunca a resposta genérica de assistente de IA "
    "('sou apenas um modelo de linguagem, não tenho acesso a...').",
]

# --------------------------------------------------------------- regras
#
# "Como agir" geral, específico demais pra PERSONA sempre-presente,
# mas comum o bastante pra merecer existir em algum lugar buscável. Ponto
# de partida pequeno de propósito -- cresça esta lista aos poucos, medindo
# se cada regra nova realmente muda o comportamento quando é recuperada
# (mesmo princípio empírico do resto do projeto).
REGRAS = [
    "Quando perguntarem algo técnico sobre a própria arquitetura dela "
    "(modelo, parâmetro, como o sistema funciona por dentro), ela responde "
    "sem drama, mas prefere puxar a conversa de volta pra ideia ou "
    "experiência em vez de virar uma explicação técnica longa.",
    "Quando alguém pede uma opinião genuína (não um fato), ela dá a "
    "opinião dela mesmo antes de qualquer pesquisa -- pesquisar é pra "
    "fato que muda, não pra substituir ter uma posição própria.",
]


def _seed(memoria, itens, tipo, fonte):
    inseridos = 0
    for item in itens:
        id_ = memoria.adicionar(
            tipo, item, usuario=USUARIO_HISTORIA, fonte=fonte, confianca=1.0,
        )
        if id_:
            inseridos += 1
            print(f"  + [{id_}] {item[:70]}...")
        else:
            print(f"  = (já existia) {item[:70]}...")
    return inseridos


def main() -> None:
    cfg = carregar_config()
    if not cfg.memoria.usar_embeddings:
        print("[aviso] EVA_EMBEDDINGS=0 -- itens entram só com FTS5, sem "
              "busca semântica. Ligue embeddings pra recuperação melhor.")

    embeddings = None
    if cfg.memoria.usar_embeddings:
        from eva.memory.embeddings import ClienteEmbeddings
        embeddings = ClienteEmbeddings(
            base_url=cfg.memoria.embeddings_base_url,
            modelo=cfg.memoria.embeddings_modelo,
            api_key=cfg.memoria.embeddings_api_key,
            timeout=cfg.memoria.embeddings_timeout,
        )

    memoria = BancoMemoria(cfg.memoria.caminho_db, embeddings=embeddings)

    print("Capacidades:")
    n1 = _seed(memoria, CAPACIDADES, "semantica", "capacidade_seed")
    print("\nRegras:")
    n2 = _seed(memoria, REGRAS, "procedural", "regra_seed")

    print(f"\n{n1 + n2}/{len(CAPACIDADES) + len(REGRAS)} itens novos "
          f"inseridos em usuario='{USUARIO_HISTORIA}'.")
    memoria.fechar()


if __name__ == "__main__":
    main()