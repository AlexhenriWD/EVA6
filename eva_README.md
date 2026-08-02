# EVA — Arquitetura Cognitiva

Sistema conversacional em que cada módulo tem uma responsabilidade única.
O modelo que conversa é a última peça do fluxo, e a única que escreve
português — todo o resto produz dado estruturado.

```
                        mensagem
                            |
                            v
                    Decision Engine          decide, nunca conversa
                            |
        +-------------------+-------------------+
        |                   |                   |
     Memória            Ferramentas          Estado
   (4 camadas)          (só JSON)      (energia, curiosidade...)
        |                   |                   |
        +-------------------+-------------------+
                            v
                    Context Builder
                            |
                            v
                       EVA (LLM)              escreve a resposta
                            |
                            v
              extrai memórias, atualiza estado
```

## Instalação

Só precisa de Python 3.10+. Nenhuma dependência externa — SQLite e HTTP
vêm da biblioteca padrão.

```bash
git clone <seu-repo> && cd EVA
python -m eva --diagnostico
```

Para conversar, é preciso um servidor de modelo rodando. O caminho mais
simples é o LM Studio:

1. Baixe o modelo da EVA (GGUF) no LM Studio
2. Aba **Developer** → **Start Server**
3. `python -m eva`

Se o modelo tiver outro nome ou porta:

```bash
export EVA_LLM_URL=http://localhost:1234/v1
export EVA_LLM_MODEL=nome-do-modelo
```

## Uso

```bash
python -m eva                        # conversa
python -m eva --debug                # mostra o que o sistema fez em cada turno
python -m eva --diagnostico          # checa conexão, memórias, estado
python -m eva -m "oi, tudo bem?"     # uma mensagem e sai
python -m eva --lembrar "uso Arch Linux"
python -m eva --memorias
```

Durante a conversa: `/debug`, `/memoria`, `/estado`, `/lembrar X`,
`/esquecer X`, `/limpar`, `/sair`.

## Os módulos

### Decision Engine (`eva/decision.py`)

Recebe a mensagem e devolve um plano: qual a intenção, se precisa de
memória, quais ferramentas chamar, qual a carga emocional. **Não escreve
uma linha de português.**

Vem com duas implementações. A padrão é por regras — determinística,
instantânea, de graça, e cobre bem os casos frequentes. A alternativa usa
um LLM (`EVA_DECISION_LLM=1`), que cobre mais casos ao custo de latência.
No plano original do projeto, este seria um modelo próprio de 100M-500M
treinado só para produzir JSON de planejamento.

Detecção de crise é sempre reavaliada por regra, mesmo com o decisor LLM
ligado: é grave demais para depender de o modelo ter classificado certo.

### Memória (`eva/memory/`)

Quatro camadas, inspiradas em psicologia cognitiva:

| camada | o que guarda | exemplo |
|---|---|---|
| semântica | fatos | "usa Arch Linux" |
| episódica | eventos | "começou o projeto EVA em julho" |
| procedural | como agir | "prefere respostas técnicas" |
| personalidade | o que funciona | "gosta de humor seco" |

SQLite com FTS5 (busca full-text com BM25). Sem servidor, arquivo único.
Um banco vetorial daria busca semântica melhor, mas exigiria um modelo de
embedding rodando junto — a interface está preparada para essa troca.

Duas soluções para limitações da busca por palavra-chave:

- **Sinônimos**: "que sistema operacional eu uso" não compartilha palavra
  nenhuma com "Arch Linux". Um mapa explícito cobre os casos comuns.
- **Fatos-núcleo**: os fatos mais acessados entram no contexto sempre, sem
  depender de busca. "Como instalo o postgres?" não casa com "usa Arch
  Linux", mas é exatamente aí que a distro muda a resposta inteira.

A extração de memórias é por regras (`extractor.py`), com filtro explícito
de hipóteses: "queria ter um gato" e "se eu usasse Arch" não viram fato.
Há extração por LLM opcional, marcada com confiança menor para ser
auditável — fato alucinado guardado é pior que fato nenhum, porque volta
como contexto em conversas futuras.

### Ferramentas (`eva/tools/`)

Regra: **ferramenta nunca retorna texto, só JSON.** Se ela devolvesse "A
previsão é 25 graus", a resposta final teria um pedaço escrito por outra
pessoa, com outro tom.

Em caso de falha devolvem `{"erro": "..."}` em vez de levantar exceção — a
EVA foi treinada para lidar com isso honestamente ("a busca deu timeout,
não vou chutar datas").

Embutidas: `hora_atual`, `calcular`, `clima` (Open-Meteo), `buscar`
(DuckDuckGo). A calculadora usa AST com operações permitidas em lista —
`eval()` em texto do usuário seria execução arbitrária de código.

Adicionar uma ferramenta:

```python
from eva.tools.registry import registro

@registro.adicionar("minha_ferramenta", "o que ela faz", {"param": "descrição"})
def minha_ferramenta(param: str) -> dict:
    return {"resultado": ...}
```

### Estado interno (`eva/state.py`)

Energia, curiosidade, confiança, estresse. Muda **devagar** — com inércia
alta, para se comportar como disposição acumulada e não como humor volátil
reagindo ao último turno.

A energia decai proporcionalmente até um piso, nunca zera: cansaço
permanente não é humor, é bug. Tempo parado entre conversas recupera.

O estado entra no contexto como dado estruturado. O modelo foi treinado
para deixar isso aparecer no comportamento, não para dizer "minha
curiosidade está em 0.91".

### Context Builder (`eva/context.py`)

Monta o pacote que a EVA recebe, no formato **exato** usado no
fine-tuning:

```
Você é EVA, uma inteligência artificial que conversa por interesse real...

Contexto:
{"fatos":["usa Arch Linux"],"ferramentas":{"clima":{"temperatura":25}}}
```

Se este formato divergir do treinado, a qualidade cai sem erro aparente.
Por isso ele vive num lugar só.

Princípio de curadoria: contexto pequeno e relevante vale mais que grande.
Item irrelevante não é neutro — faz a EVA falar de coisa que não vem ao
caso, o que soa pior do que ela não lembrar de nada.

## Configuração

Tudo por variável de ambiente:

| variável | padrão | o que faz |
|---|---|---|
| `EVA_HOME` | `~/.eva` | onde ficam banco e estado |
| `EVA_LLM_URL` | `http://localhost:1234/v1` | servidor do modelo |
| `EVA_LLM_MODEL` | `eva` | nome do modelo |
| `EVA_DECISION_LLM` | `0` | usar LLM no Decision Engine |
| `EVA_DEBUG` | `0` | log detalhado |

## Testes

```bash
python tests/test_eva.py
```

27 testes cobrindo memória, extração, decisão, ferramentas, estado,
contexto e o ciclo completo. Incluem os casos que já quebraram durante o
desenvolvimento: consulta maliciosa no FTS, código malicioso na
calculadora, estado corrompido, LLM fora do ar, energia zerando.

## O que ainda não existe

Do plano original, faltam: visão (modelo pronto, produzindo descrição
estruturada), Body Controller (comandos de alto nível para drivers) e
módulo de pesquisa dedicado. A arquitetura já os comporta — entram como
ferramentas ou como novas entradas para o Context Builder.
