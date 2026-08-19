"""
Configuracao central do sistema EVA.

Tudo que e ajustavel fica aqui, para nao espalhar constante magica pelo
codigo. Valores podem vir do ambiente, o que permite trocar de modelo ou
de banco sem editar arquivo.

No Windows/PowerShell nao existe `export`: por isso o .env e carregado
AQUI, na importacao deste modulo, e nao no ponto de entrada.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Carregado na importacao do modulo, nao em __main__.py -- porque nem todo
# ponto de entrada passa por __main__.py. `python -m eva.integrations.discord`,
# `python -m eva.cli`, os testes em tests/, o simular_portao.py: nenhum
# desses roda __main__.py, e todos precisam do .env. Bug real que isso
# fecha: o DISCORD_TOKEN aparecia em "python -m eva --diagnostico" e
# sumia em "python -m eva.integrations.discord --diagnostico", porque só
# o primeiro passava pelo __main__.py que chamava load_dotenv(). Carregar
# aqui, no módulo que todo mundo importa antes de ler qualquer variável,
# fecha esse buraco de vez.
#
# override=False: se a variável já foi definida no ambiente real (ex.:
# $env:NOME="valor" no PowerShell, ou variável de sistema/CI), essa
# definição explícita vale mais que o que está no arquivo .env.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass  # python-dotenv não instalado: segue só com o ambiente real

RAIZ = Path(os.environ.get("EVA_HOME", Path.home() / ".eva"))


@dataclass
class LLMConfig:
    """Onde a EVA (o modelo conversacional) roda.

    O padrao aponta para o LM Studio local, que expoe uma API compativel
    com a da OpenAI. Trocar para outro provedor e so mudar base_url.
    """
    base_url: str = os.environ.get("EVA_LLM_URL", "http://localhost:1234/v1")
    api_key: str = os.environ.get("EVA_LLM_KEY", "lm-studio")
    modelo: str = os.environ.get("EVA_LLM_MODEL", "eva-3b")

    # 0.7 e nao 0.85: o modelo e um 3B com LoRA de personalidade. Temperatura
    # alta em modelo pequeno degrada coerencia mais rapido do que aumenta
    # variedade, e a variedade ja vem do fine-tuning.
    temperatura: float = float(os.environ.get("EVA_TEMP", "0.7"))
    top_p: float = 0.9

    # Teto de texto. O dataset tem mediana de 74 caracteres e p99 de 222,
    # entao 400 tokens ja e folga generosa -- serve para as respostas longas
    # (eva_longas_*.jsonl) sem permitir divagacao.
    max_tokens: int = 400

    # Em voz o teto e menor: 400 tokens viram uns 40 segundos de fala, tempo
    # demais para alguem esperando numa call. Combina com o formato
    # "MODO: VOZ. Seja concisa (max 2-3 frases)." do treino.
    # 200, não 120: no teste de call real, respostas sobre tópicos abertos
    # (ex. "fale sobre IA no mercado") pareciam cortar antes de terminar o
    # pensamento, e o modelo fechava com pergunta de volta como saída barata
    # quando sentia estar no limite. 200 tokens em fala ainda são ~20s --
    # longo pra voz, mas dá margem pra medir se o corte era a causa do
    # tique antes de mexer em dataset ou em MODO: VOZ. Se o tique continuar
    # igual com este valor, o teto não era a causa raiz.
    max_tokens_voz: int = int(os.environ.get("EVA_MAX_TOKENS_VOZ", "200"))

    timeout: int = 120

    # Como o bloco de dados e serializado no system prompt: "json" ou "prosa".
    #
    # O JSON so tinha respaldo empirico com o modelo fine-tunado abandonado
    # -- nao aparece em nenhum dos 1.135 exemplos de treino, funcionava por
    # capacidade herdada do Qwen2.5-Instruct base, nao pela LoRA. Sem esse
    # modelo, JSON deixa de ter qualquer vantagem sobre prosa e passa a ser
    # so risco (exige o modelo "parsear" estrutura em vez de so ler texto
    # dividido em linhas). Padrao trocado pra "prosa" por isso.
    formato_contexto: str = os.environ.get("EVA_FORMATO_CONTEXTO", "prosa")

    # Linha aditiva de humor/carisma (ver LINHA_CARISMA em context.py).
    # Ligada por padrão a pedido explícito -- desligue com EVA_CARISMA=0
    # se sentir que ela ficou insistindo em piada ou saindo do personagem.
    carisma: bool = os.environ.get("EVA_CARISMA", "1") == "1"

    # Penalidade de repetição -- sem isso, NENHUM parâmetro anti-repetição
    # ia pro LM Studio, e o comportamento ficava inteiramente dependente do
    # preset default do modelo carregado lá (pode estar fraco/desligado).
    # Achado real: sessão de teste longa saiu com respostas quase
    # idênticas pra mensagens de usuário diferentes -- sintoma clássico
    # disso faltar, não necessariamente falha do modelo em si.
    #
    # repeat_penalty é o parâmetro nativo do llama.cpp (>1.0 penaliza
    # token repetido, 1.0 = desligado). frequency_penalty/presence_penalty
    # são o par equivalente da API OpenAI -- backend diferente pode
    # respeitar um ou outro; mandar os dois não tem custo, o servidor
    # ignora o que não reconhece. 1.15/0.3/0.3 são pontos de partida
    # moderados -- teste e ajuste se sentir a fala engessada demais (alto
    # repeat_penalty pode fazer o modelo evitar até repetição natural,
    # tipo "sim, sim" ou uma palavra-chave que devia mesmo se repetir).
    repeat_penalty: float = float(os.environ.get("EVA_REPEAT_PENALTY", "1.15"))
    frequency_penalty: float = float(os.environ.get("EVA_FREQUENCY_PENALTY", "0.3"))
    presence_penalty: float = float(os.environ.get("EVA_PRESENCE_PENALTY", "0.3"))

    # Linha curta e sempre presente sobre capacidade real (roda local, ouve
    # e fala em call de Discord, tem memória) -- ver AUTOCONHECIMENTO em
    # context.py. Complementar ao RAG de história/capacidades, não
    # substituto: cobre o caso em que a pergunta é vaga demais pra busca
    # semântica achar algo específico. EVA_AUTOCONHECIMENTO=0 desliga se
    # quiser comparar antes/depois.
    autoconhecimento: bool = os.environ.get("EVA_AUTOCONHECIMENTO", "1") == "1"


@dataclass
class DecisionConfig:
    """Decision Engine.

    Na arquitetura ECA ele e um modelo pequeno e separado, treinado so para
    produzir JSON de planejamento. Comecamos com regras -- sao deterministicas,
    rapidas e nao custam nada. Quando o modelo proprio existir, e so apontar
    `modelo` para ele e ligar `usar_llm`.
    """
    usar_llm: bool = os.environ.get("EVA_DECISION_LLM", "0") == "1"
    base_url: str = os.environ.get("EVA_DECISION_URL", "http://localhost:1234/v1")
    api_key: str = os.environ.get("EVA_LLM_KEY", "lm-studio")
    modelo: str = os.environ.get("EVA_DECISION_MODEL", "eva")
    temperatura: float = 0.0  # decisao deve ser consistente, nao criativa
    # 200 nao bastava para modelo que "pensa" antes de responder (ex:
    # minicpm-v-4.6) -- confirmado em teste real: o modelo gastava os 200
    # tokens inteiros em reasoning_content e nunca chegava a escrever o
    # JSON, finish_reason "length" no meio do raciocinio. O decision.py
    # tambem tinha isso HARDCODED direto na chamada, ignorando esse campo
    # -- corrigido junto, agora os dois concordam.
    max_tokens: int = 700
    timeout: int = 60

    # Groq como backend PRINCIPAL de decisão (rápido, na nuvem, não disputa
    # VRAM com os outros modelos locais -- motivo real de ligar isso: três
    # modelos locais competindo por GPU tornava a troca entre eles mais
    # lenta). O local (campos acima) continua existindo e vira RESERVA
    # automática -- se o Groq falhar ou bater limite de uso, cai pro LM
    # Studio local sem intervenção manual, e só se os dois falharem cai
    # pra decisão por regra (fallback que já existia).
    groq_ativo: bool = os.environ.get("EVA_DECISION_GROQ", "0") == "1"
    groq_url: str = os.environ.get("EVA_DECISION_GROQ_URL", "https://api.groq.com/openai/v1")
    groq_modelo: str = os.environ.get("EVA_DECISION_GROQ_MODEL", "openai/gpt-oss-20b")
    # Reaproveita a mesma chave que já existe pro STT via Groq -- um só
    # lugar pra configurar a chave, não duas variáveis fazendo a mesma coisa.
    groq_key: str = os.environ.get("GROQ_API_KEY", "")


@dataclass
class MemoriaConfig:
    caminho_db: Path = field(default_factory=lambda: RAIZ / "memoria.db")

    # Quantos itens de cada tipo entram no contexto por vez. Numeros baixos
    # de proposito: contexto grande dilui o que importa e custa tokens.
    max_fatos: int = 6
    # Teto separado de max_fatos: história dela é conteúdo diferente de fato
    # sobre a pessoa, e não deveria competir pelo mesmo espaço/relevância.
    max_historia: int = int(os.environ.get("EVA_MAX_HISTORIA", "2"))
    # Mesmo raciocínio, mas pras "regras" gerais de comportamento dela mesma
    # (tipo procedural, usuário reservado -- ver seed_capacidades_eva.py e
    # o bloco de busca em orchestrator._buscar_memorias). Baixo de propósito:
    # é reforço pontual sobre algo relevante ao assunto, não é pra
    # substituir a PERSONA.
    max_regras_propria: int = int(os.environ.get("EVA_MAX_REGRAS", "2"))
    max_episodios: int = 4
    max_procedimentos: int = 3
    max_personalidade: int = 3

    # Abaixo disso, o resultado da busca e considerado irrelevante e
    # descartado -- e melhor nao passar contexto do que passar contexto
    # errado, que faz a EVA falar de coisa que nao vem ao caso.
    score_minimo: float = 0.05

    # Quantos turnos de conversa recente vao no historico
    janela_historico: int = 12

    # Extracao por LLM, alem das regras. Roda em TODO turno, em segundo
    # plano -- nao atrasa a resposta ao usuario, mas custa uma chamada a
    # mais ao LM Studio por turno. Aponta para a mesma config do decisor
    # por padrao (mesmo eva-3b), mas e uma secao separada de proposito:
    # se um dia isso pesar na GPU (visao + conversa + extracao competindo),
    # e so trocar o modelo aqui sem mexer em decisao.
    extrair_com_llm: bool = os.environ.get("EVA_MEMORIA_LLM", "1") == "1"
    extrator_base_url: str = os.environ.get(
        "EVA_MEMORIA_LLM_URL", os.environ.get("EVA_DECISION_URL", "http://localhost:1234/v1"))
    extrator_api_key: str = os.environ.get("EVA_LLM_KEY", "lm-studio")
    extrator_modelo: str = os.environ.get(
        "EVA_MEMORIA_LLM_MODEL", os.environ.get("EVA_DECISION_MODEL", "eva"))
    extrator_timeout: int = 30

    # Embeddings (busca hibrida BM25 + semantica, ver memory/embeddings.py
    # e memory/store.py). Aponta para o mesmo LM Studio, modelo
    # nomic-embed-text -- ele roda junto com eva-3b e minicpm-v-4.6 sem
    # problema, e o custo por chamada e pequeno (768 floats, nao geracao
    # de texto). Desligavel por completo se quiser rodar so com FTS5, do
    # jeito que era antes desta secao existir.
    usar_embeddings: bool = os.environ.get("EVA_EMBEDDINGS", "1") == "1"
    embeddings_base_url: str = os.environ.get(
        "EVA_EMBEDDINGS_URL", "http://localhost:1234/v1")
    embeddings_api_key: str = os.environ.get("EVA_LLM_KEY", "lm-studio")
    embeddings_modelo: str = os.environ.get(
        "EVA_EMBEDDINGS_MODEL", "text-embedding-nomic-embed-text-v1.5@q4_k_m")
    embeddings_timeout: int = 30

    # Consolidacao periodica (memory/consolidacao.py): memorias episodicas
    # antigas e similares viram um resumo semantico, reduzindo o que
    # acumula sem limite. Roda sob demanda a cada N turnos, nao em todo
    # turno -- e trabalho pesado o suficiente pra nao valer a pena
    # verificar toda hora.
    consolidar_com_llm: bool = os.environ.get("EVA_CONSOLIDAR", "1") == "1"
    consolidar_a_cada_turnos: int = int(
        os.environ.get("EVA_CONSOLIDAR_INTERVALO", "50"))
    consolidar_dias_minimo: float = 14.0

    # Auto-reflexão: a cada N turnos (por pessoa, mesma lógica de
    # consolidar_a_cada_turnos), pede pro extrator (mesmo cliente da
    # extração de fatos -- tipicamente o modelo pequeno/decisor) observar
    # o que a conversa recente revelou sobre a PRÓPRIA EVA, e grava como
    # história/lore dela (usuario reservado, tipo semantica, fonte
    # 'auto_reflexao', confiança baixa -- ver extrair_personalidade_propria
    # em memory/extractor.py e _autorrefletir em orchestrator.py). V1
    # deliberadamente simples: sem promoção de confiança por repetição,
    # sem poda automática de observação que não se confirma -- ver como se
    # comporta em uso real antes de complicar.
    autorreflexao_ativa: bool = os.environ.get("EVA_AUTORREFLEXAO", "1") == "1"
    autorreflexao_a_cada_turnos: int = int(
        os.environ.get("EVA_AUTORREFLEXAO_INTERVALO", "40"))


@dataclass
class EstadoConfig:
    caminho: Path = field(default_factory=lambda: RAIZ / "estado.json")
    # Velocidade com que o estado interno se move por interacao. Baixo de
    # proposito: estado deve mudar devagar, senao vira reacao imediata ao
    # ultimo turno em vez de humor.
    inercia: float = 0.92


@dataclass
class IdentidadeConfig:
    """Quem e quem, para a linha situacional do system prompt.

    As tres formas vem do dataset: criador (138 exemplos), conhecido (210)
    e nenhuma linha (784). Ver eva/identity.py.
    """
    # Depois de quantos turnos um desconhecido vira "alguem que voce conhece
    # bem". Conta turnos e nao dias porque quem trocou 30 mensagens e mais
    # conhecido que quem disse oi uma vez por mes durante um ano.
    turnos_para_conhecido: int = int(os.environ.get("EVA_TURNOS_CONHECIDO", "30"))


@dataclass
class ConscienciaConfig:
    """Quando a EVA fala sem ser chamada.

    Os dois primeiros numeros vieram da calibracao em producao do sistema
    antigo (MIN_SILENCE=40, SPEAK_COOLDOWN=30). Nao mexa neles sem rodar
    ferramentas/simular_portao.py antes -- e barato de testar e caro de
    descobrir errado numa call.
    """
    ativa: bool = os.environ.get("EVA_CONSCIENCIA", "1") == "1"
    silencio_minimo: float = float(os.environ.get("EVA_SILENCIO_MIN", "40"))
    cooldown_fala: float = float(os.environ.get("EVA_COOLDOWN_FALA", "30"))
    limiar_base: float = float(os.environ.get("EVA_LIMIAR_FALA", "0.55"))
    # Força do impulso "vazio" (silêncio comprido, sem fio nem evento
    # específico -- ver FORCA_PADRAO em consciousness.py). Padrão 0.35
    # inalterado -- essa análise NÃO decidiu que o valor está errado, só
    # tornou ele configurável: com limiar_base=0.55 e os coeficientes de
    # estado atuais, a matemática do portão faz "vazio" só passar quando
    # curiosidade E energia estão simultaneamente no máximo (1.0) e
    # estresse é zero -- condição de borda, não "às vezes". O próprio
    # comentário em consciousness.py já diz que isso é DE PROPÓSITO ("o
    # mais fraco, quase sempre barrado"). Se você decidir que quer ela
    # puxando assunto do nada com mais frequência em condição normal (não
    # só de borda), suba este valor -- mas teste em
    # ferramentas/simular_portao.py antes de rodar numa call de verdade,
    # mesma política dos dois campos acima.
    forca_vazio: float = float(os.environ.get("EVA_INICIATIVA_VAZIO_FORCA", "0.35"))
    intervalo_tick: float = 5.0
    max_fios: int = 8
    horas_para_fio_azedar: float = 48.0
    # Cooldown da pesquisa por curiosidade (ver bridge_client._tentar_curiosidade).
    # Separado do TTL do impulso "pesquisa" (180s) -- sem isso, cada vez que o
    # impulso expirasse sem ninguém ter ouvido, o tick seguinte tentaria
    # pesquisar de novo: rajada de chamadas ao SearXNG por hora de silêncio.
    cooldown_curiosidade: float = float(os.environ.get("EVA_COOLDOWN_CURIOSIDADE", "600"))


@dataclass
class DashboardConfig:
    """Painel local de controle e debug -- ver eva/dashboard.py.

    Roda um servidor HTTP simples (stdlib, sem dependência nova) na sua
    própria máquina. Liga por padrão porque é ferramenta de debug -- em
    uso de longo prazo sem necessidade de inspecionar nada, pode desligar
    com EVA_DASHBOARD=0.
    """
    ativa: bool = os.environ.get("EVA_DASHBOARD", "1") == "1"
    # 127.0.0.1, não 0.0.0.0: sem isso o painel (sem autenticação nenhuma)
    # ficaria acessível para qualquer outro dispositivo na sua rede local.
    # Mude só se souber exatamente o que está fazendo.
    host: str = os.environ.get("EVA_DASHBOARD_HOST", "127.0.0.1")
    porta: int = int(os.environ.get("EVA_DASHBOARD_PORT", "8790"))


@dataclass
class VisaoConfig:
    """Reação a tela -- captura local do PC, MiniCPM-V para análise.

    Desligada por padrão (ativa=False): captura de tela é sensível --
    sempre ligar sob escolha explícita, nunca como default silencioso.
    Liga com EVA_VISAO=1.
    """
    ativa: bool = os.environ.get("EVA_VISAO", "0") == "1"

    # Qual monitor (mss: 0 é "todos combinados", 1 é tipicamente o
    # principal) e em que largura reduzir antes de mandar pro modelo.
    # 672px corta os tokens de visão por 3-4x frente a 1280x720 -- é o
    # que tira a análise de ~5s para ~2s no MiniCPM-V.
    monitor: int = int(os.environ.get("EVA_VISAO_MONITOR", "1"))
    largura_captura: int = int(os.environ.get("EVA_VISAO_LARGURA", "672"))

    # Detector de diferença (captura.py): quantos desvios-padrão acima da
    # média recente para disparar MACRO, e um piso absoluto para não
    # disparar em ruído quando a linha de base ainda é ~0 (tela parada).
    limiar_desvios: float = 2.5
    limiar_minimo_absoluto: float = 3.0

    # MiniCPM-V via LM Studio -- servidor separado do conversacional
    # (mesmo LM Studio, endpoint igual, mas nunca a mesma requisição) para
    # não competir por latência com decisão/conversa.
    base_url: str = os.environ.get("EVA_VISAO_URL", "http://localhost:1234/v1")
    api_key: str = os.environ.get("EVA_LLM_KEY", "lm-studio")
    modelo: str = os.environ.get("EVA_VISAO_MODEL", "minicpm-v-4.6")
    timeout: int = 45

    # Rajada de quadros por análise -- "vídeo de pobre": vários quadros na
    # mesma mensagem em vez de um só, para o modelo entender sequência.
    # 5 quadros / 0.35s = ~1.6s de janela temporal, mesmo padrão do código
    # de referência do projeto (V4/unified_vision_system).
    rajada_quadros: int = int(os.environ.get("EVA_VISAO_RAJADA", "5"))
    rajada_intervalo: float = 0.35

    # Abaixo desta sobreposição de palavras entre a descrição nova e a
    # cena anterior, é considerada mudança real (vira cena+evento). Acima,
    # é a mesma cena com palavras diferentes -- nada muda. Mesmo valor do
    # código de referência (_detect_visual_change, threshold 0.4).
    limiar_mudanca_cena: float = 0.4

    # Cena mais velha que isso não entra mais no contexto -- evita
    # contexto_visual "fantasma" de quando a visão já foi desligada.
    cena_ttl: float = 300.0

    # Cena mudou há menos que isso: injeta no prompt MESMO sem a pessoa
    # mencionar a tela explicitamente -- uma mudança recém-capturada é
    # provavelmente ainda relevante pro que está sendo dito. Mais velha
    # que isso, só injeta com referência explícita (ver decision.py,
    # visao_relevante). Bem menor que cena_ttl de propósito: cena_ttl é
    # "ainda faz sentido existir", isto é "ainda é óbvio que é sobre
    # isso sem precisar perguntar".
    janela_relevancia_recente: float = 15.0

    # Intervalo entre ticks (captura + avaliação do detector, barato).
    # Não é o intervalo de ANÁLISE (isso só acontece quando o detector
    # aprova) -- é de quanto em quanto tempo checar se algo mudou.
    tick_intervalo: float = float(os.environ.get("EVA_VISAO_TICK", "2.0"))

    debug: bool = os.environ.get("EVA_DEBUG", "0") == "1"


@dataclass
class VozConfig:
    """Speech-to-text e text-to-speech."""
    # STT via Groq (Whisper). Modelos: whisper-large-v3-turbo (rapido) ou
    # whisper-large-v3 (mais preciso).
    stt_chave: str = os.environ.get("GROQ_API_KEY", "")
    stt_modelo: str = os.environ.get("EVA_STT_MODEL", "whisper-large-v3-turbo")
    stt_idioma: str = os.environ.get("EVA_STT_IDIOMA", "pt")

    # Backend de STT: "groq" (padrao, via API) ou "whisper_cpp" (local, via
    # SERVIDOR HTTP persistente do whisper.cpp -- ver WhisperCppServerSTT
    # em stt.py). Exige compilar com GGML_VULKAN=1: o caminho HIP/ROCm
    # crashou em runtime nesta maquina. Se pedido mas o servidor nao
    # responder, cai pra Groq com aviso em vez de travar (ver criar_stt).
    stt_backend: str = os.environ.get("EVA_STT_BACKEND", "groq")

    # URL do servidor whisper.cpp (examples/server do repositorio) -- pra
    # onde o cliente HTTP (WhisperCppServerSTT, ver stt.py) manda cada
    # transcrição. Também usada pra derivar host/porta na hora de SUBIR o
    # servidor (ver stt_whisper_cpp_server_exe abaixo e SupervisorWhisper
    # em integrations/discord.py) e pra checar se ele já está rodando
    # antes de tentar subir outro em cima (evitaria erro de porta em uso).
    stt_whisper_cpp_url: str = os.environ.get("EVA_STT_WHISPER_CPP_URL", "http://127.0.0.1:8090")

    # Caminho do BINÁRIO do servidor (whisper-server.exe -- gerado no
    # mesmo build do whisper-cli, dentro de build*/bin/Release ou
    # equivalente). Preencha isso pra discord.py subir o servidor sozinho
    # junto com o bridge.js. Deixe vazio se preferir subir na mão num
    # terminal separado (mesmo padrão que o LM Studio já usa hoje) --
    # nesse caso discord.py detecta que já tem algo respondendo em
    # stt_whisper_cpp_url e não tenta subir outro.
    stt_whisper_cpp_server_exe: str = os.environ.get("EVA_STT_WHISPER_CPP_SERVER_EXE", "")
    # Mesmo modelo que já era usado com o whisper-cli -- reaproveita o
    # nome de variável que você já tinha no .env.
    stt_whisper_cpp_modelo: str = os.environ.get("EVA_STT_WHISPER_CPP_MODELO", "")
    # Threads pro servidor (-t). Antes era passado por CHAMADA (cli, uma
    # vez por frase); agora é passado uma vez só, na inicialização do
    # servidor. Vazio = usa o padrão do binário (4).
    stt_whisper_cpp_threads: int | None = int(os.environ.get("EVA_STT_WHISPER_CPP_THREADS", "0")) or None

    # TTS: Pocket TTS, unico backend. Suporta portugues via o modelo
    # DESTILADO `portuguese` (pocket-tts >= 2.0.0) -- ver eva/voice/pocket.py
    # pro mapeamento completo e o motivo de ter trocado do preview
    # `portuguese_24l`. Piper e edge-tts foram removidos -- manter tres
    # caminhos de sintese significava tres timbres diferentes e tres formas
    # de quebrar, e a clonagem de voz do Pocket e o motivo de existir voz.
    tts_backend: str = os.environ.get("EVA_TTS_BACKEND", "pocket")
    tts_idioma: str = os.environ.get("EVA_TTS_IDIOMA", "pt")
    # WAV de referencia para clonar a voz (5 segundos bastam).
    tts_voz: str = os.environ.get("EVA_TTS_VOZ", "")
    # Quantizacao int8 do modelo de TTS (~30% mais rapido em CPU, medido
    # pelo Kyutai, principalmente nos modelos "_24l"). Desligada por
    # padrao -- ver aviso em eva/voice/pocket.py.
    tts_quantize: bool = os.environ.get("EVA_TTS_QUANTIZE", "0") == "1"

    # Quanto acumular ANTES de mandar o play_start no streaming de voz
    # frase-a-frase, em ms de audio (48kHz estereo 16 bits). Absorve a
    # diferenca entre velocidade de sintese e velocidade de reproducao
    # em tempo real -- sem isso, se a proxima frase demorar mais pra
    # gerar do que a frase atual leva pra tocar, a fila de audio no
    # bridge.js esvazia e ele preenche com silencio (efeito de fala
    # cortada). 400ms e ponto de partida empirico: cobre o first-chunk
    # latency tipico do Pocket TTS (~200ms) com folga, sem atrasar a
    # resposta de forma perceptivel. Ajuste e meça no seu hardware.
    tts_pre_buffer_ms: int = int(os.environ.get("EVA_TTS_PRE_BUFFER_MS", "400"))

    # Streaming de voz (frase a frase, play_start/play_chunk/play_end) vs.
    # bloqueante (espera a resposta inteira + sintese inteira, so entao
    # toca). SEGUNDA tentativa e SEGUNDA reversao: primeira vez, suspeita
    # era o modelo de TTS antigo (portuguese_24l) perto do limite de tempo
    # real em CPU; trocamos pro modelo destilado + forcamos CPU no Pocket
    # (POCKET_TTS_DEVICE=cpu, ver pocket.py -- havia GPU AMD/ROCm entrando
    # por auto-deteccao, kernel de atencao marcado experimental pelo
    # proprio PyTorch) e o audio CONTINUOU ruim. Ou seja: nao era so a
    # sintese. Suspeita agora migra pro mecanismo de PLAYBACK em si --
    # bridge.js monta o audio em tempo real com um timer manual de 20ms
    # (setInterval) puxando de um buffer que os pedacos preenchem
    # conforme chegam (startPlayStream/pushPlayChunk), bem mais frágil que
    # o caminho bloqueante: aquele manda o clipe inteiro pronto pro
    # @discordjs/voice de uma vez (createAudioResource com stream comum),
    # sem nenhum timer nosso tentando bater 20ms certinho por cima de um
    # event loop que também está fazendo I/O de rede. Padrao voltou a ser
    # bloqueante; codigo do streaming continua existindo (nao apaga
    # investimento de engenharia por uma reversao), so nao roda por
    # padrao. Ver _falar_stream em bridge_client.py para o historico
    # completo.
    voz_streaming: bool = os.environ.get("EVA_VOZ_STREAMING", "0") == "1"

    # Vocabulario passado ao Whisper para melhorar nomes proprios e termos
    # tecnicos que ele erraria sem contexto.
    stt_vocabulario: str = os.environ.get("EVA_STT_VOCAB", "EVA, Alex")


@dataclass
class DiscordConfig:
    token: str = os.environ.get("DISCORD_TOKEN", "")
    prefixo: str = os.environ.get("EVA_DISCORD_PREFIXO", "!eva ")
    # Se definido, a EVA responde a tudo nesse canal sem precisar de mencao.
    canal_dedicado: str = os.environ.get("EVA_DISCORD_CANAL", "")
    # Segundos de silencio para considerar que a pessoa terminou de falar.
    # AGORA REALMENTE USADO: o bridge.js le EVA_VOZ_SILENCIO direto do
    # ambiente (mesma variavel, ele converte pra ms) e passa pro
    # `duration` do EndBehaviorType.AfterSilence do @discordjs/voice --
    # antes esse valor so existia aqui e o bridge.js tinha 1000ms fixo no
    # codigo. Esse numero e um piso de latencia que se soma a QUALQUER
    # backend de STT (a captura so termina depois desse silencio). Baixei
    # o padrao pra 0.7s como ponto de partida; abaixo disso corre risco de
    # cortar fala com pausa natural no meio -- meça numa call real e
    # ajuste caso perceba a EVA interrompendo antes da pessoa terminar.
    silencio_para_responder: float = float(os.environ.get("EVA_VOZ_SILENCIO", "0.7"))


@dataclass
class EVAConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    decisao: DecisionConfig = field(default_factory=DecisionConfig)
    memoria: MemoriaConfig = field(default_factory=MemoriaConfig)
    estado: EstadoConfig = field(default_factory=EstadoConfig)
    identidade: IdentidadeConfig = field(default_factory=IdentidadeConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    consciencia: ConscienciaConfig = field(default_factory=ConscienciaConfig)
    visao: VisaoConfig = field(default_factory=VisaoConfig)
    voz: VozConfig = field(default_factory=VozConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)

    # Id do dono da instancia. Serve de escopo padrao quando ninguem passa
    # `usuario` -- caso da CLI. Integracoes multiusuario passam sempre.
    usuario: str = os.environ.get("EVA_USER", "alex")
    nome_criador: str = os.environ.get("EVA_NOME_CRIADOR", "Alex")

    debug: bool = os.environ.get("EVA_DEBUG", "0") == "1"

    def preparar_diretorios(self) -> None:
        # Aceita caminhos como str tambem: e facil atribuir uma string por
        # engano ao configurar, e o erro que aparecia ('str' nao tem
        # atributo 'parent') nao apontava para a causa.
        self.memoria.caminho_db = Path(self.memoria.caminho_db)
        self.estado.caminho = Path(self.estado.caminho)
        RAIZ.mkdir(parents=True, exist_ok=True)
        self.memoria.caminho_db.parent.mkdir(parents=True, exist_ok=True)
        self.estado.caminho.parent.mkdir(parents=True, exist_ok=True)


def carregar_config() -> EVAConfig:
    cfg = EVAConfig()
    cfg.preparar_diretorios()
    _avisar_env_divergente()
    return cfg


def _avisar_env_divergente() -> None:
    """Avisa se uma variável de ambiente do SISTEMA está sobrepondo o .env.

    Existe por causa de um bug real: GROQ_API_KEY ficou definida como
    variável de ambiente persistente do Windows (User ou Machine), de uma
    configuração antiga. Com override=False no load_dotenv() -- que é o
    certo, pois uma variável de ambiente explícita deve valer mais que um
    arquivo -- essa chave velha vencia o .env silenciosamente. O sintoma era
    confuso: "editei o .env mas continua pegando a chave errada", sem
    nenhum erro apontando a causa até o 401 da Groq. Este aviso mostra qual
    arquivo .env foi carregado e sinaliza quando o valor efetivo não bate
    com o que está escrito nele -- para as poucas variáveis onde isso
    normalmente indica engano, não intenção.
    """
    try:
        from dotenv import dotenv_values, find_dotenv
    except ImportError:
        return

    caminho = find_dotenv(usecwd=True)
    if not caminho:
        return

    valores_arquivo = dotenv_values(caminho)
    # Só as credenciais/seletores que costumam ficar "presas" por engano
    # em setup antigo. Não checa tudo: EVA_HOME ou EVA_USER divergirem do
    # .env é uso normal (override intencional via $env:), não bug a
    # sinalizar. EVA_TTS_BACKEND e CARTESIA_API_KEY entraram aqui pelo
    # mesmo motivo que abriu esta função pro GROQ_API_KEY: "editei o .env
    # e nada mudou" é exatamente o sintoma de uma variável de sistema
    # antiga vencendo em silêncio, e antes disso não havia checagem
    # nenhuma pra troca de backend de TTS.
    for chave in ("GROQ_API_KEY", "DISCORD_TOKEN", "EVA_TTS_BACKEND", "CARTESIA_API_KEY"):
        do_arquivo = valores_arquivo.get(chave)
        efetivo = os.environ.get(chave)
        if do_arquivo and efetivo and do_arquivo != efetivo:
            print(
                f"[config] aviso: {chave} no ambiente do sistema difere do "
                f"que está em {caminho}. A variável de ambiente está "
                f"vencendo -- se não foi intencional, rode no PowerShell:\n"
                f"    [System.Environment]::SetEnvironmentVariable(\"{chave}\", $null, \"User\")\n"
                f"e abra um terminal novo."
            )