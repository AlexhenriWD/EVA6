"""
Configuracao central do sistema EVA.

Tudo que e ajustavel fica aqui, para nao espalhar constante magica pelo
codigo. Valores podem vir do ambiente, o que permite trocar de modelo ou
de banco sem editar arquivo.

No Windows/PowerShell nao existe `export`: carregue um `.env` com
python-dotenv antes de importar este modulo, ou use `$env:NOME="valor"`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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
    max_tokens_voz: int = 120

    timeout: int = 120

    # Como o bloco de dados e serializado no system prompt: "json" ou "prosa".
    #
    # O JSON e o que o Context Builder sempre produziu, mas nao aparece em
    # nenhum dos 1.135 exemplos de treino -- funciona por capacidade herdada
    # do Qwen2.5-Instruct base, nao pela LoRA. A prosa fica mais perto do
    # "Contexto visual: ..." que o modelo viu. Existem os dois aqui para dar
    # para comparar sem editar codigo.
    formato_contexto: str = os.environ.get("EVA_FORMATO_CONTEXTO", "json")


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
    max_tokens: int = 200
    timeout: int = 60


@dataclass
class MemoriaConfig:
    caminho_db: Path = field(default_factory=lambda: RAIZ / "memoria.db")

    # Quantos itens de cada tipo entram no contexto por vez. Numeros baixos
    # de proposito: contexto grande dilui o que importa e custa tokens.
    max_fatos: int = 6
    max_episodios: int = 4
    max_procedimentos: int = 3
    max_personalidade: int = 3

    # Abaixo disso, o resultado da busca e considerado irrelevante e
    # descartado -- e melhor nao passar contexto do que passar contexto
    # errado, que faz a EVA falar de coisa que nao vem ao caso.
    score_minimo: float = 0.05

    # Quantos turnos de conversa recente vao no historico
    janela_historico: int = 12


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
    intervalo_tick: float = 5.0
    max_fios: int = 8
    horas_para_fio_azedar: float = 48.0


@dataclass
class VozConfig:
    """Speech-to-text e text-to-speech."""
    # STT via Groq (Whisper). Modelos: whisper-large-v3-turbo (rapido) ou
    # whisper-large-v3 (mais preciso).
    stt_chave: str = os.environ.get("GROQ_API_KEY", "")
    stt_modelo: str = os.environ.get("EVA_STT_MODEL", "whisper-large-v3-turbo")
    stt_idioma: str = os.environ.get("EVA_STT_IDIOMA", "pt")

    # TTS: Pocket TTS, unico backend. Suporta portugues via o modelo
    # `portuguese_24l`. Piper e edge-tts foram removidos -- manter tres
    # caminhos de sintese significava tres timbres diferentes e tres formas
    # de quebrar, e a clonagem de voz do Pocket e o motivo de existir voz.
    tts_backend: str = os.environ.get("EVA_TTS_BACKEND", "pocket")
    tts_idioma: str = os.environ.get("EVA_TTS_IDIOMA", "pt")
    # WAV de referencia para clonar a voz (5 segundos bastam).
    tts_voz: str = os.environ.get("EVA_TTS_VOZ", "")

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
    silencio_para_responder: float = float(os.environ.get("EVA_VOZ_SILENCIO", "1.0"))


@dataclass
class EVAConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    decisao: DecisionConfig = field(default_factory=DecisionConfig)
    memoria: MemoriaConfig = field(default_factory=MemoriaConfig)
    estado: EstadoConfig = field(default_factory=EstadoConfig)
    identidade: IdentidadeConfig = field(default_factory=IdentidadeConfig)
    consciencia: ConscienciaConfig = field(default_factory=ConscienciaConfig)
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
    return cfg