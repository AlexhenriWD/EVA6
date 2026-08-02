"""
Configuracao central do sistema EVA.

Tudo que e ajustavel fica aqui, para nao espalhar constante magica pelo
codigo. Valores podem vir do ambiente, o que permite trocar de modelo ou
de banco sem editar arquivo.
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
    modelo: str = os.environ.get("EVA_LLM_MODEL", "eva")
    temperatura: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 400
    timeout: int = 120


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
    modelo: str = os.environ.get("EVA_DECISION_MODEL", "eva")
    temperatura: float = 0.0  # decisao deve ser consistente, nao criativa
    max_tokens: int = 200


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
class VozConfig:
    """Speech-to-text e text-to-speech."""
    # STT via Groq (Whisper). Modelos: whisper-large-v3-turbo (rápido) ou
    # whisper-large-v3 (mais preciso).
    stt_chave: str = os.environ.get("GROQ_API_KEY", "")
    stt_modelo: str = os.environ.get("EVA_STT_MODEL", "whisper-large-v3-turbo")
    stt_idioma: str = os.environ.get("EVA_STT_IDIOMA", "pt")

    # TTS: "piper" (offline, PT-BR), "edge" (online, PT-BR), "pocket"
    # (offline, mas só EN/FR na data desta implementação).
    # Vazio = escolhe automaticamente o primeiro disponível que suporte o idioma.
    tts_backend: str = os.environ.get("EVA_TTS_BACKEND", "")
    tts_idioma: str = os.environ.get("EVA_TTS_IDIOMA", "pt")
    tts_voz: str = os.environ.get("EVA_TTS_VOZ", "")

    # Vocabulário passado ao Whisper para melhorar nomes próprios e termos
    # técnicos que ele erraria sem contexto.
    stt_vocabulario: str = os.environ.get("EVA_STT_VOCAB", "EVA, Alex")


@dataclass
class DiscordConfig:
    token: str = os.environ.get("DISCORD_TOKEN", "")
    prefixo: str = os.environ.get("EVA_DISCORD_PREFIXO", "!eva ")
    # Se definido, a EVA responde a tudo nesse canal sem precisar de menção.
    canal_dedicado: str = os.environ.get("EVA_DISCORD_CANAL", "")
    # Segundos de silêncio para considerar que a pessoa terminou de falar.
    silencio_para_responder: float = float(os.environ.get("EVA_VOZ_SILENCIO", "1.0"))


@dataclass
class EVAConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    decisao: DecisionConfig = field(default_factory=DecisionConfig)
    memoria: MemoriaConfig = field(default_factory=MemoriaConfig)
    estado: EstadoConfig = field(default_factory=EstadoConfig)
    voz: VozConfig = field(default_factory=VozConfig)
    discord: DiscordConfig = field(default_factory=DiscordConfig)

    usuario: str = os.environ.get("EVA_USER", "usuario")
    debug: bool = os.environ.get("EVA_DEBUG", "0") == "1"

    def preparar_diretorios(self) -> None:
        # Aceita caminhos como str também: é fácil atribuir uma string por
        # engano ao configurar, e o erro que aparecia ('str' não tem
        # atributo 'parent') não apontava para a causa.
        self.memoria.caminho_db = Path(self.memoria.caminho_db)
        self.estado.caminho = Path(self.estado.caminho)
        RAIZ.mkdir(parents=True, exist_ok=True)
        self.memoria.caminho_db.parent.mkdir(parents=True, exist_ok=True)
        self.estado.caminho.parent.mkdir(parents=True, exist_ok=True)


def carregar_config() -> EVAConfig:
    cfg = EVAConfig()
    cfg.preparar_diretorios()
    return cfg