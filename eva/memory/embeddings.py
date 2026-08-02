"""
Embeddings via nomic-embed-text, servido pelo LM Studio.

Por que embedding além do FTS5
-------------------------------
FTS5/BM25 (em store.py) casa PALAVRA. "que sistema operacional eu uso" não
acha "usa Arch Linux" -- as frases não compartilham termo nenhum. É por
isso que existe o mapa SINONIMOS manual em store.py: uma ponte cara de
manter e que nunca cobre tudo.

Embedding casa SIGNIFICADO. "sistema operacional" e "Arch Linux" ficam
próximos no espaço vetorial mesmo sem palavra em comum. É complementar ao
FTS5, não substituto -- embedding é ruim exatamente onde BM25 é bom: nome
próprio, termo raro, sigla ("Alex", "postgres", "LoRA") tendem a borrar em
embedding e casar exato em BM25. A busca híbrida em store.py soma os dois.

PREFIXO DE TAREFA -- o detalhe que quebra tudo sem avisar
-----------------------------------------------------------
O nomic-embed-text foi treinado para exigir um prefixo dizendo qual é a
tarefa: `search_document:` no texto que está sendo indexado, `search_query:`
no texto da busca. Sem isso, o modelo ainda responde (não dá erro), mas a
qualidade do embedding cai -- é o tipo de bug que não aparece em log
nenhum, só em busca ruim sem explicação. Ver:
https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

Este módulo é o ÚNICO lugar que decide o prefixo -- por isso `gerar_documento`
e `gerar_consulta` são funções separadas em vez de um único `gerar(texto)`
com parâmetro opcional. Separado assim, é impossível esquecer o prefixo por
engano: o nome da função já obriga a escolha.
"""

from __future__ import annotations

import hashlib
import json
import struct
import urllib.error
import urllib.request

DIMENSOES = 768  # nomic-embed-text-v1.5 -- confira se trocar de modelo


class ErroEmbedding(Exception):
    pass


class ClienteEmbeddings:
    def __init__(self, base_url: str, modelo: str, api_key: str = "lm-studio",
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.modelo = modelo
        self.api_key = api_key
        self.timeout = timeout
        # Cache em processo: mesma frase repetida na mesma sessão (comum em
        # fatos-núcleo reconsultados a cada turno) não gera nova chamada.
        # Não persiste entre reinícios de propósito -- é otimização de
        # sessão, não armazenamento; o vetor definitivo mora no SQLite.
        self._cache: dict[str, list[float]] = {}

    def _post(self, texto: str) -> list[float]:
        chave = hashlib.md5(texto.encode("utf-8")).hexdigest()
        if chave in self._cache:
            return self._cache[chave]

        payload = json.dumps({"model": self.modelo, "input": texto}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                dados = json.loads(r.read())
        except urllib.error.HTTPError as e:
            corpo = e.read().decode("utf-8", errors="replace")[:300]
            raise ErroEmbedding(f"HTTP {e.code}: {corpo}") from e
        except urllib.error.URLError as e:
            raise ErroEmbedding(
                f"não consegui falar com o servidor de embeddings em "
                f"{self.base_url}. O LM Studio está rodando com o modelo "
                f"'{self.modelo}' carregado? Detalhe: {e.reason}"
            ) from e

        try:
            vetor = dados["data"][0]["embedding"]
        except (KeyError, IndexError) as e:
            raise ErroEmbedding(f"resposta em formato inesperado: {str(dados)[:200]}") from e

        self._cache[chave] = vetor
        return vetor

    def gerar_documento(self, texto: str) -> list[float]:
        """Embedding de algo que vai ser GUARDADO e depois buscado.

        Use para: o conteúdo de uma memória ao salvar.
        """
        return self._post("search_document: " + texto)

    def gerar_consulta(self, texto: str) -> list[float]:
        """Embedding de uma BUSCA -- o texto que descreve o que se procura.

        Use para: a pergunta/mensagem do usuário na hora de buscar memórias.
        Prefixo diferente do documento é proposital: o nomic foi treinado
        assimetricamente, consulta e documento não usam o mesmo espaço de
        forma simétrica mesmo sendo o mesmo modelo.
        """
        return self._post("search_query: " + texto)

    def disponivel(self) -> bool:
        try:
            self._post("teste")
            return True
        except ErroEmbedding:
            return False


# --------------------------------------------------- serialização em BLOB


def vetor_para_blob(vetor: list[float]) -> bytes:
    """Serializa para BLOB compacto -- 768 floats de 32 bits = 3072 bytes.

    Não usa JSON/pickle: JSON de 768 floats é ~4x maior e mais lento de
    parsear; pickle amarra ao formato interno do Python sem ganho nenhum
    aqui. struct é o formato mais direto para "lista de float32 crua".
    """
    return struct.pack(f"<{len(vetor)}f", *vetor)


def blob_para_vetor(blob: bytes) -> list[float]:
    n = len(blob) // 4  # float32 = 4 bytes
    return list(struct.unpack(f"<{n}f", blob))


def cosseno(a: list[float], b: list[float]) -> float:
    """Similaridade de cosseno, sem depender de numpy.

    O store não tem numpy como dependência hoje, e trazer numpy só para
    isto (vetores de 768 posições, poucas centenas de comparações por
    busca) não compensa o custo de mais uma dependência pesada. Python
    puro aqui não é gargalo -- é iterar 768 números umas centenas de
    vezes, não uma operação sobre matriz grande.
    """
    produto = soma_a = soma_b = 0.0
    for x, y in zip(a, b):
        produto += x * y
        soma_a += x * x
        soma_b += y * y
    if soma_a == 0.0 or soma_b == 0.0:
        return 0.0
    return produto / ((soma_a ** 0.5) * (soma_b ** 0.5))
