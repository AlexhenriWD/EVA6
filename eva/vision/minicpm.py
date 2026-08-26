"""
Cliente do modelo de visão -- MiniCPM-V 4.6 via LM Studio.

Mesmo padrão de eva/llm.py (urllib da stdlib, sem aiohttp) para não
introduzir uma segunda forma de falar com o LM Studio no mesmo projeto.

RAJADA, NÃO IMAGEM ÚNICA
--------------------------
"Vídeo" pelo LM Studio/llama.cpp não existe de verdade -- o llama.cpp
suporta MiniCPM-V como VLM de imagem, não a parte omni/streaming temporal
do MiniCPM-o. O que dá pra fazer, e é o que este módulo faz, é mandar
VÁRIOS quadros na mesma mensagem (mesmo formato multi-imagem que Groq e
outros providers usam) -- o modelo lê como uma sequência curta, o que já
captura movimento/mudança entre quadros muito melhor que uma foto isolada.
É "vídeo de pobre", e funciona.

SEPARAÇÃO DE MODELO -- por que a visão não compete com decisão
------------------------------------------------------------------
`decision.py` roda por regra, não por LLM, e isso é deliberado: decisão
acontece em TODO turno e precisa ser instantânea. Visão é lenta (rajada +
inferência num modelo de alguns bilhões de parâmetros, ~2s mesmo depois de
reduzir resolução) e roda só quando o DetectorDiferenca aprova. Se as duas
tarefas dividissem o mesmo modelo no LM Studio, uma chamada de visão em
voo bloquearia decisão -- e o atraso apareceria bem na hora errada, no
meio de uma conversa. MiniCPM-V fica só para visão.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request


class ErroVisao(Exception):
    pass


class ClienteVisao:
    def __init__(self, base_url: str, modelo: str, api_key: str = "lm-studio",
                 timeout: int = 45, max_tokens: int = 200,
                 temperatura: float = 0.3):
        self.base_url = base_url.rstrip("/")
        self.modelo = modelo
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperatura = temperatura

    def analisar(self, quadros_jpeg: list[bytes], prompt: str) -> str:
        """Manda 1+ quadros JPEG numa única mensagem multimodal.

        Com mais de um quadro, o prompt deveria deixar claro que é uma
        sequência (ver PROMPT_CENA/PROMPT_EVENTO neste módulo) -- sem essa
        instrução o modelo às vezes descreve cada imagem separadamente em
        vez de entender como uma cena contínua.
        """
        if not quadros_jpeg:
            raise ErroVisao("nenhum quadro para analisar")

        conteudo = [{"type": "text", "text": prompt}]
        for jpeg in quadros_jpeg:
            b64 = base64.b64encode(jpeg).decode("ascii")
            conteudo.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        payload = {
            "model": self.modelo,
            "messages": [{"role": "user", "content": conteudo}],
            "temperature": self.temperatura,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        url = self.base_url + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                dados = json.loads(r.read())
        except urllib.error.HTTPError as e:
            corpo = e.read().decode("utf-8", errors="replace")[:400]
            raise ErroVisao(f"HTTP {e.code}: {corpo}") from e
        except urllib.error.URLError as e:
            raise ErroVisao(
                f"não consegui falar com o servidor de visão em "
                f"{self.base_url}. O LM Studio está rodando com o modelo "
                f"'{self.modelo}' carregado? Detalhe: {e.reason}"
            ) from e

        try:
            return dados["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise ErroVisao(f"resposta em formato inesperado: {str(dados)[:200]}") from e

    def disponivel(self) -> bool:
        try:
            req = urllib.request.Request(
                self.base_url + "/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                dados = json.loads(r.read())
            ids = {m.get("id") for m in dados.get("data", [])}
            return self.modelo in ids
        except Exception:
            return False


# Prompts curtos de propósito -- resposta longa custa tokens de saída e
# tempo, e "o que está acontecendo" cabe em uma ou duas frases na maioria
# dos casos. O formato de saída (prosa curta, não JSON de objetos/faces)
# segue o único exemplo do dataset que tem contexto visual:
# "Contexto visual: usuário jogando Hades (roguelike)." -- prosa, não
# estrutura. É o que o Context Builder espera colar depois do prefixo.

PROMPT_CENA = (
    "Estas imagens são quadros em sequência, com pequeno intervalo entre "
    "cada um, da tela de um computador. Descreva em UMA frase curta e "
    "direta o que está na tela agora -- que aplicativo, jogo ou documento, "
    "e o que está acontecendo em termos gerais. Não descreva quadro por "
    "quadro, descreva a cena como um todo. Máximo 20 palavras."
)

# Câmera do ROBÔ, não a tela -- pessoa física e espaço físico, não app/jogo.
# Um quadro só (não rajada): a cabeça acabou de se mover pra uma posição
# nova e parada, não faz sentido comparar contra o instante anterior à
# virada. Pede pessoas EXPLICITAMENTE porque é o dado que mais importa
# pra ela decidir como agir -- sem pedir, o modelo tende a descrever
# móveis/paredes e deixar gente de fora ou mencionar de passagem.
PROMPT_CENA_ROBO = (
    "Esta imagem é da câmera de um robô físico, agora. Descreva em UMA "
    "frase curta e direta o que tem à frente: se tem PESSOA(S) visível(is) "
    "diga quantas e o que estão fazendo; senão diga o que ocupa o espaço "
    "(parede, móvel, corredor, porta, obstáculo) e se o caminho à frente "
    "parece livre ou bloqueado. Máximo 25 palavras."
)