"""
Cliente para o modelo conversacional.

Fala com qualquer servidor compatível com a API da OpenAI, que é o padrão
de fato: LM Studio, Ollama, llama.cpp server, vLLM. Usa urllib da stdlib
em vez de `requests` ou do SDK oficial -- uma dependência a menos, e o que
precisamos aqui é uma chamada HTTP simples.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class ErroLLM(Exception):
    pass


class ClienteLLM:
    def __init__(self, config):
        self.cfg = config

    def _post(self, caminho: str, payload: dict, timeout: int | None = None) -> dict:
        url = self.cfg.base_url.rstrip("/") + caminho
        dados = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=dados,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {getattr(self.cfg, 'api_key', 'x')}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.cfg.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            corpo = e.read().decode("utf-8", errors="replace")[:400]
            raise ErroLLM(f"HTTP {e.code}: {corpo}") from e
        except urllib.error.URLError as e:
            raise ErroLLM(
                f"não consegui falar com o modelo em {url}.\n"
                f"Verifique se o servidor está rodando (LM Studio > Local Server).\n"
                f"Detalhe: {e.reason}"
            ) from e

    def completar(
        self,
        mensagens: list[dict],
        temperatura: float | None = None,
        max_tokens: int | None = None,
        modelo: str | None = None,
    ) -> str:
        payload = {
            "model": modelo or self.cfg.modelo,
            "messages": mensagens,
            "temperature": self.cfg.temperatura if temperatura is None else temperatura,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "top_p": getattr(self.cfg, "top_p", 0.9),
            "stream": False,
        }
        dados = self._post("/chat/completions", payload)
        try:
            return dados["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise ErroLLM(f"resposta em formato inesperado: {str(dados)[:200]}") from e

    def completar_stream(self, mensagens: list[dict], temperatura: float | None = None,
                         max_tokens: int | None = None):
        """Gera a resposta token a token. Útil para a resposta aparecer
        conforme é produzida, em vez de tudo de uma vez no fim."""
        payload = {
            "model": self.cfg.modelo,
            "messages": mensagens,
            "temperature": self.cfg.temperatura if temperatura is None else temperatura,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "top_p": getattr(self.cfg, "top_p", 0.9),
            "stream": True,
        }
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {getattr(self.cfg, 'api_key', 'x')}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
                for linha in r:
                    linha = linha.decode("utf-8").strip()
                    if not linha.startswith("data: "):
                        continue
                    corpo = linha[6:]
                    if corpo == "[DONE]":
                        break
                    try:
                        d = json.loads(corpo)
                        delta = d["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except urllib.error.URLError as e:
            raise ErroLLM(f"falha no streaming: {e.reason}") from e

    def disponivel(self) -> bool:
        """Testa se o servidor responde. Usado no diagnóstico."""
        try:
            url = self.cfg.base_url.rstrip("/") + "/models"
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {getattr(self.cfg, 'api_key', 'x')}"}
            )
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    def modelos(self) -> list[str]:
        try:
            url = self.cfg.base_url.rstrip("/") + "/models"
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {getattr(self.cfg, 'api_key', 'x')}"}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                d = json.loads(r.read())
            return [m.get("id", "?") for m in d.get("data", [])]
        except Exception:
            return []
