"""
Armazenamento das memorias da EVA.

SQLite com FTS5 (busca full-text nativa). A escolha e deliberada:

- Sem servidor, sem dependencia externa, arquivo unico que da pra copiar.
- FTS5 ja resolve busca por relevancia com BM25, que e forte o suficiente
  para o volume de uma conversa pessoal (milhares de itens, nao milhoes).
- Um banco vetorial seria melhor para busca semantica ("carro" achando
  "automovel"), mas exige modelo de embedding rodando junto, o que dobra
  o custo de inicializacao. A interface aqui esta preparada para isso:
  `Memoria.buscar` pode ser trocada sem afetar o resto do sistema.

As quatro camadas (do documento original do projeto):
    episodica    -- eventos: "Alex comecou um projeto novo"
    semantica    -- fatos: "Alex usa Arch Linux"
    procedural   -- como agir: "prefere respostas tecnicas"
    personalidade-- o que funciona com essa pessoa: piadas, ritmo, temas
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

TIPOS = ("episodica", "semantica", "procedural", "personalidade")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memorias (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tipo        TEXT NOT NULL,
    conteudo    TEXT NOT NULL,
    fonte       TEXT,
    confianca   REAL DEFAULT 0.8,
    criado_em   REAL NOT NULL,
    acessado_em REAL,
    acessos     INTEGER DEFAULT 0,
    ativo       INTEGER DEFAULT 1,
    meta        TEXT
);

CREATE INDEX IF NOT EXISTS idx_tipo  ON memorias(tipo, ativo);
CREATE INDEX IF NOT EXISTS idx_criado ON memorias(criado_em DESC);

-- Indice de busca. content='' faz do FTS um indice externo: o texto fica
-- so na tabela principal, evitando duplicacao.
CREATE VIRTUAL TABLE IF NOT EXISTS memorias_fts USING fts5(
    conteudo,
    content='memorias',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memorias BEGIN
    INSERT INTO memorias_fts(rowid, conteudo) VALUES (new.id, new.conteudo);
END;
CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memorias BEGIN
    INSERT INTO memorias_fts(memorias_fts, rowid, conteudo) VALUES('delete', old.id, old.conteudo);
END;
CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memorias BEGIN
    INSERT INTO memorias_fts(memorias_fts, rowid, conteudo) VALUES('delete', old.id, old.conteudo);
    INSERT INTO memorias_fts(rowid, conteudo) VALUES (new.id, new.conteudo);
END;

CREATE TABLE IF NOT EXISTS conversas (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    papel     TEXT NOT NULL,
    conteudo  TEXT NOT NULL,
    criado_em REAL NOT NULL,
    sessao    TEXT
);
CREATE INDEX IF NOT EXISTS idx_conv ON conversas(criado_em DESC);
"""

# Palavras curtas e comuns demais para servirem de busca -- se entrassem na
# consulta, casariam com quase tudo e o ranking perderia sentido.
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "de", "da", "do", "das", "dos", "em",
    "no", "na", "nos", "nas", "por", "para", "com", "sem", "que", "e", "ou",
    "se", "eu", "voce", "vc", "ele", "ela", "meu", "minha", "seu", "sua",
    "ao", "aos", "the", "is", "of", "to", "tem", "ter", "foi", "ser", "esta",
    "isso", "esse", "essa", "aquilo", "mais", "menos", "muito", "ja", "nao",
}


@dataclass
class Memoria:
    id: int | None
    tipo: str
    conteudo: str
    fonte: str | None = None
    confianca: float = 0.8
    criado_em: float = 0.0
    acessos: int = 0
    meta: dict | None = None
    score: float = 0.0

    def __str__(self) -> str:
        return self.conteudo


# Expansao de termos. Busca por palavra-chave nao encontra "Arch Linux"
# quando se pergunta "que sistema operacional eu uso" -- as frases nao
# compartilham nenhuma palavra. Um banco vetorial resolveria isso por
# similaridade semantica; aqui cobrimos os casos mais comuns de um
# assistente pessoal com um mapa explicito, que e barato e previsivel.
#
# Nao pretende ser exaustivo: e uma ponte ate a busca vetorial, e novos
# termos podem ser adicionados conforme aparecem no uso real.
SINONIMOS: dict[str, tuple[str, ...]] = {
    "sistema": ("linux", "windows", "macos", "ubuntu", "arch", "distro"),
    "operacional": ("linux", "windows", "macos", "so"),
    "computador": ("pc", "desktop", "notebook", "maquina"),
    "comer": ("comida", "almoco", "jantar", "vegetariano", "vegano", "dieta", "restaurante"),
    "comida": ("comer", "vegetariano", "vegano", "dieta", "almoco", "jantar"),
    "trabalho": ("emprego", "empresa", "carreira", "profissao", "chefe"),
    "emprego": ("trabalho", "empresa", "carreira", "vaga", "entrevista"),
    "animal": ("gato", "cachorro", "pet", "bicho"),
    "bicho": ("gato", "cachorro", "pet", "animal"),
    "familia": ("mae", "pai", "irma", "irmao", "filho", "filha", "esposa", "marido"),
    "casa": ("apartamento", "moradia", "mora", "endereco"),
    "estudo": ("faculdade", "curso", "mestrado", "doutorado", "escola"),
    "programar": ("codigo", "programacao", "dev", "python", "projeto"),
    "projeto": ("eva", "programar", "codigo", "trabalho"),
    "saude": ("medico", "doenca", "remedio", "terapia", "psicologo"),
    "musica": ("banda", "cantor", "album", "show"),
    "viagem": ("viajar", "ferias", "trip"),
}


def _expandir(palavras: list[str]) -> list[str]:
    """Acrescenta sinonimos conhecidos aos termos da consulta."""
    extras: list[str] = []
    for p in palavras:
        for s in SINONIMOS.get(p, ()):
            if s not in palavras and s not in extras:
                extras.append(s)
    return extras


def _preparar_consulta(texto: str, expandir: bool = True) -> str:
    """Transforma texto livre em consulta FTS5 segura.

    O FTS5 tem sintaxe propria (aspas, NEAR, *, operadores). Texto do
    usuario passado direto quebra a consulta ou, pior, muda o sentido dela.
    Extraimos so as palavras e montamos um OR explicito.
    """
    palavras = re.findall(r"\w+", texto.lower(), flags=re.UNICODE)
    uteis = [p for p in palavras if len(p) > 2 and p not in STOPWORDS]
    if not uteis:
        uteis = [p for p in palavras if len(p) > 2]
    if not uteis:
        return ""

    termos = uteis[:12]
    if expandir:
        # sinonimos entram depois dos termos originais, e o BM25 continua
        # pontuando mais alto quem casa com os termos literais
        termos = termos + _expandir(termos)[:8]

    return " OR ".join(f'"{p}"' for p in termos)


class BancoMemoria:
    def __init__(self, caminho: Path | str):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.caminho), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)
        self.con.commit()

    # ---------- escrita ----------

    def adicionar(
        self,
        tipo: str,
        conteudo: str,
        fonte: str | None = None,
        confianca: float = 0.8,
        meta: dict | None = None,
        evitar_duplicata: bool = True,
    ) -> int | None:
        """Grava uma memoria. Retorna o id, ou None se foi ignorada.

        `evitar_duplicata` compara o texto normalizado com o que ja existe do
        mesmo tipo. Sem isso, a mesma informacao entraria a cada conversa e
        acabaria dominando o contexto por repeticao, nao por relevancia.
        """
        if tipo not in TIPOS:
            raise ValueError(f"tipo invalido: {tipo}. Use um de {TIPOS}")
        conteudo = conteudo.strip()
        if not conteudo:
            return None

        if evitar_duplicata:
            existente = self._achar_similar(tipo, conteudo)
            if existente is not None:
                # reforca a confianca do que ja existe em vez de duplicar
                self.con.execute(
                    "UPDATE memorias SET confianca = MIN(1.0, confianca + 0.05), "
                    "acessado_em = ? WHERE id = ?",
                    (time.time(), existente),
                )
                self.con.commit()
                return existente

        cur = self.con.execute(
            "INSERT INTO memorias (tipo, conteudo, fonte, confianca, criado_em, meta) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tipo, conteudo, fonte, confianca, time.time(),
             json.dumps(meta, ensure_ascii=False) if meta else None),
        )
        self.con.commit()
        return cur.lastrowid

    def _achar_similar(self, tipo: str, conteudo: str) -> int | None:
        """Detecta duplicata por sobreposicao de palavras significativas."""
        alvo = set(re.findall(r"\w+", conteudo.lower())) - STOPWORDS
        if not alvo:
            return None
        for row in self.con.execute(
            "SELECT id, conteudo FROM memorias WHERE tipo=? AND ativo=1", (tipo,)
        ):
            outras = set(re.findall(r"\w+", row["conteudo"].lower())) - STOPWORDS
            if not outras:
                continue
            jaccard = len(alvo & outras) / len(alvo | outras)
            if jaccard >= 0.7:
                return row["id"]
        return None

    def esquecer(self, id_memoria: int) -> None:
        """Desativa (nao apaga) -- permite auditar o que foi removido."""
        self.con.execute("UPDATE memorias SET ativo=0 WHERE id=?", (id_memoria,))
        self.con.commit()

    def esquecer_por_texto(self, termo: str) -> int:
        """Desativa memorias que casem com um termo. Retorna quantas."""
        achadas = self.buscar(termo, limite=50, score_minimo=0.0)
        for m in achadas:
            if m.id:
                self.esquecer(m.id)
        return len(achadas)

    # ---------- leitura ----------

    def buscar(
        self,
        consulta: str,
        tipo: str | None = None,
        limite: int = 5,
        score_minimo: float = 0.0,
    ) -> list[Memoria]:
        """Busca por relevancia (BM25), com ajuste por confianca e recencia."""
        fts = _preparar_consulta(consulta)
        if not fts:
            return []

        sql = """
            SELECT m.*, bm25(memorias_fts) AS rank
            FROM memorias_fts
            JOIN memorias m ON m.id = memorias_fts.rowid
            WHERE memorias_fts MATCH ? AND m.ativo = 1
        """
        params: list = [fts]
        if tipo:
            sql += " AND m.tipo = ?"
            params.append(tipo)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limite * 3)  # pega extra para reordenar depois

        try:
            linhas = list(self.con.execute(sql, params))
        except sqlite3.OperationalError:
            # consulta malformada nao deve derrubar a conversa
            return []

        agora = time.time()
        resultados = []
        for r in linhas:
            # bm25 do SQLite e negativo, quanto menor melhor
            base = -float(r["rank"])
            # memoria recente vale um pouco mais; meia-vida de ~60 dias
            idade_dias = (agora - r["criado_em"]) / 86400
            recencia = 0.5 ** (idade_dias / 60)
            score = base * (0.6 + 0.4 * r["confianca"]) * (0.75 + 0.25 * recencia)

            resultados.append(Memoria(
                id=r["id"], tipo=r["tipo"], conteudo=r["conteudo"],
                fonte=r["fonte"], confianca=r["confianca"],
                criado_em=r["criado_em"], acessos=r["acessos"],
                meta=json.loads(r["meta"]) if r["meta"] else None,
                score=score,
            ))

        resultados.sort(key=lambda m: m.score, reverse=True)
        resultados = [m for m in resultados if m.score >= score_minimo][:limite]

        for m in resultados:
            self.con.execute(
                "UPDATE memorias SET acessos = acessos + 1, acessado_em = ? WHERE id = ?",
                (agora, m.id),
            )
        self.con.commit()
        return resultados

    def listar(self, tipo: str | None = None, limite: int = 50) -> list[Memoria]:
        sql = "SELECT * FROM memorias WHERE ativo=1"
        params: list = []
        if tipo:
            sql += " AND tipo=?"
            params.append(tipo)
        sql += " ORDER BY criado_em DESC LIMIT ?"
        params.append(limite)
        return [
            Memoria(
                id=r["id"], tipo=r["tipo"], conteudo=r["conteudo"], fonte=r["fonte"],
                confianca=r["confianca"], criado_em=r["criado_em"], acessos=r["acessos"],
                meta=json.loads(r["meta"]) if r["meta"] else None,
            )
            for r in self.con.execute(sql, params)
        ]

    def contar(self) -> dict[str, int]:
        return {
            r["tipo"]: r["n"]
            for r in self.con.execute(
                "SELECT tipo, COUNT(*) n FROM memorias WHERE ativo=1 GROUP BY tipo"
            )
        }

    # ---------- historico de conversa ----------

    def registrar_turno(self, papel: str, conteudo: str, sessao: str | None = None) -> None:
        self.con.execute(
            "INSERT INTO conversas (papel, conteudo, criado_em, sessao) VALUES (?,?,?,?)",
            (papel, conteudo, time.time(), sessao),
        )
        self.con.commit()

    def historico(self, limite: int = 12) -> list[dict]:
        linhas = list(self.con.execute(
            "SELECT papel, conteudo, criado_em FROM conversas ORDER BY id DESC LIMIT ?",
            (limite,),
        ))
        return [{"role": r["papel"], "content": r["conteudo"], "em": r["criado_em"]}
                for r in reversed(linhas)]

    def ultimo_turno_em(self) -> float | None:
        r = self.con.execute("SELECT MAX(criado_em) m FROM conversas").fetchone()
        return r["m"] if r and r["m"] else None

    def fechar(self) -> None:
        self.con.close()
