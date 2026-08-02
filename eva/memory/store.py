"""
Armazenamento das memorias da EVA.

SQLite com FTS5 (busca full-text nativa). A escolha e deliberada:

- Sem servidor, sem dependencia externa, arquivo unico que da pra copiar.
- FTS5 ja resolve busca por relevancia com BM25, que e forte o suficiente
  para o volume de uma conversa pessoal (milhares de itens, nao milhoes).
- Um banco vetorial seria melhor para busca semantica ("carro" achando
  "automovel"), mas exige modelo de embedding rodando junto, o que dobra
  o custo de inicializacao. A interface aqui esta preparada para isso:
  `buscar` pode ser trocada sem afetar o resto do sistema.

As quatro camadas (do documento original do projeto):
    episodica    -- eventos: "Alex comecou um projeto novo"
    semantica    -- fatos: "Alex usa Arch Linux"
    procedural   -- como agir: "prefere respostas tecnicas"
    personalidade-- o que funciona com essa pessoa: piadas, ritmo, temas

MULTIUSUARIO
------------
Toda memoria e todo turno pertencem a um `usuario`. Isso nao e enfeite: a
EVA vive num servidor de Discord com varias pessoas na mesma call, e sem
escopo ela citaria para uma pessoa um fato que outra contou. Por isso
`usuario` e obrigatorio e nomeado -- passar por engano na posicao errada
levanta TypeError em vez de vazar memoria em silencio.

O estado interno NAO e escopado: energia, curiosidade e estresse sao dela,
nao da conversa. Quem cuida disso e o GerenciadorEstado.

CONCORRENCIA
------------
A conexao usa check_same_thread=False porque as integracoes chamam de
threads diferentes (asyncio.to_thread). SQLite aguenta isso, mas cursores
compartilhados nao: dois turnos simultaneos no mesmo cursor dao
"recursive use of cursors" ou leitura parcial. O RLock resolve.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

TIPOS = ("episodica", "semantica", "procedural", "personalidade")

# O schema roda em duas etapas, e a ordem importa: um banco criado antes do
# multiusuario nao tem a coluna `usuario`, e um indice sobre ela falharia
# com "no such column". Entao: tabelas -> migracao -> indices e gatilhos.
SCHEMA_TABELAS = """
CREATE TABLE IF NOT EXISTS memorias (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario     TEXT NOT NULL DEFAULT 'default',
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

CREATE TABLE IF NOT EXISTS conversas (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario   TEXT NOT NULL DEFAULT 'default',
    papel     TEXT NOT NULL,
    conteudo  TEXT NOT NULL,
    criado_em REAL NOT NULL,
    sessao    TEXT
);

-- Quem e quem. Alimenta a linha situacional do system prompt.
CREATE TABLE IF NOT EXISTS pessoas (
    usuario   TEXT PRIMARY KEY,
    nome      TEXT,
    relacao   TEXT NOT NULL DEFAULT 'desconhecido',
    turnos    INTEGER DEFAULT 0,
    criado_em REAL NOT NULL,
    visto_em  REAL
);
"""

SCHEMA_INDICES = """
CREATE INDEX IF NOT EXISTS idx_tipo   ON memorias(usuario, tipo, ativo);
CREATE INDEX IF NOT EXISTS idx_criado ON memorias(criado_em DESC);
CREATE INDEX IF NOT EXISTS idx_conv   ON conversas(usuario, id DESC);

-- Indice de busca. content='memorias' faz do FTS um indice externo: o texto
-- fica so na tabela principal, evitando duplicacao. O filtro por usuario
-- vive no JOIN com `memorias`, entao o indice nao precisa saber de usuario.
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
    usuario: str = "default"
    fonte: str | None = None
    confianca: float = 0.8
    criado_em: float = 0.0
    acessos: int = 0
    meta: dict | None = None
    score: float = 0.0
    # Quando esta memoria foi absorvida por um resumo semantico (ver
    # consolidacao.py), ou None se nunca consolidada. Nao afeta busca --
    # a memoria original continua pesquisavel mesmo apos consolidar; o
    # resumo e um ADICIONAL, nao uma substituicao que apaga o detalhe.
    consolidado_em: float | None = None

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
    def __init__(self, caminho: Path | str, embeddings=None):
        """`embeddings` é opcional -- um ClienteEmbeddings (ver embeddings.py).

        Sem ele, o banco funciona exatamente como antes: só FTS5/BM25. Com
        ele, `adicionar` grava o vetor junto e `buscar` passa a combinar
        BM25 com similaridade de cosseno. Opcional de propósito: um banco
        criado sem embeddings continua abrindo e funcionando se um dia o
        LM Studio estiver fora do ar -- só perde a metade semântica da
        busca, não trava.
        """
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.embeddings = embeddings
        self.con = sqlite3.connect(str(self.caminho), check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        # WAL deixa leitura e escrita concorrerem sem bloquear uma a outra.
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.executescript(SCHEMA_TABELAS)
        self._migrar()
        fts_nova = not self._existe("memorias_fts")
        self.con.executescript(SCHEMA_INDICES)
        if fts_nova:
            # Banco que ja tinha memorias mas nao tinha indice de busca: os
            # gatilhos so pegam o que entrar daqui pra frente, entao o que ja
            # estava la ficaria invisivel para `buscar`.
            self.con.execute(
                "INSERT INTO memorias_fts(memorias_fts) VALUES('rebuild')")
        self.con.commit()

    def _existe(self, nome: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM sqlite_master WHERE name=?", (nome,)).fetchone() is not None

    def _migrar(self) -> None:
        """Acrescenta `usuario` em bancos criados antes do multiusuario.

        Bancos antigos tem os dados de uma pessoa so, entao 'default' e o
        rotulo certo -- e o mesmo padrao do schema novo. Quem quiser
        renomear depois usa `renomear_usuario`.
        """
        for tabela in ("memorias", "conversas"):
            colunas = {r["name"] for r in
                       self.con.execute(f"PRAGMA table_info({tabela})")}
            if not colunas:
                continue
            if "usuario" not in colunas:
                self.con.execute(
                    f"ALTER TABLE {tabela} ADD COLUMN usuario TEXT NOT NULL "
                    "DEFAULT 'default'"
                )

        # embedding: NULL em linhas antigas ate serem regeradas (nao ha
        # como recalcular em massa sem custo de rede -- fica pra
        # consolidacao.py preencher sob demanda, nao aqui na migracao).
        # consolidado_em: quando a memoria foi absorvida por um resumo
        # semantico; NULL enquanto nunca consolidada. Ver consolidacao.py.
        colunas_mem = {r["name"] for r in
                       self.con.execute("PRAGMA table_info(memorias)")}
        if colunas_mem and "embedding" not in colunas_mem:
            self.con.execute("ALTER TABLE memorias ADD COLUMN embedding BLOB")
        if colunas_mem and "consolidado_em" not in colunas_mem:
            self.con.execute(
                "ALTER TABLE memorias ADD COLUMN consolidado_em REAL")

    def renomear_usuario(self, de: str, para: str) -> int:
        """Move todas as memorias e turnos de um id para outro."""
        with self._lock:
            n = 0
            for tabela in ("memorias", "conversas"):
                cur = self.con.execute(
                    f"UPDATE {tabela} SET usuario=? WHERE usuario=?", (para, de))
                n += cur.rowcount
            self.con.execute("UPDATE pessoas SET usuario=? WHERE usuario=?",
                             (para, de))
            self.con.commit()
            return n

    # ---------- escrita ----------

    def adicionar(
        self,
        tipo: str,
        conteudo: str,
        *,
        usuario: str,
        fonte: str | None = None,
        confianca: float = 0.8,
        meta: dict | None = None,
        evitar_duplicata: bool = True,
        gerar_embedding: bool = True,
    ) -> int | None:
        """Grava uma memoria. Retorna o id, ou None se foi ignorada.

        `evitar_duplicata` compara o texto normalizado com o que ja existe do
        mesmo tipo e do mesmo usuario. Sem isso, a mesma informacao entraria
        a cada conversa e acabaria dominando o contexto por repeticao, nao
        por relevancia.

        `gerar_embedding` chama o servidor de embeddings (rede) SE
        `self.embeddings` estiver configurado. BEST-EFFORT: falha aqui
        (LM Studio fora do ar, timeout) nunca impede a memoria de ser
        salva -- ela so fica sem embedding, e busca cai pra FTS5 puro
        naquele item ate uma consolidacao futura preencher. `adicionar` e
        chamado no caminho de resposta ao usuario (via _pos_processar); se
        a rede de embeddings travar, a conversa nao pode travar junto.
        Quem quer desligar por completo (ex: importacao em massa, onde
        centenas de chamadas de rede sequenciais seriam inviaveis) passa
        gerar_embedding=False e roda em lote depois.
        """
        if tipo not in TIPOS:
            raise ValueError(f"tipo invalido: {tipo}. Use um de {TIPOS}")
        conteudo = conteudo.strip()
        if not conteudo:
            return None

        embedding_blob = None
        if gerar_embedding and self.embeddings is not None:
            try:
                from .embeddings import vetor_para_blob
                embedding_blob = vetor_para_blob(
                    self.embeddings.gerar_documento(conteudo))
            except Exception:
                embedding_blob = None  # best-effort -- ver docstring acima

        with self._lock:
            if evitar_duplicata:
                existente = self._achar_similar(tipo, conteudo, usuario)
                if existente is not None:
                    # reforca a confianca do que ja existe em vez de duplicar
                    self.con.execute(
                        "UPDATE memorias SET confianca = MIN(1.0, confianca + 0.05), "
                        "acessado_em = ? WHERE id = ?",
                        (time.time(), existente),
                    )
                    # Se essa memoria existente ainda nao tinha embedding
                    # (foi salva antes do servidor estar disponivel, por
                    # exemplo) e agora conseguimos gerar um, preenche --
                    # e como a consolidacao vai encontrar o vetor sem
                    # esperar uma rodada de preenchimento em lote separada.
                    if embedding_blob is not None:
                        self.con.execute(
                            "UPDATE memorias SET embedding=? "
                            "WHERE id=? AND embedding IS NULL",
                            (embedding_blob, existente),
                        )
                    self.con.commit()
                    return existente

            cur = self.con.execute(
                "INSERT INTO memorias (usuario, tipo, conteudo, fonte, confianca, "
                "criado_em, meta, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (usuario, tipo, conteudo, fonte, confianca, time.time(),
                 json.dumps(meta, ensure_ascii=False) if meta else None,
                 embedding_blob),
            )
            self.con.commit()
            return cur.lastrowid

    def _achar_similar(self, tipo: str, conteudo: str, usuario: str) -> int | None:
        """Detecta duplicata por sobreposicao de palavras significativas."""
        alvo = set(re.findall(r"\w+", conteudo.lower())) - STOPWORDS
        if not alvo:
            return None
        for row in self.con.execute(
            "SELECT id, conteudo FROM memorias "
            "WHERE tipo=? AND usuario=? AND ativo=1", (tipo, usuario)
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
        with self._lock:
            self.con.execute("UPDATE memorias SET ativo=0 WHERE id=?", (id_memoria,))
            self.con.commit()

    def esquecer_por_texto(self, termo: str, *, usuario: str) -> int:
        """Desativa memorias que casem com um termo. Retorna quantas."""
        achadas = self.buscar(termo, usuario=usuario, limite=50, score_minimo=0.0)
        for m in achadas:
            if m.id:
                self.esquecer(m.id)
        return len(achadas)

    # ---------- leitura ----------

    def buscar(
        self,
        consulta: str,
        *,
        usuario: str,
        tipo: str | None = None,
        limite: int = 5,
        score_minimo: float = 0.0,
    ) -> list[Memoria]:
        """Busca por relevancia. BM25 sempre; embedding complementa quando
        `self.embeddings` esta configurado.

        HIBRIDA, NAO SUBSTITUTA: BM25 continua rodando do jeito que sempre
        rodou -- nome proprio, termo raro, sigla ("Alex", "postgres")
        casam melhor em BM25 que em embedding, que tende a borrar isso.
        O embedding entra pra cobrir o caso que BM25 estruturalmente nao
        resolve: pergunta e fato sem NENHUMA palavra em comum ("que
        sistema operacional eu uso" vs "usa Arch Linux"). Os dois
        conjuntos de resultado sao unidos por id (uma memoria pode vir
        dos dois) e reordenados pelo melhor score entre os dois metodos.

        Sem `self.embeddings` configurado, ou se a chamada de rede falhar,
        cai para BM25 puro -- exatamente o comportamento de antes desta
        mudanca. Best-effort, igual a escrita: busca nunca trava esperando
        embedding.
        """
        agora = time.time()
        por_id: dict[int, Memoria] = {}

        # ---------------------------------------------------------- BM25
        fts = _preparar_consulta(consulta)
        if fts:
            sql = """
                SELECT m.*, bm25(memorias_fts) AS rank
                FROM memorias_fts
                JOIN memorias m ON m.id = memorias_fts.rowid
                WHERE memorias_fts MATCH ? AND m.ativo = 1 AND m.usuario = ?
            """
            params: list = [fts, usuario]
            if tipo:
                sql += " AND m.tipo = ?"
                params.append(tipo)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limite * 3)

            with self._lock:
                try:
                    linhas = list(self.con.execute(sql, params))
                except sqlite3.OperationalError:
                    # consulta malformada nao deve derrubar a conversa
                    linhas = []

            for r in linhas:
                # bm25 do SQLite e negativo, quanto menor melhor
                base = -float(r["rank"])
                idade_dias = (agora - r["criado_em"]) / 86400
                recencia = 0.5 ** (idade_dias / 60)
                score = base * (0.6 + 0.4 * r["confianca"]) * (0.75 + 0.25 * recencia)
                por_id[r["id"]] = self._para_memoria(r, score=score)

        # ------------------------------------------------------ embedding
        if self.embeddings is not None:
            candidatos = self._buscar_por_embedding(
                consulta, usuario=usuario, tipo=tipo, limite=limite * 3, agora=agora)
            for mem in candidatos:
                # Uma memoria que ja veio do BM25 fica com o MAIOR dos dois
                # scores, nao a soma -- somar inflaria artificialmente algo
                # que so e relevante por uma via, e o objetivo aqui e unir
                # cobertura, nao acumular pontuacao.
                if mem.id in por_id:
                    por_id[mem.id].score = max(por_id[mem.id].score, mem.score)
                else:
                    por_id[mem.id] = mem

        resultados = sorted(por_id.values(), key=lambda m: m.score, reverse=True)
        resultados = [m for m in resultados if m.score >= score_minimo][:limite]

        if resultados:
            with self._lock:
                for m in resultados:
                    self.con.execute(
                        "UPDATE memorias SET acessos = acessos + 1, acessado_em = ? "
                        "WHERE id = ?", (agora, m.id),
                    )
                self.con.commit()
        return resultados

    def _buscar_por_embedding(
        self, consulta: str, *, usuario: str, tipo: str | None,
        limite: int, agora: float,
    ) -> list[Memoria]:
        """Metade semantica da busca hibrida. So chamada por `buscar` --
        nao publica porque nao faz sentido sem o lado BM25 complementando.
        """
        from .embeddings import blob_para_vetor, cosseno

        try:
            vetor_consulta = self.embeddings.gerar_consulta(consulta)
        except Exception:
            return []  # best-effort -- ver docstring de `buscar`

        sql = ("SELECT * FROM memorias WHERE ativo=1 AND usuario=? "
               "AND embedding IS NOT NULL")
        params: list = [usuario]
        if tipo:
            sql += " AND tipo=?"
            params.append(tipo)

        with self._lock:
            linhas = list(self.con.execute(sql, params))

        pontuados = []
        for r in linhas:
            try:
                vetor_mem = blob_para_vetor(r["embedding"])
            except Exception:
                continue  # blob corrompido/formato antigo -- pula, nao quebra
            sim = cosseno(vetor_consulta, vetor_mem)
            if sim <= 0:
                continue
            idade_dias = (agora - r["criado_em"]) / 86400
            recencia = 0.5 ** (idade_dias / 60)
            # Escala de score PROPOSITALMENTE no mesmo raio de grandeza do
            # BM25 (tipicamente 0-tantos, nao 0.0-1.0) -- cosseno bruto
            # (0 a 1) ficaria sempre perdendo na comparacao "max" contra
            # BM25 mesmo quando semanticamente e o resultado certo. 10x e
            # calibracao inicial; ajuste observando buscas reais.
            score = sim * 10 * (0.6 + 0.4 * r["confianca"]) * (0.75 + 0.25 * recencia)
            pontuados.append(self._para_memoria(r, score=score))

        pontuados.sort(key=lambda m: m.score, reverse=True)
        return pontuados[:limite]

    def fatos_nucleo(self, *, usuario: str, limite: int = 3) -> list[Memoria]:
        """Fatos estaveis e frequentemente uteis sobre a pessoa.

        Criterio: confianca alta e muitos acessos -- ou seja, o que ja se
        provou relevante em conversas anteriores.

        Existe porque a busca por palavra-chave falha justamente onde o fato
        mais importa. "como instalo o postgres?" nao casa com "usa Arch
        Linux", mas e exatamente ai que saber a distro muda a resposta
        inteira. Sao poucos itens, entao o custo em tokens e baixo.
        """
        with self._lock:
            linhas = list(self.con.execute(
                "SELECT * FROM memorias WHERE tipo='semantica' AND ativo=1 "
                "AND usuario=? AND confianca >= 0.8 "
                "ORDER BY acessos DESC, confianca DESC LIMIT ?",
                (usuario, limite),
            ))
        return [self._para_memoria(r) for r in linhas]

    def listar(self, *, usuario: str, tipo: str | None = None,
               limite: int = 50) -> list[Memoria]:
        sql = "SELECT * FROM memorias WHERE ativo=1 AND usuario=?"
        params: list = [usuario]
        if tipo:
            sql += " AND tipo=?"
            params.append(tipo)
        sql += " ORDER BY criado_em DESC LIMIT ?"
        params.append(limite)
        with self._lock:
            return [self._para_memoria(r) for r in self.con.execute(sql, params)]

    def contar(self, *, usuario: str | None = None) -> dict[str, int]:
        """Quantas memorias por tipo. Sem `usuario`, conta o banco inteiro."""
        sql = "SELECT tipo, COUNT(*) n FROM memorias WHERE ativo=1"
        params: list = []
        if usuario:
            sql += " AND usuario=?"
            params.append(usuario)
        sql += " GROUP BY tipo"
        with self._lock:
            return {r["tipo"]: r["n"] for r in self.con.execute(sql, params)}

    @staticmethod
    def _para_memoria(r, score: float = 0.0) -> Memoria:
        chaves = r.keys()
        return Memoria(
            id=r["id"], tipo=r["tipo"], conteudo=r["conteudo"],
            usuario=r["usuario"] if "usuario" in chaves else "default",
            fonte=r["fonte"], confianca=r["confianca"],
            criado_em=r["criado_em"], acessos=r["acessos"],
            meta=json.loads(r["meta"]) if r["meta"] else None,
            score=score,
            consolidado_em=r["consolidado_em"] if "consolidado_em" in chaves else None,
        )

    # ---------- historico de conversa ----------

    def registrar_turno(self, papel: str, conteudo: str, *, usuario: str,
                        sessao: str | None = None) -> None:
        with self._lock:
            self.con.execute(
                "INSERT INTO conversas (usuario, papel, conteudo, criado_em, sessao) "
                "VALUES (?,?,?,?,?)",
                (usuario, papel, conteudo, time.time(), sessao),
            )
            self.con.commit()

    def historico(self, *, usuario: str, limite: int = 12) -> list[dict]:
        with self._lock:
            linhas = list(self.con.execute(
                "SELECT papel, conteudo, criado_em FROM conversas "
                "WHERE usuario=? ORDER BY id DESC LIMIT ?",
                (usuario, limite),
            ))
        return [{"role": r["papel"], "content": r["conteudo"], "em": r["criado_em"]}
                for r in reversed(linhas)]

    def ultimo_turno_em(self, *, usuario: str | None = None) -> float | None:
        sql = "SELECT MAX(criado_em) m FROM conversas"
        params: list = []
        if usuario:
            sql += " WHERE usuario=?"
            params.append(usuario)
        with self._lock:
            r = self.con.execute(sql, params).fetchone()
        return r["m"] if r and r["m"] else None

    # ---------- pessoas ----------

    def pessoa(self, usuario: str) -> dict:
        """Registro de identidade. Cria na primeira vez que a pessoa aparece."""
        with self._lock:
            r = self.con.execute(
                "SELECT * FROM pessoas WHERE usuario=?", (usuario,)).fetchone()
            if r is None:
                agora = time.time()
                self.con.execute(
                    "INSERT INTO pessoas (usuario, nome, relacao, turnos, criado_em, "
                    "visto_em) VALUES (?,?,?,?,?,?)",
                    (usuario, "", "desconhecido", 0, agora, agora),
                )
                self.con.commit()
                return {"usuario": usuario, "nome": "", "relacao": "desconhecido",
                        "turnos": 0}
            return {"usuario": r["usuario"], "nome": r["nome"] or "",
                    "relacao": r["relacao"], "turnos": r["turnos"] or 0}

    def salvar_pessoa(self, usuario: str, *, nome: str | None = None,
                      relacao: str | None = None, turnos: int | None = None) -> None:
        self.pessoa(usuario)  # garante que existe
        campos, params = [], []
        if nome is not None:
            campos.append("nome=?")
            params.append(nome)
        if relacao is not None:
            campos.append("relacao=?")
            params.append(relacao)
        if turnos is not None:
            campos.append("turnos=?")
            params.append(turnos)
        campos.append("visto_em=?")
        params.append(time.time())
        params.append(usuario)
        with self._lock:
            self.con.execute(
                f"UPDATE pessoas SET {', '.join(campos)} WHERE usuario=?", params)
            self.con.commit()

    def usuarios(self) -> list[str]:
        with self._lock:
            return [r["usuario"] for r in self.con.execute(
                "SELECT DISTINCT usuario FROM memorias ORDER BY usuario")]

    # ---------- suporte a consolidacao.py ----------

    def candidatos_para_consolidar(
        self, *, usuario: str, tipo: str | None = None,
        dias_minimo: float = 14.0, limite: int = 50,
    ) -> list[Memoria]:
        """Memorias elegiveis para virar resumo semantico.

        Criterio: mais velhas que `dias_minimo`, ainda ativas, ainda com
        embedding gerado (sem embedding nao ha como agrupar por
        similaridade -- ver consolidacao.py), e nunca consolidadas.
        `dias_minimo` existe para nao consolidar algo ainda "quente": uma
        memoria de ontem ainda deveria aparecer como ela mesma no
        contexto, nao pre-resumida.
        """
        limite_tempo = time.time() - dias_minimo * 86400
        sql = ("SELECT * FROM memorias WHERE usuario=? AND ativo=1 "
               "AND embedding IS NOT NULL AND consolidado_em IS NULL "
               "AND criado_em < ?")
        params: list = [usuario, limite_tempo]
        if tipo:
            sql += " AND tipo=?"
            params.append(tipo)
        sql += " ORDER BY criado_em ASC LIMIT ?"
        params.append(limite)

        with self._lock:
            linhas = list(self.con.execute(sql, params))
        return [self._para_memoria(r) for r in linhas]

    def marcar_consolidadas(self, ids: list[int]) -> None:
        """Marca memorias como absorvidas por um resumo.

        NAO desativa (`ativo` continua 1) -- a memoria original permanece
        buscavel. Consolidar e sobre criar um resumo ADICIONAL para reduzir
        o que entra no contexto por padrao, nao sobre apagar detalhe.
        """
        if not ids:
            return
        agora = time.time()
        with self._lock:
            placeholders = ",".join("?" * len(ids))
            self.con.execute(
                f"UPDATE memorias SET consolidado_em=? WHERE id IN ({placeholders})",
                [agora, *ids],
            )
            self.con.commit()

    def fechar(self) -> None:
        with self._lock:
            self.con.close()