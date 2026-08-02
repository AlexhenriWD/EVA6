"""
Orquestrador -- o ciclo cognitivo completo.

    mensagem
        |
        v
    Decision Engine        decide o que buscar/executar (não conversa)
        |
        +--> Memória       fatos, episódios, preferências
        +--> Ferramentas   JSON, nunca texto
        +--> Estado        energia, curiosidade, estresse
        |
        v
    Context Builder        monta o pacote no formato que a EVA foi treinada
        |
        v
    EVA (LLM)              escreve a resposta -- só ela escreve português
        |
        v
    Pós-processo           extrai memórias novas, atualiza estado

A ordem importa: a EVA aparece por último, e é a única peça que produz
texto para o usuário. Todo o resto produz dado estruturado.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import EVAConfig, carregar_config
from .context import ContextBuilder
from .decision import DecisorPorLLM, DecisorPorRegras, Plano
from .llm import ClienteLLM, ErroLLM
from .memory.extractor import extrair_por_regras
from .memory.store import BancoMemoria
from .state import GerenciadorEstado
from .tools.builtin import carregar_ferramentas


@dataclass
class Resultado:
    """Tudo que aconteceu num turno. Útil para debug e para a CLI mostrar
    o que rolou por baixo."""
    resposta: str
    plano: Plano
    memorias_usadas: dict = field(default_factory=dict)
    ferramentas: dict = field(default_factory=dict)
    memorias_novas: list = field(default_factory=list)
    contexto: dict = field(default_factory=dict)
    ms: int = 0
    erro: str | None = None


class EVA:
    def __init__(self, config: EVAConfig | None = None):
        self.cfg = config or carregar_config()

        self.memoria = BancoMemoria(self.cfg.memoria.caminho_db)
        self.estado = GerenciadorEstado(self.cfg.estado.caminho, self.cfg.estado.inercia)
        self.ferramentas = carregar_ferramentas()
        self.llm = ClienteLLM(self.cfg.llm)
        self.builder = ContextBuilder(self.cfg)

        if self.cfg.decisao.usar_llm:
            from .llm import ClienteLLM as _C
            cliente_dec = _C(self.cfg.decisao)
            self.decisor = DecisorPorLLM(cliente_dec, self.ferramentas)
        else:
            self.decisor = DecisorPorRegras()

        # Tempo parado recupera energia -- sem isso a EVA acumularia cansaço
        # entre sessões e ficaria permanentemente exausta.
        self.estado.aplicar_tempo_decorrido()

    # ------------------------------------------------------------ ciclo

    def responder(self, mensagem: str, stream: bool = False):
        """Executa um turno completo. Com stream=True devolve um gerador."""
        inicio = time.time()
        mensagem = mensagem.strip()
        if not mensagem:
            return Resultado(resposta="", plano=Plano(), erro="mensagem vazia")

        historico = self.memoria.historico(self.cfg.memoria.janela_historico)

        # 1. decidir
        plano = self.decisor.decidir(mensagem, historico)

        # 2. buscar memória
        memorias = self._buscar_memorias(plano)

        # 3. executar ferramentas
        resultados = self._executar_ferramentas(plano)

        # 4. montar contexto
        ctx = self.builder.montar(
            plano=plano,
            memorias=memorias,
            resultados_ferramentas=resultados,
            estado=self.estado.estado,
            historico=historico,
        )
        mensagens = ctx.para_chat(mensagem)

        if stream:
            return self._responder_stream(mensagem, plano, memorias, resultados, ctx,
                                          mensagens, inicio)

        # 5. gerar resposta
        try:
            resposta = self.llm.completar(mensagens)
            erro = None
        except ErroLLM as e:
            resposta = ""
            erro = str(e)

        # 6. pós-processo
        novas = self._pos_processar(mensagem, resposta, plano, sucesso=erro is None)

        return Resultado(
            resposta=resposta,
            plano=plano,
            memorias_usadas={k: [m.conteudo for m in v] for k, v in memorias.items() if v},
            ferramentas=resultados,
            memorias_novas=novas,
            contexto=ctx.bruto,
            ms=int((time.time() - inicio) * 1000),
            erro=erro,
        )

    def _responder_stream(self, mensagem, plano, memorias, resultados, ctx,
                          mensagens, inicio):
        """Gera a resposta em pedaços; o pós-processo roda no fim."""
        partes: list[str] = []

        def gerar():
            erro = None
            try:
                for pedaco in self.llm.completar_stream(mensagens):
                    partes.append(pedaco)
                    yield pedaco
            except ErroLLM as e:
                erro = str(e)
                yield f"\n[erro: {erro}]"

            resposta = "".join(partes)
            novas = self._pos_processar(mensagem, resposta, plano, sucesso=erro is None)
            gerar.resultado = Resultado(
                resposta=resposta, plano=plano,
                memorias_usadas={k: [m.conteudo for m in v] for k, v in memorias.items() if v},
                ferramentas=resultados, memorias_novas=novas, contexto=ctx.bruto,
                ms=int((time.time() - inicio) * 1000), erro=erro,
            )

        gerar.resultado = None
        return gerar()

    # ------------------------------------------------------------ etapas

    def _buscar_memorias(self, plano: Plano) -> dict[str, list]:
        if not plano.precisa_memoria or not plano.consulta_memoria:
            return {}

        limites = {
            "semantica": self.cfg.memoria.max_fatos,
            "episodica": self.cfg.memoria.max_episodios,
            "procedural": self.cfg.memoria.max_procedimentos,
            "personalidade": self.cfg.memoria.max_personalidade,
        }
        saida = {}
        for tipo, limite in limites.items():
            achadas = self.memoria.buscar(
                plano.consulta_memoria, tipo=tipo, limite=limite,
                score_minimo=self.cfg.memoria.score_minimo,
            )
            if achadas:
                saida[tipo] = achadas

        # Fatos-núcleo entram SEMPRE, sem depender de busca.
        #
        # Motivo: busca por palavra-chave falha justamente onde o fato mais
        # importa. "como instalo o postgres?" não casa com "usa Arch Linux",
        # mas é exatamente aí que saber a distro muda a resposta inteira.
        # São poucos itens (os mais acessados e de maior confiança), então
        # o custo em tokens é baixo e o ganho é alto.
        nucleo = self._fatos_nucleo()
        if nucleo:
            existentes = {m.id for m in saida.get("semantica", [])}
            extras = [m for m in nucleo if m.id not in existentes]
            if extras:
                saida["semantica"] = (saida.get("semantica", []) + extras)[
                    : self.cfg.memoria.max_fatos
                ]
        return saida

    def _fatos_nucleo(self, limite: int = 3) -> list:
        """Fatos estáveis e frequentemente úteis sobre a pessoa.

        Critério: confiança alta e muitos acessos -- ou seja, o que já se
        provou relevante em conversas anteriores.
        """
        from .memory.store import Memoria
        linhas = self.memoria.con.execute(
            "SELECT * FROM memorias WHERE tipo='semantica' AND ativo=1 "
            "AND confianca >= 0.8 ORDER BY acessos DESC, confianca DESC LIMIT ?",
            (limite,),
        )
        return [
            Memoria(id=r["id"], tipo=r["tipo"], conteudo=r["conteudo"], fonte=r["fonte"],
                    confianca=r["confianca"], criado_em=r["criado_em"], acessos=r["acessos"])
            for r in linhas
        ]

    def _executar_ferramentas(self, plano: Plano) -> dict:
        if not plano.precisa_ferramenta:
            return {}
        saida = {}
        for f in plano.ferramentas:
            nome = f.get("nome")
            if not nome:
                continue
            saida[nome] = self.ferramentas.executar(nome, **(f.get("args") or {}))
        return saida

    def _pos_processar(self, mensagem: str, resposta: str, plano: Plano,
                       sucesso: bool) -> list:
        # histórico
        self.memoria.registrar_turno("user", mensagem)
        if resposta:
            self.memoria.registrar_turno("assistant", resposta)

        # novas memórias
        novas = []
        if plano.guardar_memoria:
            for item in extrair_por_regras(mensagem):
                id_ = self.memoria.adicionar(
                    tipo=item["tipo"], conteudo=item["conteudo"],
                    fonte=item["fonte"], confianca=item["confianca"],
                )
                if id_:
                    novas.append(item)

        # estado interno
        #
        # O foco registra o assunto que vem dominando as conversas, e entra
        # no contexto de turnos futuros. Intenções sensíveis ficam de fora:
        # "foco: crise" apareceria no contexto de conversas seguintes, muito
        # depois do momento ter passado, e enquadraria a pessoa por algo que
        # ela disse uma vez.
        FOCO_IGNORADO = {"conversa", "crise", "emocional", "saudacao", "sobre_si"}
        self.estado.registrar_interacao({
            "novidade": plano.novidade,
            "complexidade": plano.complexidade,
            "carga_emocional": plano.carga_emocional,
            "sucesso": sucesso,
            "assunto": plano.intencao if plano.intencao not in FOCO_IGNORADO else None,
        })
        return novas

    # ------------------------------------------------------------- extras

    def lembrar(self, conteudo: str, tipo: str = "semantica") -> int | None:
        """Adiciona uma memória manualmente."""
        return self.memoria.adicionar(tipo, conteudo, fonte="manual", confianca=1.0)

    def esquecer(self, termo: str) -> int:
        return self.memoria.esquecer_por_texto(termo)

    def diagnostico(self) -> dict:
        return {
            "llm_disponivel": self.llm.disponivel(),
            "modelos": self.llm.modelos(),
            "modelo_configurado": self.cfg.llm.modelo,
            "url": self.cfg.llm.base_url,
            "memorias": self.memoria.contar(),
            "estado": self.estado.estado.para_contexto(),
            "interacoes": self.estado.estado.total_interacoes,
            "ferramentas": self.ferramentas.nomes(),
            "decisor": "llm" if self.cfg.decisao.usar_llm else "regras",
            "banco": str(self.cfg.memoria.caminho_db),
        }

    def fechar(self) -> None:
        self.estado.salvar()
        self.memoria.fechar()
