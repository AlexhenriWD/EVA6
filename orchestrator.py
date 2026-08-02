"""
Orquestrador -- o ciclo cognitivo completo.

    mensagem
        |
        v
    Decision Engine        decide o que buscar/executar (não conversa)
        |
        +--> Identidade    quem é a pessoa -> linha situacional do prompt
        +--> Memória       fatos, episódios, preferências (por pessoa)
        +--> Ferramentas   JSON, nunca texto
        +--> Estado        energia, curiosidade, estresse (dela, global)
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

SÍNCRONO DE PROPÓSITO
---------------------
`responder` é síncrono: é mais simples de testar e a CLI usa direto. As
integrações que vivem num laço asyncio (bridge, Discord) chamam via
`responder_async`, que joga numa thread. O que torna isso seguro é o lock
do BancoMemoria -- sem ele, duas pessoas falando ao mesmo tempo corrompem
o cursor do SQLite.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .config import EVAConfig, carregar_config
from .context import ContextBuilder
from .decision import DecisorPorLLM, DecisorPorRegras, Plano
from .identity import Pessoa, promover
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
    usuario: str = ""
    memorias_usadas: dict = field(default_factory=dict)
    ferramentas: dict = field(default_factory=dict)
    memorias_novas: list = field(default_factory=list)
    contexto: dict = field(default_factory=dict)
    system: str = ""
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
            self.decisor = DecisorPorLLM(ClienteLLM(self.cfg.decisao), self.ferramentas)
        else:
            self.decisor = DecisorPorRegras()

        # Tempo parado recupera energia -- sem isso a EVA acumularia cansaço
        # entre sessões e ficaria permanentemente exausta.
        self.estado.aplicar_tempo_decorrido()

        self._registrar_criador()

    def _registrar_criador(self) -> None:
        """Marca o dono da instância como criador, se ainda não estiver.

        Só age quando a relação ainda é a padrão: se você reclassificar
        alguém à mão, o reinício não desfaz.
        """
        uid = self.cfg.usuario
        p = self.memoria.pessoa(uid)
        if p["relacao"] == "desconhecido":
            self.memoria.salvar_pessoa(uid, nome=self.cfg.nome_criador,
                                       relacao="criador")

    # ------------------------------------------------------------ ciclo

    def responder(
        self,
        mensagem: str,
        *,
        usuario: str | None = None,
        stream: bool = False,
        modo_voz: bool = False,
        contexto_visual: str | None = None,
    ):
        """Executa um turno completo. Com stream=True devolve um gerador.

        `usuario` é o id de quem falou (id do Discord, por exemplo). Sem ele,
        cai no dono da instância -- que é o certo para a CLI, e errado para
        qualquer integração multiusuário, então as integrações passam sempre.
        """
        inicio = time.time()
        usuario = usuario or self.cfg.usuario
        mensagem = mensagem.strip()
        if not mensagem:
            return Resultado(resposta="", plano=Plano(), usuario=usuario,
                             erro="mensagem vazia")

        historico = self.memoria.historico(
            usuario=usuario, limite=self.cfg.memoria.janela_historico)

        # 1. decidir
        plano = self.decisor.decidir(mensagem, historico)

        # 2. quem é a pessoa
        identidade = self._identidade(usuario)

        # 3. buscar memória
        memorias = self._buscar_memorias(plano, usuario)

        # 4. executar ferramentas
        resultados = self._executar_ferramentas(plano)

        # 5. montar contexto
        ctx = self.builder.montar(
            plano=plano,
            memorias=memorias,
            resultados_ferramentas=resultados,
            estado=self.estado.estado,
            historico=historico,
            identidade=identidade,
            modo_voz=modo_voz,
            contexto_visual=contexto_visual,
        )
        mensagens = ctx.para_chat(mensagem)

        # Em voz, o teto de tokens é bem menor: a mediana do dataset é 74
        # caracteres e o p99 é 222. 400 tokens viram uns 40 segundos de fala,
        # tempo demais para alguém esperando numa call.
        teto = self.cfg.llm.max_tokens_voz if modo_voz else self.cfg.llm.max_tokens

        if stream:
            return self._responder_stream(
                mensagem, usuario, plano, memorias, resultados, ctx,
                mensagens, teto, inicio)

        # 6. gerar resposta
        try:
            resposta = self.llm.completar(mensagens, max_tokens=teto)
            erro = None
        except ErroLLM as e:
            resposta = ""
            erro = str(e)

        # 7. pós-processo
        novas = self._pos_processar(mensagem, resposta, plano, usuario,
                                    sucesso=erro is None)

        return Resultado(
            resposta=resposta,
            plano=plano,
            usuario=usuario,
            memorias_usadas={k: [m.conteudo for m in v] for k, v in memorias.items() if v},
            ferramentas=resultados,
            memorias_novas=novas,
            contexto=ctx.bruto,
            system=ctx.system,
            ms=int((time.time() - inicio) * 1000),
            erro=erro,
        )

    async def responder_async(self, mensagem: str, **kwargs):
        """Versão para quem vive num laço asyncio.

        O trabalho pesado (HTTP para o LLM, SQLite) é bloqueante e roda numa
        thread. Chamar `responder` direto de dentro do laço trava o event
        loop por segundos: o bridge de voz perde frames e o heartbeat do
        Discord atrasa.
        """
        return await asyncio.to_thread(
            lambda: self.responder(mensagem, **kwargs))

    def _responder_stream(self, mensagem, usuario, plano, memorias, resultados,
                          ctx, mensagens, teto, inicio):
        """Gera a resposta em pedaços; o pós-processo roda no fim."""
        partes: list[str] = []

        def gerar():
            erro = None
            try:
                for pedaco in self.llm.completar_stream(mensagens, max_tokens=teto):
                    partes.append(pedaco)
                    yield pedaco
            except ErroLLM as e:
                erro = str(e)
                yield f"\n[erro: {erro}]"

            resposta = "".join(partes)
            novas = self._pos_processar(mensagem, resposta, plano, usuario,
                                        sucesso=erro is None)
            gerar.resultado = Resultado(
                resposta=resposta, plano=plano, usuario=usuario,
                memorias_usadas={k: [m.conteudo for m in v] for k, v in memorias.items() if v},
                ferramentas=resultados, memorias_novas=novas, contexto=ctx.bruto,
                system=ctx.system,
                ms=int((time.time() - inicio) * 1000), erro=erro,
            )

        gerar.resultado = None
        return gerar()

    # ------------------------------------------------------------ etapas

    def _identidade(self, usuario: str) -> str | None:
        """A segunda linha do system prompt, ou None.

        None é o caso mais treinado (784 dos 1.135 exemplos não têm segunda
        linha), então não inventar é seguro.
        """
        reg = self.memoria.pessoa(usuario)
        p = Pessoa(user_id=usuario, nome=reg["nome"], relacao=reg["relacao"],
                   turnos=reg["turnos"])
        if promover(p, self.cfg.identidade.turnos_para_conhecido):
            self.memoria.salvar_pessoa(usuario, relacao=p.relacao)
        return p.linha()

    def _buscar_memorias(self, plano: Plano, usuario: str) -> dict[str, list]:
        saida: dict[str, list] = {}

        if plano.precisa_memoria and plano.consulta_memoria:
            limites = {
                "semantica": self.cfg.memoria.max_fatos,
                "episodica": self.cfg.memoria.max_episodios,
                "procedural": self.cfg.memoria.max_procedimentos,
                "personalidade": self.cfg.memoria.max_personalidade,
            }
            for tipo, limite in limites.items():
                achadas = self.memoria.buscar(
                    plano.consulta_memoria, usuario=usuario, tipo=tipo,
                    limite=limite, score_minimo=self.cfg.memoria.score_minimo,
                )
                if achadas:
                    saida[tipo] = achadas

        # Fatos-núcleo entram SEMPRE, mesmo quando o plano dispensa memória.
        # A busca por palavra-chave falha justamente onde o fato mais importa,
        # e "não precisa de memória" é uma decisão sobre a pergunta, não sobre
        # o que é preciso saber para respondê-la bem.
        nucleo = self.memoria.fatos_nucleo(usuario=usuario)
        if nucleo:
            existentes = {m.id for m in saida.get("semantica", [])}
            extras = [m for m in nucleo if m.id not in existentes]
            if extras:
                saida["semantica"] = (saida.get("semantica", []) + extras)[
                    : self.cfg.memoria.max_fatos
                ]
        return saida

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
                       usuario: str, sucesso: bool) -> list:
        # histórico
        self.memoria.registrar_turno("user", mensagem, usuario=usuario)
        if resposta:
            self.memoria.registrar_turno("assistant", resposta, usuario=usuario)

        reg = self.memoria.pessoa(usuario)
        self.memoria.salvar_pessoa(usuario, turnos=reg["turnos"] + 1)

        # novas memórias
        novas = []
        if plano.guardar_memoria:
            for item in extrair_por_regras(mensagem):
                id_ = self.memoria.adicionar(
                    tipo=item["tipo"], conteudo=item["conteudo"],
                    usuario=usuario, fonte=item["fonte"],
                    confianca=item["confianca"],
                )
                if id_:
                    novas.append(item)

        # estado interno
        #
        # Global de propósito: energia, curiosidade e estresse são dela, não
        # da conversa. Uma discussão difícil com alguém deixa a EVA cansada
        # também para a próxima pessoa -- que é como funciona com gente.
        #
        # O foco registra o assunto que vem dominando as conversas. Intenções
        # sensíveis ficam de fora: "foco: crise" apareceria no contexto de
        # conversas seguintes, muito depois do momento ter passado, e
        # enquadraria a pessoa por algo que ela disse uma vez.
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

    def lembrar(self, conteudo: str, tipo: str = "semantica",
                usuario: str | None = None) -> int | None:
        """Adiciona uma memória manualmente."""
        return self.memoria.adicionar(
            tipo, conteudo, usuario=usuario or self.cfg.usuario,
            fonte="manual", confianca=1.0)

    def esquecer(self, termo: str, usuario: str | None = None) -> int:
        return self.memoria.esquecer_por_texto(
            termo, usuario=usuario or self.cfg.usuario)

    def apresentar(self, usuario: str, nome: str,
                   relacao: str = "conhecido") -> None:
        """Registra quem é alguém. Muda a linha situacional do prompt."""
        self.memoria.salvar_pessoa(usuario, nome=nome, relacao=relacao)

    def diagnostico(self) -> dict:
        return {
            "llm_disponivel": self.llm.disponivel(),
            "modelos": self.llm.modelos(),
            "modelo_configurado": self.cfg.llm.modelo,
            "url": self.cfg.llm.base_url,
            "formato_contexto": self.cfg.llm.formato_contexto,
            "memorias": self.memoria.contar(),
            "usuarios": self.memoria.usuarios(),
            "estado": self.estado.estado.para_contexto(),
            "interacoes": self.estado.estado.total_interacoes,
            "ferramentas": self.ferramentas.nomes(),
            "decisor": "llm" if self.cfg.decisao.usar_llm else "regras",
            "banco": str(self.cfg.memoria.caminho_db),
        }

    def fechar(self) -> None:
        self.estado.salvar()
        self.memoria.fechar()