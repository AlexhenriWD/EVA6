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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import EVAConfig, carregar_config
from .context import ContextBuilder
from .decision import DecisorPorLLM, DecisorPorRegras, Plano, clientes_decisao
from .identity import Pessoa, promover
from .llm import ClienteLLM, ErroLLM, STOP_CONVERSA
from .memory.extractor import extrair_por_llm, extrair_por_regras, extrair_personalidade_propria
from .memory.store import BancoMemoria, USUARIO_HISTORIA
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


class RespostaEmStream:
    """Iterável dos pedaços de texto de uma resposta em streaming.

    `.resultado` fica None até a iteração terminar; depois de esgotado,
    tem o Resultado completo (pós-processamento já feito). Existe porque
    generator não aceita atributo arbitrário -- o padrão antigo
    (`gerar.resultado`) setava no objeto FUNÇÃO, que ninguém de fora
    consegue alcançar. Sem consumidor real de stream=True até hoje,
    ninguém tinha notado.
    """
    def __init__(self):
        self._gerador = None
        self.resultado: Resultado | None = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._gerador is None:
            raise StopIteration
        return next(self._gerador)


class _ConfigExtrator:
    """Adaptador leve: ClienteLLM só precisa destes atributos (duck typing).

    Não é um dataclass em EVAConfig porque não é configuração de primeira
    classe -- é a ponte entre MemoriaConfig (onde os valores vivem de
    verdade) e o formato que ClienteLLM espera.
    """
    def __init__(self, mem_cfg):
        self.base_url = mem_cfg.extrator_base_url
        self.api_key = mem_cfg.extrator_api_key
        self.modelo = mem_cfg.extrator_modelo
        self.temperatura = 0.0  # extração é tarefa estruturada, não criativa
        self.top_p = 0.9
        self.max_tokens = 300
        self.timeout = mem_cfg.extrator_timeout


class EVA:
    def __init__(self, config: EVAConfig | None = None):
        self.cfg = config or carregar_config()

        embeddings = None
        if self.cfg.memoria.usar_embeddings:
            from .memory.embeddings import ClienteEmbeddings
            embeddings = ClienteEmbeddings(
                base_url=self.cfg.memoria.embeddings_base_url,
                modelo=self.cfg.memoria.embeddings_modelo,
                api_key=self.cfg.memoria.embeddings_api_key,
                timeout=self.cfg.memoria.embeddings_timeout,
            )
        self.memoria = BancoMemoria(self.cfg.memoria.caminho_db, embeddings=embeddings)
        self.estado = GerenciadorEstado(self.cfg.estado.caminho, self.cfg.estado.inercia)
        self.ferramentas = carregar_ferramentas()
        self.llm = ClienteLLM(self.cfg.llm)
        self.builder = ContextBuilder(self.cfg)

        # Cliente separado do de conversa: extração de fatos é tarefa
        # estruturada (JSON), não conversa, e pode ser um modelo diferente
        # no futuro sem afetar o eva-3b conversacional. `_config_extrator`
        # é um objeto simples que só carrega o que ClienteLLM precisa
        # (base_url, api_key, modelo, temperatura, top_p, max_tokens,
        # timeout) -- não é um dataclass registrado em EVAConfig porque só
        # existe para essa injeção.
        self.llm_extrator = ClienteLLM(_ConfigExtrator(self.cfg.memoria))

        if self.cfg.decisao.usar_llm:
            principal, reserva = clientes_decisao(self.cfg.decisao)
            self.decisor = DecisorPorLLM(principal, self.ferramentas, cliente_reserva=reserva)
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
        modo_multicanal: bool = False,
    ):
        """Executa um turno completo. Com stream=True devolve um gerador.

        `usuario` é o id de quem falou (id do Discord, por exemplo). Sem ele,
        cai no dono da instância -- que é o certo para a CLI, e errado para
        qualquer integração multiusuário, então as integrações passam sempre.

        `modo_multicanal=True` é para quando `mensagem` já vem pré-formatada
        como várias linhas "[canal] quem: texto" (ver bridge_client, que
        decide quando agregar várias falas antes de chamar aqui). Nesse
        caso `usuario` continua sendo uma pessoa só -- por ora, quem
        encerrou a janela de agregação -- porque memória/identidade são por
        pessoa; não existe ainda um conceito de "memória do grupo". Ponto
        conhecido de simplificação, não esquecido.
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

        # 3. buscar memória + 4. executar ferramentas -- independentes
        # entre si (nenhuma usa o resultado da outra, as duas só dependem
        # de `plano`/`usuario`/`mensagem`), então rodam em paralelo em vez
        # de sequencial. O ganho real aparece quando plano.ferramentas
        # inclui busca web: o SearXNG pode levar 1-3s (ou até os 10s de
        # timeout da ferramenta, se o SearXNG estiver frio), e antes esse
        # tempo se somava DEPOIS do tempo de busca de memória. Agora os
        # dois correm ao mesmo tempo, e o turno espera só o mais lento dos
        # dois, não a soma dos dois.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futuro_memorias = pool.submit(self._buscar_memorias, plano, usuario, mensagem)
            futuro_ferramentas = pool.submit(self._executar_ferramentas, plano)
            memorias = futuro_memorias.result()
            resultados = futuro_ferramentas.result()

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
            modo_multicanal=modo_multicanal,
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
            resposta = self.llm.completar(mensagens, max_tokens=teto, parar=STOP_CONVERSA)
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


    def pre_visualizar(
        self, mensagem: str, *, usuario: str | None = None,
        modo_voz: bool = False, contexto_visual: str | None = None,
    ) -> dict:
        """Monta o que SERIA enviado ao modelo, sem chamar o modelo.

        Reaproveita passo a passo a mesma pipeline de responder() (decisão,
        identidade, memória, ferramentas, contexto) -- só para antes do
        passo de gerar a resposta. Existe para o dashboard de debug: ver
        o prompt exato que seria montado para uma mensagem de teste, sem
        gastar uma chamada real ao modelo nem esperar geração.

        Duplica ~10 linhas de responder() de propósito, em vez de extrair
        um helper compartilhado -- responder() já é código testado e em
        produção (stream, tratamento de erro do LLM, pós-processamento);
        arriscar isso por uma ferramenta de debug não vale a pena.

        Síncrono e thread-safe (BancoMemoria usa RLock) -- chamável direto
        da thread do servidor do dashboard, sem ponte com asyncio.
        """
        usuario = usuario or self.cfg.usuario
        mensagem = mensagem.strip()
        historico = self.memoria.historico(
            usuario=usuario, limite=self.cfg.memoria.janela_historico)
        plano = self.decisor.decidir(mensagem, historico)
        identidade = self._identidade(usuario)
        memorias = self._buscar_memorias(plano, usuario, mensagem)
        resultados = self._executar_ferramentas(plano)
        ctx = self.builder.montar(
            plano=plano, memorias=memorias, resultados_ferramentas=resultados,
            estado=self.estado.estado, historico=historico, identidade=identidade,
            modo_voz=modo_voz, contexto_visual=contexto_visual,
        )
        teto = self.cfg.llm.max_tokens_voz if modo_voz else self.cfg.llm.max_tokens
        return {
            "usuario": usuario,
            "system": ctx.system,
            "mensagens": ctx.para_chat(mensagem),
            "plano": plano.to_dict(),
            "memorias_usadas": {k: [m.conteudo for m in v]
                               for k, v in memorias.items() if v},
            "ferramentas": resultados,
            "max_tokens": teto,
            "identidade": identidade,
        }

    def falar_sozinha(self, ideia: str, *, usuario: str | None = None,
                      modo_voz: bool = True) -> Resultado:
        """Produz uma fala espontânea a partir de um impulso aprovado.

        Quem decide SE ela fala é o portão em eva/consciousness.py. Aqui ela
        já tem permissão -- este método só escreve.

        A lista de mensagens termina no histórico, sem turno de `user`: não
        houve pergunta. Servidores compatíveis com a API da OpenAI aplicam o
        template e abrem o turno do assistente de qualquer forma. Se o seu
        não abrir, o sintoma é resposta vazia, e o conserto é acrescentar
        uma mensagem de user mínima -- mas evite, porque inventar um turno
        de usuário que não existiu polui o histórico.
        """
        inicio = time.time()
        usuario = usuario or self.cfg.usuario
        plano = Plano(intencao="iniciativa", precisa_memoria=True,
                      guardar_memoria=False, consulta_memoria=ideia)

        historico = self.memoria.historico(
            usuario=usuario, limite=self.cfg.memoria.janela_historico)
        memorias = self._buscar_memorias(plano, usuario, ideia)

        ctx = self.builder.montar(
            plano=plano, memorias=memorias, resultados_ferramentas={},
            estado=self.estado.estado, historico=historico,
            identidade=self._identidade(usuario),
            modo_voz=modo_voz, iniciativa=ideia,
        )
        mensagens = [{"role": "system", "content": ctx.system}] + ctx.mensagens

        teto = self.cfg.llm.max_tokens_voz if modo_voz else self.cfg.llm.max_tokens
        try:
            resposta = self.llm.completar(mensagens, max_tokens=teto, parar=STOP_CONVERSA)
            erro = None
        except ErroLLM as e:
            resposta, erro = "", str(e)

        # Só o lado dela entra no histórico -- não houve turno de usuário.
        # E `guardar_memoria=False` no plano evita o outro erro: extrair
        # "fato sobre o usuário" de uma frase que quem escreveu foi ela.
        if resposta:
            self.memoria.registrar_turno("assistant", resposta, usuario=usuario)

        return Resultado(resposta=resposta, plano=plano, usuario=usuario,
                         contexto=ctx.bruto, system=ctx.system,
                         ms=int((time.time() - inicio) * 1000), erro=erro)

    async def falar_sozinha_async(self, ideia: str, **kwargs) -> Resultado:
        return await asyncio.to_thread(lambda: self.falar_sozinha(ideia, **kwargs))
    
    def _responder_stream(self, mensagem, usuario, plano, memorias, resultados,
                          ctx, mensagens, teto, inicio):
        """Gera a resposta em pedaços; o pós-processo roda no fim."""
        partes: list[str] = []
        envelope = RespostaEmStream()

        def gerar():
            erro = None
            try:
                for pedaco in self.llm.completar_stream(mensagens, max_tokens=teto, parar=STOP_CONVERSA):
                    partes.append(pedaco)
                    yield pedaco
            except ErroLLM as e:
                # Erro do LLM não vira texto pra falar/mostrar -- fica só em
                # resultado.erro, igual o caminho não-streaming já faz.
                erro = str(e)

            resposta = "".join(partes)
            novas = self._pos_processar(mensagem, resposta, plano, usuario,
                                        sucesso=erro is None)
            envelope.resultado = Resultado(
                resposta=resposta, plano=plano, usuario=usuario,
                memorias_usadas={k: [m.conteudo for m in v] for k, v in memorias.items() if v},
                ferramentas=resultados, memorias_novas=novas, contexto=ctx.bruto,
                system=ctx.system,
                ms=int((time.time() - inicio) * 1000), erro=erro,
            )

        envelope._gerador = gerar()
        return envelope

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

    def _buscar_memorias(self, plano: Plano, usuario: str, mensagem: str) -> dict[str, list]:
        saida: dict[str, list] = {}

        if plano.precisa_memoria and plano.consulta_memoria:
            limites = {
                "semantica": self.cfg.memoria.max_fatos,
                "episodica": self.cfg.memoria.max_episodios,
                "procedural": self.cfg.memoria.max_procedimentos,
                "personalidade": self.cfg.memoria.max_personalidade,
            }
            # Achado real ao investigar delay: as 4 chamadas abaixo usam o
            # MESMO texto (plano.consulta_memoria), e cada uma disparava
            # sua PRÓPRIA requisição de embedding pro LM Studio -- 4
            # chamadas de rede por turno só pra gerar o mesmo vetor de
            # novo. Calcula uma vez aqui e reusa nas 4 (ver
            # BancoMemoria.calcular_embedding_consulta/buscar). Rodam em
            # paralelo também: são independentes entre si (só o filtro de
            # tipo muda), o SQLite já serializa a parte de banco pelo lock
            # de BancoMemoria, então o ganho real é não empilhar a parte
            # de CPU/rede de cada uma atrás da outra.
            vetor_plano = self.memoria.calcular_embedding_consulta(plano.consulta_memoria)
            with ThreadPoolExecutor(max_workers=len(limites)) as pool:
                futuros = {
                    tipo: pool.submit(
                        self.memoria.buscar, plano.consulta_memoria, usuario=usuario,
                        tipo=tipo, limite=limite, score_minimo=self.cfg.memoria.score_minimo,
                        vetor_consulta=vetor_plano,
                    )
                    for tipo, limite in limites.items()
                }
                for tipo, futuro in futuros.items():
                    achadas = futuro.result()
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

        # História/lore da própria EVA -- independente de precisa_memoria, de
        # propósito: essa flag decide se a pergunta precisa saber algo sobre A
        # PESSOA, não sobre a EVA. "sobre_si" desativa precisa_memoria e sem
        # este passo a busca de história nunca rodaria exatamente na pergunta
        # que deveria usá-la.
        if mensagem.strip():
            # Mesmo raciocínio do bloco acima: história e regras abaixo
            # usam o MESMO texto (`mensagem`) em dois `buscar` separados --
            # calcula o embedding uma vez e reusa nos dois, em paralelo.
            vetor_mensagem = self.memoria.calcular_embedding_consulta(mensagem)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futuro_historia = pool.submit(
                    self.memoria.buscar, mensagem, usuario=USUARIO_HISTORIA,
                    tipo="semantica", limite=self.cfg.memoria.max_historia,
                    score_minimo=self.cfg.memoria.score_minimo,
                    vetor_consulta=vetor_mensagem,
                )
                # Regras gerais de comportamento dela mesma -- mesmo
                # raciocínio da história acima (independente de
                # precisa_memoria), mas tipo "procedural" em vez de
                # "semantica": são "como agir" e não "fato", então entram
                # no MESMO balde que já existe pra "como agir com essa
                # pessoa" (ver ctx["preferencias"] em context.py) em vez de
                # um rótulo novo -- mistura o "como agir com fulano" (por
                # pessoa) com o "como eu costumo agir" (geral dela), mas os
                # dois já são frases de orientação de comportamento; não há
                # ambiguidade real de leitura para o modelo. Existe pra
                # tirar "regra" específica demais da PERSONA
                # estático (que fica cada vez mais longo e sem filtro de
                # relevância) e mover pra algo buscado só quando bate com o
                # assunto -- ver seed_capacidades_eva.py.
                futuro_regras = pool.submit(
                    self.memoria.buscar, mensagem, usuario=USUARIO_HISTORIA,
                    tipo="procedural", limite=self.cfg.memoria.max_regras_propria,
                    score_minimo=self.cfg.memoria.score_minimo,
                    vetor_consulta=vetor_mensagem,
                )
                historia = futuro_historia.result()
                regras = futuro_regras.result()

            if historia:
                existentes = {m.id for m in saida.get("semantica", [])}
                extras = [m for m in historia if m.id not in existentes]
                if extras:
                    saida["semantica"] = saida.get("semantica", []) + extras

            if regras:
                existentes = {m.id for m in saida.get("procedural", [])}
                extras = [m for m in regras if m.id not in existentes]
                if extras:
                    saida["procedural"] = saida.get("procedural", []) + extras
        return saida

    def _executar_ferramentas(self, plano: Plano) -> dict:
        if not plano.precisa_ferramenta:
            return {}
        chamadas = [
            (f.get("nome"), f.get("args") or {})
            for f in plano.ferramentas if f.get("nome")
        ]
        if not chamadas:
            return {}
        if len(chamadas) == 1:
            nome, args = chamadas[0]
            return {nome: self.ferramentas.executar(nome, **args)}

        # Mais de uma ferramenta no mesmo plano: rodam em paralelo. Eram
        # sequenciais antes (um for simples), e cada ferramenta é uma
        # chamada própria (ex.: busca web no SearXNG pode levar 1-3s+),
        # então dois planos com ferramenta dupla pagavam a SOMA dos dois
        # tempos por padrão -- nenhuma ferramenta depende do resultado de
        # outra dentro do mesmo turno, não há motivo para isso ser serial.
        saida: dict = {}
        with ThreadPoolExecutor(max_workers=len(chamadas)) as pool:
            futuros = {
                pool.submit(self.ferramentas.executar, nome, **args): nome
                for nome, args in chamadas
            }
            for futuro, nome in futuros.items():
                saida[nome] = futuro.result()
        return saida

    def _pos_processar(self, mensagem: str, resposta: str, plano: Plano,
                       usuario: str, sucesso: bool) -> list:
        # histórico -- só registra o par se a EVA respondeu de verdade.
        # Registrar o turno do usuário sozinho deixava entradas "user"
        # seguidas no histórico na próxima mensagem -- ChatML tolera, mas
        # quebra com erro duro em template de alternância estrita
        # (Mistral/Nemo, usado por Violet-Lotus e outros modelos de RP).
        if resposta:
            self.memoria.registrar_turno("user", mensagem, usuario=usuario)
            self.memoria.registrar_turno("assistant", resposta, usuario=usuario)

        reg = self.memoria.pessoa(usuario)
        self.memoria.salvar_pessoa(usuario, turnos=reg["turnos"] + 1)

        # novas memórias -- só as de regra entram aqui, porque são
        # instantâneas (regex, sem chamada de rede) e o Resultado devolvido
        # ao chamador já reflete o que foi guardado. A extração por LLM roda
        # à parte, em segundo plano -- ver _disparar_extracao_llm.
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

        # Extração por LLM: cobre o que a regra não pega (declaração
        # indireta, "estou testando minha IA e o desenvolvimento vai bem"
        # não tem forma fixa nenhuma pra regex casar). Roda em SEGUNDO
        # PLANO -- não faz o usuário esperar uma segunda chamada ao modelo
        # depois que a resposta já saiu. Por isso não entra em `novas`: essa
        # lista alimenta o Resultado que volta pro chamador AGORA, e o que
        # a extração em fundo salvar só existe no banco depois, para o
        # PRÓXIMO turno usar.
        if plano.guardar_memoria and self.cfg.memoria.extrair_com_llm:
            self._disparar_extracao_llm(usuario)

        # Consolidação periódica (memory/consolidacao.py): não a cada
        # turno -- é trabalho pesado (embedding de dezenas de memórias,
        # comparação par a par) que só compensa de tempos em tempos.
        # `reg["turnos"]` já é o contador de turnos DESSA pessoa, então o
        # intervalo é por usuário, não global -- alguém que conversa muito
        # consolida mais vezes que alguém que mal fala, o que é o
        # comportamento certo (mais conversa = mais memória acumulando).
        turnos_novo = reg["turnos"] + 1
        intervalo = self.cfg.memoria.consolidar_a_cada_turnos
        if (self.cfg.memoria.consolidar_com_llm and intervalo > 0
                and turnos_novo % intervalo == 0):
            self._disparar_consolidacao(usuario)

        # Auto-reflexão: a EVA "olhando pra trás" pro que ela mesma acabou
        # de dizer, tentando notar traço específico que apareceu nessa
        # troca. Mesmo gatilho por contagem de turnos que a consolidação
        # acima (intervalo próprio, EVA_AUTORREFLEXAO_INTERVALO) -- não
        # tem por que rodar em todo turno, e reflexão sobre um só turno
        # tende a supergeneralizar de uma amostra pequena demais.
        intervalo_reflexao = self.cfg.memoria.autorreflexao_a_cada_turnos
        if (self.cfg.memoria.autorreflexao_ativa and intervalo_reflexao > 0
                and resposta and turnos_novo % intervalo_reflexao == 0):
            self._disparar_autorreflexao(usuario)

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
            # Números sempre completos, direto do estado bruto -- NÃO
            # para_contexto(). Aquele virou qualitativo e omite estresse
            # perto de 0 de propósito (é pro prompt do modelo, ver
            # state.py); diagnóstico é o painel que você vê, precisa dos
            # quatro números sempre, mesmo que estresse esteja baixo.
            "estado": {
                "energia": round(self.estado.estado.energia, 2),
                "curiosidade": round(self.estado.estado.curiosidade, 2),
                "confianca": round(self.estado.estado.confianca, 2),
                "estresse": round(self.estado.estado.estresse, 2),
            },
            "interacoes": self.estado.estado.total_interacoes,
            "ferramentas": self.ferramentas.nomes(),
            "decisor": "llm" if self.cfg.decisao.usar_llm else "regras",
            "banco": str(self.cfg.memoria.caminho_db),
        }

    async def pesquisar_lacuna(self, consulta: str) -> str | None:
        """Busca em fundo o que uma mensagem pode ter deixado sem resposta
        atualizada, e devolve um resumo pronto para virar impulso de
        iniciativa -- ou None se a busca não trouxe nada útil.

        Quem chama isto é o bridge_client (ou qualquer integração que tenha
        uma Consciencia por perto), a partir de `Resultado.plano.
        possivel_lacuna`. O EVA não conhece Consciencia de propósito -- ele
        continua utilizável sozinho pela CLI, sem call nem iniciativa
        nenhuma. A ponte fica do lado de quem já sabe orquestrar os dois.

        `async def` direto, sem variante sync: pesquisar_lacuna só faz
        sentido vindo de uma integração que já vive num loop (bridge,
        Discord) -- não existe caso de uso pela CLI síncrona.
        """
        try:
            resultado = await asyncio.to_thread(
                self.ferramentas.executar, "buscar", consulta=consulta)
        except Exception as e:
            if self.cfg.debug:
                print(f"[lacuna] erro ao buscar '{consulta[:60]}': {e}")
            return None

        if not isinstance(resultado, dict) or resultado.get("erro"):
            return None

        resumo = resultado.get("resumo")
        if not resumo:
            relacionados = resultado.get("relacionados") or []
            resumo = relacionados[0] if relacionados else None
        if not resumo:
            return None

        # "segundo uma busca" fica explícito de propósito -- você pediu que
        # ela possa citar que pesquisou, em vez de falar como se já soubesse.
        return f"segundo uma busca sobre isso: {resumo}"

    def _disparar_extracao_llm(self, usuario: str) -> None:
        """Roda extrair_por_llm em segundo plano, sem bloquear a resposta.

        Dois caminhos porque `responder` é chamado tanto de dentro de um
        event loop (bridge, Discord) quanto de fora dele (CLI). Detectar
        qual é o certo em vez de assumir um dos dois evita que a CLI quebre
        silenciosamente sem loop, ou que o caminho async bloqueie por usar
        thread manual onde já existe um loop rodando pra fazer isso melhor.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            loop.create_task(self._extrair_e_salvar_async(usuario))
        else:
            threading.Thread(
                target=self._extrair_e_salvar, args=(usuario,), daemon=True
            ).start()

    async def _extrair_e_salvar_async(self, usuario: str) -> None:
        await asyncio.to_thread(self._extrair_e_salvar, usuario)

    def _extrair_e_salvar(self, usuario: str) -> None:
        """O trabalho de verdade: busca histórico recente, pede fatos ao
        LLM, grava o que vier. Roda fora do caminho principal -- exceção
        aqui NUNCA deve derrubar a conversa, só fica no log.

        `self.memoria` é seguro entre threads (BancoMemoria tem RLock), mas
        se a EVA for fechada (`fechar()`) enquanto isso ainda roda, o
        SQLite pode já estar fechado -- daí o except genérico no fim: uma
        falha aqui é invisível para o usuário por natureza (a resposta já
        foi entregue), então não vale derrubar nada, só avisar no log.

        O log do resultado (sucesso, vazio ou erro) SEMPRE aparece, não só
        com EVA_DEBUG=1 -- esse era o motivo real de "a EVA quase não
        guarda nada" parecer misterioso: se o modelo extrator falhasse por
        qualquer motivo (schema mal seguido, modelo trocado, timeout), não
        existia sintoma nenhum, nem no console. Sem essa visibilidade,
        qualquer ajuste na extração é tiro no escuro.
        """
        try:
            historico = self.memoria.historico(usuario=usuario, limite=8)
            fatos = extrair_por_llm(historico, self.llm_extrator)
            for item in fatos:
                self.memoria.adicionar(
                    tipo=item["tipo"], conteudo=item["conteudo"],
                    usuario=usuario, fonte=item["fonte"],
                    confianca=item["confianca"],
                )
            if fatos:
                print(f"[memoria-llm] {usuario}: {[f['conteudo'] for f in fatos]}")
            elif self.cfg.debug:
                print(f"[memoria-llm] {usuario}: nada extraído neste turno")
        except Exception as e:
            print(f"[memoria-llm] erro ao extrair para {usuario}: {e}")

    def _disparar_consolidacao(self, usuario: str) -> None:
        """Mesmo padrão de _disparar_extracao_llm: thread solta no
        caminho síncrono, task no caminho async. Consolidação é ainda mais
        best-effort que extração -- se falhar, o pior caso é a memória
        continuar crescendo sem resumir, não uma perda de dado.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            loop.create_task(self._consolidar_async(usuario))
        else:
            threading.Thread(
                target=self._consolidar, args=(usuario,), daemon=True
            ).start()

    async def _consolidar_async(self, usuario: str) -> None:
        await asyncio.to_thread(self._consolidar, usuario)

    def _consolidar(self, usuario: str) -> None:
        """O trabalho de verdade: agrupa memórias antigas similares em
        resumos. Ver memory/consolidacao.py para o algoritmo.

        Sem `self.memoria.embeddings` configurado (LM Studio sem o modelo
        de embedding, ou EVA_EMBEDDINGS=0), `candidatos_para_consolidar`
        sempre devolve vazio -- não há vetor pra comparar -- então isso
        roda sem erro e sem efeito, sem precisar de um if extra aqui.
        """
        try:
            from .memory.consolidacao import consolidar_usuario
            resumos = consolidar_usuario(
                self.memoria, usuario=usuario,
                dias_minimo=self.cfg.memoria.consolidar_dias_minimo,
            )
            if resumos and self.cfg.debug:
                print(f"[consolidacao] {usuario}: {len(resumos)} resumo(s) -> {resumos}")
        except Exception as e:
            if self.cfg.debug:
                print(f"[consolidacao] erro para {usuario}: {e}")

    def _disparar_autorreflexao(self, usuario: str) -> None:
        """Mesmo padrão dual sync/async de _disparar_extracao_llm --
        thread solta no caminho síncrono, task no caminho async.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            loop.create_task(self._autorrefletir_async(usuario))
        else:
            threading.Thread(
                target=self._autorrefletir, args=(usuario,), daemon=True
            ).start()

    async def _autorrefletir_async(self, usuario: str) -> None:
        await asyncio.to_thread(self._autorrefletir, usuario)

    def _autorrefletir(self, usuario: str) -> None:
        """O trabalho de verdade: pega o histórico recente, pede pro
        extrator (llm_extrator -- mesmo cliente da extração de fatos,
        aponta pro que EVA_MEMORIA_LLM_MODEL/EVA_DECISION_MODEL configurar,
        tipicamente o modelo pequeno/decisor, ex. minicpm-v-4.6) uma
        observação sobre a PRÓPRIA EVA, e grava como história/lore dela --
        mesmo usuário reservado e mesmo tipo que seed_historia_eva.py, só
        com fonte='auto_reflexao' e confiança mais baixa (ver
        extrair_personalidade_propria em memory/extractor.py) pra ficar
        auditável e distinguível do que foi escrito à mão.

        Roda fora do caminho principal, mesma política de _extrair_e_salvar:
        exceção aqui nunca derruba a conversa, só fica no log.
        """
        try:
            historico = self.memoria.historico(usuario=usuario, limite=8)
            observacoes = extrair_personalidade_propria(historico, self.llm_extrator)
            for obs in observacoes:
                id_ = self.memoria.adicionar(
                    tipo=obs["tipo"], conteudo=obs["conteudo"],
                    usuario=USUARIO_HISTORIA, fonte=obs["fonte"],
                    confianca=obs["confianca"],
                )
                if id_:
                    print(f"[autorreflexao] nova observação: {obs['conteudo']}")
            if not observacoes and self.cfg.debug:
                print(f"[autorreflexao] nada específico neste lote (usuario={usuario})")
        except Exception as e:
            print(f"[autorreflexao] erro: {e}")

    def fechar(self) -> None:
        self.estado.salvar()
        self.memoria.fechar()