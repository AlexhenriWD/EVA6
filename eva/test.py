"""
diagnostico_lm_studio.py

Script isolado pra descobrir onde o pipeline está travando. Roda 3 testes
em sequência, cada um com timeout curto e prints explícitos — se travar,
você vê exatamente em qual teste, sem esperar 60s+ no escuro.

Uso:
    python3 diagnostico_lm_studio.py
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

LM_STUDIO_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
CEREBRO_MODEL = os.getenv("EVA_CEREBRO_MODEL") or os.getenv("MINICPM_MODEL") or "minicpm-v-4.6"
PERSONALIDADE_MODEL = os.getenv("EVA_PERSONALIDADE_MODEL", "rocinante-12b-v1.1")


async def teste_1_conectividade():
    print("\n=== TESTE 1: LM Studio está respondendo? ===")
    import requests
    t0 = time.time()
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
        elapsed = time.time() - t0
        print(f"✅ Respondeu em {elapsed:.2f}s, status={resp.status_code}")
        modelos = [m["id"] for m in resp.json().get("data", [])]
        print(f"   Modelos carregados agora: {modelos}")
        print(f"   Cérebro esperado: {CEREBRO_MODEL!r} — {'PRESENTE' if CEREBRO_MODEL in modelos else '!!! AUSENTE !!!'}")
        print(f"   Personalidade esperado: {PERSONALIDADE_MODEL!r} — {'PRESENTE' if PERSONALIDADE_MODEL in modelos else '!!! AUSENTE !!!'}")
        return modelos
    except Exception as e:
        print(f"❌ FALHOU: {e}")
        return []


async def teste_2_chat_personalidade(modelos_carregados):
    print("\n=== TESTE 2: Chamada de chat real pro modelo de Personalidade ===")
    if PERSONALIDADE_MODEL not in modelos_carregados:
        print(f"⚠️  Pulando — {PERSONALIDADE_MODEL!r} não está entre os modelos carregados no LM Studio agora.")
        print("   Isso sozinho já explicaria silêncio total: o LM Studio provavelmente")
        print("   devolve erro 404/'model not found', e dependendo de como isso é")
        print("   tratado, pode não gerar traceback visível.")
        return

    import requests
    t0 = time.time()
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": PERSONALIDADE_MODEL,
                "messages": [{"role": "user", "content": "responda só a palavra: oi"}],
                "max_tokens": 20,
            },
            timeout=30,
        )
        elapsed = time.time() - t0
        print(f"✅ Respondeu em {elapsed:.2f}s, status={resp.status_code}")
        if resp.status_code == 200:
            print(f"   Conteúdo: {resp.json()['choices'][0]['message']['content']!r}")
        else:
            print(f"   Corpo do erro: {resp.text[:300]}")
    except requests.exceptions.Timeout:
        print(f"❌ TIMEOUT após 30s — o modelo está carregando, travado, ou VRAM insuficiente")
    except Exception as e:
        print(f"❌ FALHOU: {e}")


async def teste_3_pipeline_completo():
    print("\n=== TESTE 3: Pipeline completo (Cerebro + Personalidade) com timeout de 40s ===")
    try:
        from cerebro import Cerebro
        from personalidade import Personalidade
        from eva_brain import EvaBrain
        from lm_studio_client import LMStudioClient
    except ImportError as e:
        print(f"❌ Import falhou: {e} — rode este script na mesma pasta dos outros arquivos (core/)")
        return

    lm_cerebro = LMStudioClient(model_name=CEREBRO_MODEL, base_url=LM_STUDIO_URL)
    lm_personalidade = LMStudioClient(model_name=PERSONALIDADE_MODEL, base_url=LM_STUDIO_URL)

    cerebro = Cerebro(client=lm_cerebro, planner=None)  # planner=None pra isolar: sem tools
    personalidade = Personalidade(client=lm_personalidade, biblia_path="biblia_condensada.md", memory=None)
    eva_brain = EvaBrain(cerebro=cerebro, personalidade=personalidade, memory_system=None)

    t0 = time.time()
    try:
        resultado = await asyncio.wait_for(
            eva_brain.process_message(user_id="teste_diagnostico", message="oi, tudo bem?", context_type="text"),
            timeout=40.0,
        )
        elapsed = time.time() - t0
        print(f"✅ Respondeu em {elapsed:.2f}s")
        print(f"   Resposta: {resultado.get('response')!r}")
        print(f"   Metadata: {resultado.get('metadata')}")
    except asyncio.TimeoutError:
        print(f"❌ TRAVOU — não respondeu em 40s. Isso confirma o sintoma que você está vendo.")
    except Exception as e:
        import traceback
        print(f"❌ EXCEÇÃO (isso pelo menos dá um traceback, ao contrário do bot real):")
        traceback.print_exc()


async def main():
    print(f"LM_STUDIO_BASE_URL = {LM_STUDIO_URL}")
    print(f"EVA_CEREBRO_MODEL = {CEREBRO_MODEL}")
    print(f"EVA_PERSONALIDADE_MODEL = {PERSONALIDADE_MODEL}")

    modelos = await teste_1_conectividade()
    await teste_2_chat_personalidade(modelos)
    await teste_3_pipeline_completo()

    print("\n=== FIM DO DIAGNÓSTICO ===")


if __name__ == "__main__":
    asyncio.run(main())
