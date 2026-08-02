"""EVA -- arquitetura cognitiva conversacional."""
from .orchestrator import EVA, Resultado
from .config import EVAConfig, carregar_config

__version__ = "0.1.0"
__all__ = ["EVA", "Resultado", "EVAConfig", "carregar_config"]
