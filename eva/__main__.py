"""Ponto de entrada de `python -m eva`.

O carregamento do .env vive em eva/config.py, não aqui -- veja o
comentário lá. Este arquivo só existe para o `python -m eva` funcionar.
"""
from .cli import main
import sys

sys.exit(main())