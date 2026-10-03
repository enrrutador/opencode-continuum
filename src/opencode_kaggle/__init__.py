"""opencode_kaggle — Kaggle adapters for OpenCode Continuum.

Nota: no re-exportamos `bootstrap` como función porque sombrea al
submódulo `opencode_kaggle.bootstrap` y rompe `import ... as bs`.
Usá siempre: from opencode_kaggle.bootstrap import bootstrap
"""

__version__ = "5.0.0"

__all__ = ["__version__"]
