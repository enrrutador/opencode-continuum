"""Secrets abstraction.

Source of truth on Kaggle: kaggle_secrets.UserSecretsClient.
Fallback: environment variables.
Secrets are never written to disk in plain text, never logged,
and never included in shell command arguments.
"""

from __future__ import annotations

import os
from typing import Optional


class SecretsManager:
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError

    def get_required(self, key: str) -> str:
        val = self.get(key)
        if val is None or val == "":
            raise RuntimeError(f"Required secret '{key}' is not available.")
        return val


class EnvSecrets(SecretsManager):
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key, default)


class KaggleSecrets(SecretsManager):
    """Read secrets from Kaggle Secrets via UserSecretsClient."""

    def __init__(self):
        self._client = None
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            self._client = UserSecretsClient()
        except Exception:
            self._client = None

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        if self._client is not None:
            try:
                return self._client.get_secret(key)
            except Exception:
                pass
        return os.environ.get(key, default)


def get_secrets_manager(runtime: Optional[str] = None) -> SecretsManager:
    if runtime is None:
        from .runtime import detect_runtime

        runtime = detect_runtime()
    if runtime == "kaggle":
        return KaggleSecrets()
    return EnvSecrets()


def load_secret(key: str, runtime: Optional[str] = None) -> Optional[str]:
    return get_secrets_manager(runtime).get(key)


def load_required_secret(key: str, runtime: Optional[str] = None) -> str:
    return get_secrets_manager(runtime).get_required(key)
