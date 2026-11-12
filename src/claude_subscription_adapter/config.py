import json
import os
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODELS = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9001
    api_key: str | None = None
    models: dict[str, str] = field(default_factory=lambda: DEFAULT_MODELS.copy())

    def validate(self):
        try:
            local = ip_address(self.host).is_loopback
        except ValueError:
            local = self.host == "localhost"
        if not local and not self.api_key:
            raise ValueError("CSA_API_KEY is required for non-loopback binding")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not isinstance(self.models, dict):
            # This is operator-supplied configuration; cli.main presents every invalid setting uniformly.
            raise ValueError("CSA_MODELS must be a JSON object mapping non-empty string aliases to model names")  # noqa: TRY004
        for alias, model in self.models.items():
            if not isinstance(alias, str) or not alias.strip() or not isinstance(model, str) or not model.strip():
                raise ValueError("CSA_MODELS must map non-empty string aliases to non-empty string model names")

    @classmethod
    def from_env(cls):
        # Local configuration is convenient for development, while real environment variables remain authoritative.
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        raw_models = os.getenv("CSA_MODELS", json.dumps(DEFAULT_MODELS))
        try:
            models = json.loads(raw_models)
        except json.JSONDecodeError as exc:
            raise ValueError("CSA_MODELS must be valid JSON object mapping aliases to model names") from exc
        try:
            port = int(os.getenv("CSA_PORT", "9001"))
        except ValueError as exc:
            raise ValueError("CSA_PORT must be an integer") from exc

        config = cls(
            host=os.getenv("CSA_HOST", "127.0.0.1"),
            port=port,
            api_key=os.getenv("CSA_API_KEY"),
            models=models,
        )
        config.validate()
        return config
