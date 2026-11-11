import os
from dataclasses import dataclass, field
from ipaddress import ip_address


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 9001
    api_key: str | None = None
    models: dict[str, str] = field(default_factory=dict)

    def validate(self):
        try:
            local = ip_address(self.host).is_loopback
        except ValueError:
            local = self.host == "localhost"
        if not local and not self.api_key:
            raise ValueError("CSA_API_KEY is required for non-loopback binding")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")

    @classmethod
    def from_env(cls):
        import json

        return cls(
            host=os.getenv("CSA_HOST", "127.0.0.1"),
            port=int(os.getenv("CSA_PORT", "9001")),
            api_key=os.getenv("CSA_API_KEY"),
            models=json.loads(os.getenv("CSA_MODELS", "{}")),
        )
