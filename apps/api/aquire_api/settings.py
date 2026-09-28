"""Environment-backed API settings with no import-time side effects."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    bundle_root: Path
    allow_uncalibrated: bool = False
    max_upload_bytes: int = 5 * 1024 * 1024
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)

    @classmethod
    def from_env(cls) -> "Settings":
        root = Path(os.environ.get("AQUIRE_BUNDLE_ROOT", "prototype_bundle/current"))
        allow = os.environ.get("AQUIRE_ALLOW_UNCALIBRATED", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        origins = tuple(
            value.strip()
            for value in os.environ.get("AQUIRE_CORS_ORIGINS", "http://localhost:5173").split(",")
            if value.strip()
        )
        return cls(bundle_root=root, allow_uncalibrated=allow, cors_origins=origins)
