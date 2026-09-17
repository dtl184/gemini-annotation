"""Environment-derived configuration. No config files - just env vars and CLI flags."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_ANNOTATOR_URL = "http://127.0.0.1:5111"


@dataclass
class Config:
    gemini_api_key: str
    gemini_model: str = DEFAULT_MODEL
    annotator_url: str = DEFAULT_ANNOTATOR_URL


def load_config(*, model: str | None = None, annotator_url: str | None = None) -> Config:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment."
        )
    return Config(
        gemini_api_key=api_key,
        gemini_model=model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        annotator_url=annotator_url or os.environ.get("ANNOTATOR_URL", DEFAULT_ANNOTATOR_URL),
    )
