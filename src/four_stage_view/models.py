"""Configuration contains environment variable names, never API credentials."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class ImageConfig(StrictModel):
    """Image model Codex generates the texture with (Codex's imagegen skill CLI, OpenAI Images API compatible)."""

    model: str = Field(min_length=1)
    api_key_env: str = Field(default="AGILE_IMAGE_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")
    base_url: str | None = None

    @model_validator(mode="after")
    def credential_free_endpoint(self):
        if self.base_url:
            url = urlsplit(self.base_url)
            if (url.scheme not in ("https", "http") or not url.hostname or url.username
                    or url.password or url.query or url.fragment):
                raise ValueError("base_url must be an HTTP(S) endpoint without credentials or query parameters")
        return self


class AgentConfig(StrictModel):
    """Codex CLI: video analysis critic (step 1) and texture generator (step 2); uses the CLI's own provider."""

    executable: str = "codex"
    model: str | None = None
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max", "ultra"] | None = None
    timeout_seconds: int = Field(default=600, ge=30, le=3600)
    review_context: Literal["full", "images_only"] = "images_only"


class TextureConfig(StrictModel):
    """Step 2: Codex generates the texture of the fitted object mesh."""

    size: int = Field(default=2048, ge=256, le=4096)                 # texture.png is size x size
    max_image_calls: int = Field(default=8, ge=1, le=50)             # budget given to Codex
    timeout_seconds: int = Field(default=3600, ge=60, le=4 * 3600)  # one Codex session


class EgoConfig(StrictModel):
    image: ImageConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    texture: TextureConfig = Field(default_factory=TextureConfig)
