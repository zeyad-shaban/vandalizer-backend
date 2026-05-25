from typing import Literal

from pydantic import BaseModel, Field


class InpaintRequest(BaseModel):
    mode: Literal["blur", "remove", "diffusion"] = "blur"
    positive_prompt: str = ""
    strength: float = Field(default=0.7, ge=0.0, le=1.0)
    num_inference_steps: int = Field(default=2, ge=1, le=4)
