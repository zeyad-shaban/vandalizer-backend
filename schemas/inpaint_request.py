from typing import Literal

from pydantic import BaseModel, Field


class InpaintRequest(BaseModel):
    mode: Literal["blur", "remove", "diffusion"] = "blur"
    positive_prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = Field(default=4, ge=4, le=15)
