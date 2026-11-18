"""Data containers returned by the TinySD public API."""

from dataclasses import dataclass, field
from typing import Optional

import PIL.Image


@dataclass
class StableDiffusionOutput:
    """Result of an image generation request, together with its generation settings.

    Attributes:
        base_model_id: Hugging Face model ID or local path of the base checkpoint.
        lora_file_name: File name of the LoRA weights used, or ``None``.
        vae_file_name: File name of the custom VAE used, or ``None``.
        positive_prompt: The positive prompt used for generation.
        negative_prompt: The negative prompt used for generation.
        height: Image height in pixels.
        width: Image width in pixels.
        steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Base random seed, or ``None`` for non-deterministic generation.
        images: The generated PIL images.
    """

    base_model_id: str
    lora_file_name: Optional[str]
    vae_file_name: Optional[str]
    positive_prompt: str
    negative_prompt: str
    height: int
    width: int
    steps: int
    guidance_scale: float
    seed: Optional[int]
    images: list[PIL.Image.Image] = field(default_factory=list)
