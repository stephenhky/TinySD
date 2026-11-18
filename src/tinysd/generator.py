"""High-level, stateful image generator."""

import os
from os import PathLike
from pathlib import Path
from typing import Union, Literal, Optional

from .generation import generate_image
from .pipeline import get_stable_diffusion_pipeline
from .schemas import StableDiffusionOutput


class SDImageGenerator:
    """Load a pipeline once and generate images repeatedly with it.

    Example:
        >>> gen = SDImageGenerator("stabilityai/stable-diffusion-xl-base-1.0", None, None)
        >>> out = gen.generate_images("a (cat:1.2) on a mat", "blurry", seed=42, nbimages=2)
        >>> out.images[0].save("cat.png")
    """

    def __init__(
            self,
            base_model_id: Union[Literal["runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"], str],
            lora_weight_path: Optional[Union[PathLike, str]],
            vae_weight_path: Optional[Union[PathLike, str]],
            text_encoder_id: Optional[str] = None,
            tokenizer_id: Optional[str] = None,
            cuda: bool = True,
            text_inversion_embedding_path: Optional[Union[PathLike, str]] = None,
            scheduler_name: str = "Euler",
    ):
        """Build the underlying pipeline.

        Args:
            base_model_id: Hugging Face model ID or local checkpoint path.
            lora_weight_path: Optional LoRA weights path (file, directory or repo ID).
            vae_weight_path: Optional custom VAE path (file, directory or repo ID).
            text_encoder_id: Optional custom text encoder ID (requires ``tokenizer_id``).
            tokenizer_id: Optional custom tokenizer ID (requires ``text_encoder_id``).
            cuda: Use CUDA if available.
            text_inversion_embedding_path: Optional textual inversion embedding path.
            scheduler_name: A scheduler name from :data:`tinysd.schedulers.SCHEDULERS`.

        See :func:`tinysd.pipeline.get_stable_diffusion_pipeline` for details.
        """
        self._base_model_id = base_model_id
        if isinstance(lora_weight_path, Path):
            self._lora_file_name = lora_weight_path.name
            lora_weight_path = lora_weight_path.as_posix()
        elif isinstance(lora_weight_path, str):
            self._lora_file_name = os.path.basename(lora_weight_path)
        else:
            self._lora_file_name = None
        if isinstance(vae_weight_path, str):
            vae_weight_path = Path(vae_weight_path)
            self._vae_file_name = vae_weight_path.name
        elif isinstance(vae_weight_path, Path):
            self._vae_file_name = vae_weight_path.name
        else:
            self._vae_file_name = None

        self._pipe = get_stable_diffusion_pipeline(
            self._base_model_id,
            lora_weight_path,
            vae_weight_path,
            text_encoder_id=text_encoder_id,
            tokenizer_id=tokenizer_id,
            cuda=cuda,
            text_inversion_embedding_path=text_inversion_embedding_path,
            scheduler_name=scheduler_name,
        )

    def generate_images(
            self,
            positive_prompt: str,
            negative_prompt: str,
            height: int = 512,
            width: int = 512,
            steps: int = 30,
            guidance_scale: float = 7.5,
            seed: Optional[int] = None,
            nbimages: int = 1,
            clip_skip: int = 1,
    ) -> StableDiffusionOutput:
        """Generate ``nbimages`` images, one pipeline call per image.

        Args:
            positive_prompt: Prompt (A1111 or Compel syntax).
            negative_prompt: Negative prompt (A1111 or Compel syntax).
            height: Image height in pixels.
            width: Image width in pixels.
            steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Base seed; image ``i`` uses ``seed + i``. ``None`` is random.
            nbimages: Number of images to generate.
            clip_skip: CLIP skip (SD 1.x only); ``>= 2`` uses the penultimate layer.

        Returns:
            A :class:`StableDiffusionOutput` with the images and generation settings.
        """
        images = [
            generate_image(
                positive_prompt,
                negative_prompt,
                self._pipe,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=(seed + i if seed is not None else None),
                clip_skip=clip_skip,
            )[0]
            for i in range(nbimages)
        ]
        return StableDiffusionOutput(
            base_model_id=self._base_model_id,
            lora_file_name=self._lora_file_name,
            vae_file_name=self._vae_file_name,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            images=images
        )
