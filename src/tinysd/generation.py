"""Low-level image generation with Compel prompt weighting."""

from typing import Union, Optional

import PIL.Image
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline

from .prompt import convert_a1111_to_compel
from .utils import _ensure_text_model_attr, _is_sdxl_pipeline


def generate_image(
        positive_prompt: Union[str, list[str]],
        negative_prompt: Union[str, list[str]],
        stable_diffusion_pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
        height: int = 512,
        width: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        clip_skip: int = 1,
        convert_a1111_syntax: bool = True,
) -> list[PIL.Image.Image]:
    """Generate images from weighted prompts using a loaded pipeline.

    Prompts are encoded with Compel (supporting long prompts and weighting), padded
    to the same length, and passed to the pipeline as embeddings.

    Args:
        positive_prompt: Prompt or list of prompts (one image per prompt).
        negative_prompt: Negative prompt or list of negative prompts.
        stable_diffusion_pipeline: A pipeline from
            :func:`tinysd.pipeline.get_stable_diffusion_pipeline`.
        height: Image height in pixels.
        width: Image width in pixels.
        steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed. For a batch, prompt ``i`` uses ``seed + i``.
            ``None`` means non-deterministic.
        clip_skip: For SD 1.x, values ``>= 2`` use the penultimate CLIP hidden
            states. Ignored for SDXL, which always uses the penultimate layer.
        convert_a1111_syntax: Convert A1111 prompt syntax to Compel syntax first.

    Returns:
        The list of generated PIL images.
    """
    if convert_a1111_syntax:
        if isinstance(positive_prompt, list):
            positive_prompt = [convert_a1111_to_compel(p) for p in positive_prompt]
        else:
            positive_prompt = convert_a1111_to_compel(positive_prompt)

        if isinstance(negative_prompt, list):
            negative_prompt = [convert_a1111_to_compel(p) for p in negative_prompt]
        else:
            negative_prompt = convert_a1111_to_compel(negative_prompt)

    batch_size = len(positive_prompt) if isinstance(positive_prompt, list) else 1
    if seed is None:
        generator = None
    elif batch_size > 1:
        generator = [torch.Generator("cpu").manual_seed(int(seed) + i) for i in range(batch_size)]
    else:
        generator = torch.Generator("cpu").manual_seed(int(seed))

    _ensure_text_model_attr(getattr(stable_diffusion_pipeline, "text_encoder", None))
    _ensure_text_model_attr(getattr(stable_diffusion_pipeline, "text_encoder_2", None))
    is_xl = _is_sdxl_pipeline(stable_diffusion_pipeline)

    if is_xl:
        compel = Compel(
            tokenizer=[stable_diffusion_pipeline.tokenizer, stable_diffusion_pipeline.tokenizer_2],
            text_encoder=[stable_diffusion_pipeline.text_encoder, stable_diffusion_pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        positive_prompt_embeds, positive_pooled = compel(positive_prompt)
        negative_prompt_embeds, negative_pooled = compel(negative_prompt)
        [positive_prompt_embeds, negative_prompt_embeds] = compel.pad_conditioning_tensors_to_same_length(
            [positive_prompt_embeds, negative_prompt_embeds]
        )
        sd_output = stable_diffusion_pipeline(
            prompt_embeds=positive_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=positive_pooled,
            negative_pooled_prompt_embeds=negative_pooled,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )
    else:
        emb_type = (
            ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
            if clip_skip and clip_skip >= 2
            else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
        )
        compel = Compel(
            tokenizer=stable_diffusion_pipeline.tokenizer,
            text_encoder=stable_diffusion_pipeline.text_encoder,
            returned_embeddings_type=emb_type,
            truncate_long_prompts=False,
        )
        positive_prompt_embeds = compel(positive_prompt)
        negative_prompt_embeds = compel(negative_prompt)
        [positive_prompt_embeds, negative_prompt_embeds] = compel.pad_conditioning_tensors_to_same_length(
            [positive_prompt_embeds, negative_prompt_embeds]
        )

        sd_output = stable_diffusion_pipeline(
            prompt_embeds=positive_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

    return sd_output.images
