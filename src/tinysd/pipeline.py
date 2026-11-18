"""Construction of Stable Diffusion / SDXL pipelines with optional VAE, LoRA and embeddings."""

import os
from os import PathLike
from typing import Union, Literal, Optional
import warnings

import torch
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from .schedulers import set_scheduler
from .utils import _ensure_text_model_attr

#: Deprecated model IDs mapped to their current Hugging Face Hub location.
MODEL_ID_ALIASES = {
    "runwayml/stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}


def get_stable_diffusion_pipeline(
        base_model_id: Union[Literal["runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"], str],
        lora_weights_path: Optional[Union[PathLike, str]],
        vae_input_path: Optional[Union[PathLike, str]],
        cuda: bool = True,     # quite impractical without CUDA
        text_encoder_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        text_inversion_embedding_path: Optional[Union[PathLike, str]] = None,
        scheduler_name: str = "Euler",
) -> Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    """Load a Stable Diffusion (1.x) or SDXL pipeline and apply optional add-ons.

    The pipeline class is chosen by whether ``"xl"`` appears in the model ID. Single
    ``.safetensors``/``.ckpt`` files are loaded with ``from_single_file``; anything
    else with ``from_pretrained``. The safety checker is disabled for SD 1.x.

    Args:
        base_model_id: Hugging Face model ID, local diffusers directory, or local
            single-file checkpoint. Deprecated IDs in :data:`MODEL_ID_ALIASES` are
            resolved automatically.
        lora_weights_path: Path to a LoRA weights file, a directory containing
            ``pytorch_lora_weights.safetensors``, or a Hub repo ID. ``None`` or an
            empty string skips LoRA loading.
        vae_input_path: Path to a single-file VAE, or a diffusers VAE directory/repo
            ID. ``None`` or an empty string keeps the built-in VAE.
        cuda: Use CUDA with float16 if available; otherwise CPU with float32.
        text_encoder_id: Optional custom text encoder ID. Must be given together
            with ``tokenizer_id``.
        tokenizer_id: Optional custom tokenizer ID. Must be given together with
            ``text_encoder_id``.
        text_inversion_embedding_path: Optional textual inversion embedding, loaded
            under the token ``ng_deepnegative_v1_75t``.
        scheduler_name: A scheduler name from :data:`tinysd.schedulers.SCHEDULERS`.

    Returns:
        The configured :class:`StableDiffusionPipeline` or
        :class:`StableDiffusionXLPipeline`.

    Raises:
        ValueError: If ``scheduler_name`` is unknown.

    Warns:
        UserWarning: If only one of ``text_encoder_id`` / ``tokenizer_id`` is given;
            both are then ignored.
    """
    resolved_model_id = str(base_model_id).strip()
    if resolved_model_id in MODEL_ID_ALIASES and not os.path.exists(resolved_model_id):
        resolved_model_id = MODEL_ID_ALIASES[resolved_model_id]

    use_cuda = bool(cuda and torch.cuda.is_available())
    dtype = torch.float16 if use_cuda else torch.float32

    is_xl = "xl" in resolved_model_id.lower()
    pipeline_cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

    extra_kwargs = {}
    if not is_xl:
        extra_kwargs["safety_checker"] = None  # Disable for freedom

    if text_encoder_id is not None and tokenizer_id is not None:
        text_encoder = AutoModel.from_pretrained(text_encoder_id, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        extra_kwargs["text_encoder"] = text_encoder
        extra_kwargs["tokenizer"] = tokenizer
    elif (text_encoder_id is not None) ^ (tokenizer_id is not None):
        warnings.warn("You must give `text_encoder_id` and `tokenizer_id` a value or both None! Now it is assumed both are None!")

    if os.path.isfile(resolved_model_id) and resolved_model_id.endswith((".safetensors", ".ckpt")):
        pipe = pipeline_cls.from_single_file(
            resolved_model_id,
            torch_dtype=dtype,
            **extra_kwargs,
        )
    else:
        pipe = pipeline_cls.from_pretrained(
            resolved_model_id,
            torch_dtype=dtype,
            **extra_kwargs,
        )

    if use_cuda:
        pipe = pipe.to("cuda")

    set_scheduler(pipe, scheduler_name)

    if vae_input_path is not None:
        vae_path_str = vae_input_path if isinstance(vae_input_path, str) else vae_input_path.as_posix()
        if len(vae_path_str) > 0:
            if os.path.isfile(vae_path_str):
                try:
                    pipe.vae = AutoencoderKL.from_single_file(vae_path_str, torch_dtype=dtype)
                except Exception:
                    vae_weights = load_file(vae_path_str)
                    pipe.vae.load_state_dict(vae_weights, strict=False)
            else:
                pipe.vae = AutoencoderKL.from_pretrained(vae_path_str, torch_dtype=dtype)
            if use_cuda:
                pipe.vae = pipe.vae.to("cuda", dtype=dtype)

    if lora_weights_path is not None:
        lora_path_str = lora_weights_path if isinstance(lora_weights_path, str) else lora_weights_path.as_posix()
        if len(lora_path_str) > 0:
            if os.path.isfile(lora_path_str):
                pipe.load_lora_weights(
                    os.path.dirname(lora_path_str) or ".",
                    weight_name=os.path.basename(lora_path_str),
                )
            elif os.path.isdir(lora_path_str) and os.path.isfile(
                os.path.join(lora_path_str, "pytorch_lora_weights.safetensors")
            ):
                pipe.load_lora_weights(lora_path_str, weight_name="pytorch_lora_weights.safetensors")
            else:
                pipe.load_lora_weights(lora_path_str)

    if text_inversion_embedding_path is not None:
        pipe.load_textual_inversion(text_inversion_embedding_path, token='ng_deepnegative_v1_75t')

    _ensure_text_model_attr(getattr(pipe, "text_encoder", None))
    _ensure_text_model_attr(getattr(pipe, "text_encoder_2", None))
    return pipe
