"""Internal helpers shared across TinySD modules."""

from diffusers import StableDiffusionXLPipeline


def _is_sdxl_pipeline(pipe) -> bool:
    """Return whether ``pipe`` is a Stable Diffusion XL pipeline.

    A pipeline is considered SDXL if it is a :class:`StableDiffusionXLPipeline`
    instance, or if it has both a second tokenizer and a second text encoder.

    Args:
        pipe: A diffusers pipeline (or duck-typed equivalent).

    Returns:
        ``True`` if the pipeline is SDXL-like, ``False`` otherwise.
    """
    return (
        isinstance(pipe, StableDiffusionXLPipeline)
        or (getattr(pipe, "tokenizer_2", None) is not None and getattr(pipe, "text_encoder_2", None) is not None)
    )


def _ensure_text_model_attr(text_encoder) -> None:
    """Ensure compatibility with transformers >= 5.6 where CLIPTextModel flattened `.text_model`.

    If ``text_encoder`` lacks a ``text_model`` attribute, a property returning the
    encoder itself is added to its class, so Compel can keep accessing
    ``text_encoder.text_model``.

    Args:
        text_encoder: A text encoder module, or ``None`` (no-op).
    """
    if text_encoder is not None and not hasattr(text_encoder, "text_model"):
        setattr(type(text_encoder), "text_model", property(lambda self: self))
