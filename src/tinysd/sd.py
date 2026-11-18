"""Backward-compatible facade.

The implementation now lives in dedicated modules; this module re-exports
everything that used to be defined here so ``from tinysd.sd import ...`` keeps working.
"""

from .generation import generate_image
from .generator import SDImageGenerator
from .pipeline import MODEL_ID_ALIASES, get_stable_diffusion_pipeline
from .prompt import _convert_a1111_segment_to_compel, convert_a1111_to_compel
from .schedulers import SCHEDULERS, set_scheduler
from .schemas import StableDiffusionOutput
from .utils import _ensure_text_model_attr, _is_sdxl_pipeline

__all__ = [
    "MODEL_ID_ALIASES",
    "SCHEDULERS",
    "SDImageGenerator",
    "StableDiffusionOutput",
    "convert_a1111_to_compel",
    "generate_image",
    "get_stable_diffusion_pipeline",
    "set_scheduler",
]
