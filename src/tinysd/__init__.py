"""TinySD: a small wrapper around diffusers for Stable Diffusion 1.x and SDXL.

Modules:
    - :mod:`tinysd.prompt`: A1111 -> Compel prompt syntax conversion.
    - :mod:`tinysd.schedulers`: A1111-named scheduler registry.
    - :mod:`tinysd.pipeline`: Pipeline loading (VAE, LoRA, textual inversion).
    - :mod:`tinysd.generation`: Low-level image generation.
    - :mod:`tinysd.generator`: High-level :class:`SDImageGenerator`.
    - :mod:`tinysd.schemas`: Output data containers.
"""

from .generation import generate_image
from .generator import SDImageGenerator
from .pipeline import MODEL_ID_ALIASES, get_stable_diffusion_pipeline
from .prompt import convert_a1111_to_compel
from .schedulers import SCHEDULERS, set_scheduler
from .schemas import StableDiffusionOutput

__version__ = "0.1.0"

__all__ = [
    "MODEL_ID_ALIASES",
    "SCHEDULERS",
    "SDImageGenerator",
    "StableDiffusionOutput",
    "convert_a1111_to_compel",
    "generate_image",
    "get_stable_diffusion_pipeline",
    "set_scheduler",
    "__version__",
]
