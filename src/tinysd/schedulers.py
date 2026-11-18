"""Sampler/scheduler registry using AUTOMATIC1111-style names."""

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    UniPCMultistepScheduler,
)

#: Mapping of A1111 sampler names to ``(diffusers scheduler class, extra config kwargs)``.
SCHEDULERS = {
    "Euler a": (EulerAncestralDiscreteScheduler, {}),
    "Euler": (EulerDiscreteScheduler, {}),
    "DPM++ 2M": (DPMSolverMultistepScheduler, {}),
    "DPM++ 2M Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "DPM++ 2M SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Karras": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "DPM++ SDE": (DPMSolverSinglestepScheduler, {}),
    "DPM++ SDE Karras": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
    "Heun": (HeunDiscreteScheduler, {}),
    "KDPM2": (KDPM2DiscreteScheduler, {}),
    "KDPM2 a": (KDPM2AncestralDiscreteScheduler, {}),
    "DDIM": (DDIMScheduler, {}),
    "UniPC": (UniPCMultistepScheduler, {}),
}


def set_scheduler(pipe, scheduler_name: str = "Euler"):
    """Replace the scheduler of a diffusers pipeline in place.

    The new scheduler is built from the pipeline's current scheduler config, so
    model-specific settings (e.g. beta schedule) are preserved.

    Args:
        pipe: A diffusers pipeline exposing a ``scheduler`` attribute.
        scheduler_name: A key of :data:`SCHEDULERS`, e.g. ``"DPM++ 2M Karras"``.

    Returns:
        The same pipeline object, for chaining.

    Raises:
        ValueError: If ``scheduler_name`` is not a known scheduler.
    """
    if scheduler_name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'. Available: {list(SCHEDULERS.keys())}")
    scheduler_cls, kwargs = SCHEDULERS[scheduler_name]
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
    return pipe
