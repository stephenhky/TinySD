# TinySD

TinySD is a small Python wrapper around [🤗 diffusers](https://github.com/huggingface/diffusers) for
**Stable Diffusion 1.x** and **SDXL**. It lets you generate images with a few lines of code while keeping
the conveniences of the AUTOMATIC1111 (A1111) WebUI:

- **A1111 prompt syntax**: `(word:1.3)`, `[word]`, `((word))`, `BREAK`, and `<lora:...>` tags are converted
  automatically to [Compel](https://github.com/damian0815/compel) syntax. Long prompts are not truncated.
- **A1111 sampler names**: `"Euler a"`, `"DPM++ 2M Karras"`, `"DPM++ 2M SDE Karras"`, `"UniPC"`, and others.
- **Flexible model loading**: Hugging Face Hub IDs, local diffusers folders, or single `.safetensors` / `.ckpt`
  checkpoints (e.g. from Civitai).
- **Add-ons**: custom VAE, LoRA weights, textual inversion embeddings, custom text encoder/tokenizer.
- **CLIP skip** for SD 1.x models, and **reproducible seeds**.

## Installation

Requires Python ≥ 3.9. A CUDA GPU is strongly recommended; TinySD falls back to CPU (float32) otherwise.

```bash
git clone <this-repo> TinySD
cd TinySD
pip install -e .
```

If you need a specific CUDA build of PyTorch, install it first (see [pytorch.org](https://pytorch.org)).

## Quick start

```python
from tinysd import SDImageGenerator

generator = SDImageGenerator(
    base_model_id="stabilityai/stable-diffusion-xl-base-1.0",
    lora_weight_path=None,
    vae_weight_path=None,
    scheduler_name="DPM++ 2M Karras",
)

result = generator.generate_images(
    positive_prompt="a (fluffy cat:1.3) sitting on a windowsill, golden hour, ((highly detailed))",
    negative_prompt="blurry, [low quality], watermark",
    height=1024,
    width=1024,
    steps=30,
    guidance_scale=7.0,
    seed=42,
    nbimages=2,
)

for i, image in enumerate(result.images):
    image.save(f"cat_{i}.png")

print(result.seed, result.steps, result.base_model_id)
```

`generate_images` returns a `StableDiffusionOutput` dataclass that holds the generated PIL images and the
settings used to produce them (model, LoRA/VAE file names, prompts, size, steps, guidance scale, seed).
When `seed` is given, image `i` uses `seed + i`, so results are reproducible.

## Usage examples

### Stable Diffusion 1.5 with LoRA, VAE and CLIP skip

```python
from pathlib import Path
from tinysd import SDImageGenerator

generator = SDImageGenerator(
    base_model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",  # "runwayml/..." is aliased automatically
    lora_weight_path=Path("loras/my_style.safetensors"),          # file, folder, or Hub repo ID
    vae_weight_path="vae/vae-ft-mse-840000-ema-pruned.safetensors",
    scheduler_name="Euler a",
)

result = generator.generate_images(
    positive_prompt="portrait of a knight, (intricate armor:1.2), BREAK dramatic lighting, castle background",
    negative_prompt="(worst quality:1.4), lowres, bad anatomy",
    seed=1234,
    clip_skip=2,
)
result.images[0].save("knight.png")
```

### Local single-file checkpoint and textual inversion

```python
from tinysd import SDImageGenerator

generator = SDImageGenerator(
    base_model_id="checkpoints/myModel_v10.safetensors",  # "xl" in the name selects the SDXL pipeline
    lora_weight_path=None,
    vae_weight_path=None,
    text_inversion_embedding_path="embeddings/ng_deepnegative_v1_75t.pt",
)

result = generator.generate_images(
    positive_prompt="a cozy cabin in a snowy forest",
    negative_prompt="ng_deepnegative_v1_75t, blurry",  # embedding is registered under this token
)
```

### Lower-level API

If you want to manage the pipeline yourself, for example to change the scheduler between runs or generate
a batch in one call:

```python
from tinysd import get_stable_diffusion_pipeline, generate_image, set_scheduler

pipe = get_stable_diffusion_pipeline(
    "stabilityai/stable-diffusion-xl-base-1.0",
    lora_weights_path=None,
    vae_input_path=None,
    cuda=True,
)

set_scheduler(pipe, "UniPC")

images = generate_image(
    positive_prompt=["a red apple", "a green pear"],   # one image per prompt
    negative_prompt=["blurry", "blurry"],
    stable_diffusion_pipeline=pipe,
    height=1024,
    width=1024,
    steps=25,
    seed=7,                                            # prompt i uses seed + i
)
```

### Prompt conversion only

```python
from tinysd import convert_a1111_to_compel

convert_a1111_to_compel("a (cat:1.3) on a [mat]")
# 'a (cat)1.3 on a (mat)-'

convert_a1111_to_compel("a ((red)) car <lora:style:0.8>")
# 'a ((red)+)+ car'

convert_a1111_to_compel("a cat BREAK a dog")
# '("a cat", "a dog").and()'
```

Pass `convert_a1111_syntax=False` to `generate_image` if your prompts are already written in Compel syntax.

### Available schedulers

```python
from tinysd import SCHEDULERS
print(list(SCHEDULERS))
# ['Euler a', 'Euler', 'DPM++ 2M', 'DPM++ 2M Karras', 'DPM++ 2M SDE', 'DPM++ 2M SDE Karras',
#  'DPM++ SDE', 'DPM++ SDE Karras', 'Heun', 'KDPM2', 'KDPM2 a', 'DDIM', 'UniPC']
```

## Package layout

```
src/tinysd/
├── __init__.py     # public API
├── prompt.py       # A1111 -> Compel prompt conversion
├── schedulers.py   # SCHEDULERS registry and set_scheduler()
├── pipeline.py     # get_stable_diffusion_pipeline(): model, VAE, LoRA, embeddings
├── generation.py   # generate_image(): Compel encoding + inference
├── generator.py    # SDImageGenerator high-level class
├── schemas.py      # StableDiffusionOutput dataclass
├── utils.py        # internal helpers
└── sd.py           # backward-compatible re-exports
```

## Running tests

The unit tests mock all model loading, so they run quickly and need neither a GPU nor network access:

```bash
PYTHONPATH=src python -m unittest discover -s tests -t .
# or, with pytest installed:
pytest
```

## Notes

- The safety checker is disabled for SD 1.x pipelines.
- The SD vs. SDXL choice is based on whether `"xl"` appears in the model ID or path.
- `clip_skip` applies only to SD 1.x. SDXL always uses the penultimate hidden states.
