import unittest
from pathlib import Path
from unittest import mock

from tinysd import generator as gmod
from tinysd.schemas import StableDiffusionOutput


class TestSDImageGenerator(unittest.TestCase):
    def setUp(self):
        p1 = mock.patch.object(gmod, "get_stable_diffusion_pipeline")
        p2 = mock.patch.object(gmod, "generate_image")
        self.get_pipe = p1.start()
        self.gen_image = p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.gen_image.side_effect = lambda *a, **kw: [f"img-seed-{kw['seed']}"]

    def test_init_forwards_arguments(self):
        g = gmod.SDImageGenerator(
            "model", Path("/x/lora.safetensors"), "/y/vae.safetensors",
            text_encoder_id="e", tokenizer_id="t", cuda=False,
            text_inversion_embedding_path="emb.pt", scheduler_name="DDIM",
        )
        args, kwargs = self.get_pipe.call_args
        self.assertEqual(args, ("model", "/x/lora.safetensors", Path("/y/vae.safetensors")))
        self.assertEqual(
            kwargs,
            dict(text_encoder_id="e", tokenizer_id="t", cuda=False,
                 text_inversion_embedding_path="emb.pt", scheduler_name="DDIM"),
        )
        self.assertEqual(g._lora_file_name, "lora.safetensors")
        self.assertEqual(g._vae_file_name, "vae.safetensors")
        self.assertIs(g._pipe, self.get_pipe.return_value)

    def test_init_string_lora_and_path_vae(self):
        g = gmod.SDImageGenerator("model", "/x/lora.pt", Path("/y/vae.pt"))
        self.assertEqual(g._lora_file_name, "lora.pt")
        self.assertEqual(g._vae_file_name, "vae.pt")

    def test_init_none_paths(self):
        g = gmod.SDImageGenerator("model", None, None)
        self.assertIsNone(g._lora_file_name)
        self.assertIsNone(g._vae_file_name)

    def test_generate_images_with_seed(self):
        g = gmod.SDImageGenerator("model", "/x/lora.pt", None)
        out = g.generate_images("pos", "neg", height=64, width=128, steps=4, guidance_scale=5.0,
                                seed=7, nbimages=3, clip_skip=2)
        self.assertIsInstance(out, StableDiffusionOutput)
        self.assertEqual(out.images, ["img-seed-7", "img-seed-8", "img-seed-9"])
        self.assertEqual(self.gen_image.call_count, 3)
        args, kwargs = self.gen_image.call_args
        self.assertEqual(args, ("pos", "neg", g._pipe))
        self.assertEqual(kwargs["clip_skip"], 2)
        self.assertEqual(
            (out.base_model_id, out.lora_file_name, out.vae_file_name, out.positive_prompt,
             out.negative_prompt, out.height, out.width, out.steps, out.guidance_scale, out.seed),
            ("model", "lora.pt", None, "pos", "neg", 64, 128, 4, 5.0, 7),
        )

    def test_generate_images_without_seed(self):
        g = gmod.SDImageGenerator("model", None, None)
        out = g.generate_images("pos", "neg", nbimages=2)
        self.assertEqual(out.images, ["img-seed-None", "img-seed-None"])
        self.assertIsNone(out.seed)


class TestPublicApi(unittest.TestCase):
    def test_package_exports(self):
        import tinysd
        for name in tinysd.__all__:
            self.assertTrue(hasattr(tinysd, name), name)

    def test_legacy_sd_module_reexports(self):
        from tinysd import sd
        from tinysd.generator import SDImageGenerator
        from tinysd.prompt import convert_a1111_to_compel
        self.assertIs(sd.SDImageGenerator, SDImageGenerator)
        self.assertIs(sd.convert_a1111_to_compel, convert_a1111_to_compel)
        for name in ("get_stable_diffusion_pipeline", "generate_image", "set_scheduler",
                     "SCHEDULERS", "MODEL_ID_ALIASES", "StableDiffusionOutput",
                     "_is_sdxl_pipeline", "_ensure_text_model_attr", "_convert_a1111_segment_to_compel"):
            self.assertTrue(hasattr(sd, name), name)


if __name__ == "__main__":
    unittest.main()
