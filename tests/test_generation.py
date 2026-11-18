import types
import unittest
from unittest import mock

import torch
from compel import ReturnedEmbeddingsType

from tinysd import generation as gen


def _sd_pipe():
    pipe = mock.MagicMock()
    pipe.tokenizer_2 = None
    pipe.text_encoder_2 = None
    pipe.return_value = types.SimpleNamespace(images=["img"])
    return pipe


def _xl_pipe():
    pipe = mock.MagicMock()
    pipe.return_value = types.SimpleNamespace(images=["img"])
    return pipe


class TestGenerateImage(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(gen, "Compel")
        self.compel_cls = patcher.start()
        self.addCleanup(patcher.stop)
        self.compel = self.compel_cls.return_value
        self.compel.pad_conditioning_tensors_to_same_length.side_effect = lambda tensors: tensors

    def test_sd_path(self):
        self.compel.side_effect = lambda p: f"emb[{p}]"
        pipe = _sd_pipe()
        images = gen.generate_image("(cat:1.2)", "[ugly]", pipe, height=64, width=32, steps=5, guidance_scale=3.0)
        self.assertEqual(images, ["img"])
        _, ckw = self.compel_cls.call_args
        self.assertEqual(ckw["returned_embeddings_type"], ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED)
        self.assertFalse(ckw["truncate_long_prompts"])
        _, kw = pipe.call_args
        self.assertEqual(kw["prompt_embeds"], "emb[(cat)1.2]")
        self.assertEqual(kw["negative_prompt_embeds"], "emb[(ugly)-]")
        self.assertEqual((kw["height"], kw["width"], kw["num_inference_steps"]), (64, 32, 5))
        self.assertEqual(kw["guidance_scale"], 3.0)
        self.assertIsNone(kw["generator"])
        self.assertNotIn("pooled_prompt_embeds", kw)

    def test_clip_skip_uses_penultimate(self):
        gen.generate_image("a", "b", _sd_pipe(), clip_skip=2)
        _, ckw = self.compel_cls.call_args
        self.assertEqual(ckw["returned_embeddings_type"], ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED)

    def test_conversion_can_be_disabled(self):
        self.compel.side_effect = lambda p: p
        pipe = _sd_pipe()
        gen.generate_image("(cat:1.2)", "[ugly]", pipe, convert_a1111_syntax=False)
        _, kw = pipe.call_args
        self.assertEqual(kw["prompt_embeds"], "(cat:1.2)")
        self.assertEqual(kw["negative_prompt_embeds"], "[ugly]")

    def test_list_prompts_converted(self):
        self.compel.side_effect = lambda p: p
        pipe = _sd_pipe()
        gen.generate_image(["(a:2)", "[b]"], ["(c:0.5)", "d"], pipe)
        _, kw = pipe.call_args
        self.assertEqual(kw["prompt_embeds"], ["(a)2", "(b)-"])
        self.assertEqual(kw["negative_prompt_embeds"], ["(c)0.5", "d"])

    def test_seed_single(self):
        pipe = _sd_pipe()
        gen.generate_image("a", "b", pipe, seed=123)
        g = pipe.call_args.kwargs["generator"]
        self.assertIsInstance(g, torch.Generator)
        self.assertEqual(g.initial_seed(), 123)

    def test_seed_batch(self):
        pipe = _sd_pipe()
        gen.generate_image(["a", "b", "c"], ["n", "n", "n"], pipe, seed=10)
        gens = pipe.call_args.kwargs["generator"]
        self.assertEqual([g.initial_seed() for g in gens], [10, 11, 12])

    def test_sdxl_path(self):
        self.compel.side_effect = lambda p: (f"emb[{p}]", f"pool[{p}]")
        pipe = _xl_pipe()
        images = gen.generate_image("a", "b", pipe)
        self.assertEqual(images, ["img"])
        _, ckw = self.compel_cls.call_args
        self.assertEqual(ckw["tokenizer"], [pipe.tokenizer, pipe.tokenizer_2])
        self.assertEqual(ckw["text_encoder"], [pipe.text_encoder, pipe.text_encoder_2])
        self.assertEqual(ckw["requires_pooled"], [False, True])
        self.assertEqual(
            ckw["returned_embeddings_type"], ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
        )
        _, kw = pipe.call_args
        self.assertEqual(kw["prompt_embeds"], "emb[a]")
        self.assertEqual(kw["pooled_prompt_embeds"], "pool[a]")
        self.assertEqual(kw["negative_prompt_embeds"], "emb[b]")
        self.assertEqual(kw["negative_pooled_prompt_embeds"], "pool[b]")
        self.assertEqual(kw["output_type"], "pil")


if __name__ == "__main__":
    unittest.main()
