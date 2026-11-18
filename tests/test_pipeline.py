import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch

from tinysd import pipeline as pl


class TestGetStableDiffusionPipeline(unittest.TestCase):
    def setUp(self):
        patches = {
            "sd_cls": mock.patch.object(pl, "StableDiffusionPipeline"),
            "xl_cls": mock.patch.object(pl, "StableDiffusionXLPipeline"),
            "set_scheduler": mock.patch.object(pl, "set_scheduler"),
            "vae_cls": mock.patch.object(pl, "AutoencoderKL"),
            "load_file": mock.patch.object(pl, "load_file"),
            "auto_model": mock.patch.object(pl, "AutoModel"),
            "auto_tok": mock.patch.object(pl, "AutoTokenizer"),
            "cuda_avail": mock.patch.object(torch.cuda, "is_available", return_value=False),
        }
        self.m = {k: p.start() for k, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)
        self.sd_pipe = self.m["sd_cls"].from_pretrained.return_value
        self.xl_pipe = self.m["xl_cls"].from_pretrained.return_value
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _touch(self, name):
        path = os.path.join(self.tmp.name, name)
        Path(path).touch()
        return path

    def test_sd15_from_pretrained_cpu(self):
        pipe = pl.get_stable_diffusion_pipeline("some/sd-model", None, None)
        self.assertIs(pipe, self.sd_pipe)
        self.m["sd_cls"].from_pretrained.assert_called_once_with(
            "some/sd-model", torch_dtype=torch.float32, safety_checker=None
        )
        self.m["set_scheduler"].assert_called_once_with(self.sd_pipe, "Euler")
        self.sd_pipe.to.assert_not_called()

    def test_alias_resolved_and_stripped(self):
        pl.get_stable_diffusion_pipeline("  runwayml/stable-diffusion-v1-5 ", None, None)
        args, _ = self.m["sd_cls"].from_pretrained.call_args
        self.assertEqual(args[0], "stable-diffusion-v1-5/stable-diffusion-v1-5")

    def test_xl_selected_without_safety_checker(self):
        pipe = pl.get_stable_diffusion_pipeline("stabilityai/stable-diffusion-xl-base-1.0", None, None)
        self.assertIs(pipe, self.xl_pipe)
        _, kwargs = self.m["xl_cls"].from_pretrained.call_args
        self.assertNotIn("safety_checker", kwargs)

    def test_single_file_checkpoint(self):
        ckpt = self._touch("model.safetensors")
        pipe = pl.get_stable_diffusion_pipeline(ckpt, None, None)
        self.m["sd_cls"].from_single_file.assert_called_once()
        self.m["sd_cls"].from_pretrained.assert_not_called()
        self.assertIs(pipe, self.m["sd_cls"].from_single_file.return_value)

    def test_cuda_moves_pipe_and_uses_fp16(self):
        self.m["cuda_avail"].return_value = True
        moved = self.sd_pipe.to.return_value
        pipe = pl.get_stable_diffusion_pipeline("some/sd-model", None, None, cuda=True)
        _, kwargs = self.m["sd_cls"].from_pretrained.call_args
        self.assertEqual(kwargs["torch_dtype"], torch.float16)
        self.sd_pipe.to.assert_called_once_with("cuda")
        self.assertIs(pipe, moved)

    def test_cuda_flag_false_stays_on_cpu(self):
        self.m["cuda_avail"].return_value = True
        pl.get_stable_diffusion_pipeline("some/sd-model", None, None, cuda=False)
        self.sd_pipe.to.assert_not_called()

    def test_custom_scheduler_name_forwarded(self):
        pl.get_stable_diffusion_pipeline("some/sd-model", None, None, scheduler_name="DDIM")
        self.m["set_scheduler"].assert_called_once_with(self.sd_pipe, "DDIM")

    def test_text_encoder_and_tokenizer(self):
        pl.get_stable_diffusion_pipeline("some/sd-model", None, None, text_encoder_id="enc", tokenizer_id="tok")
        self.m["auto_model"].from_pretrained.assert_called_once_with("enc", torch_dtype=torch.float32)
        self.m["auto_tok"].from_pretrained.assert_called_once_with("tok")
        _, kwargs = self.m["sd_cls"].from_pretrained.call_args
        self.assertIs(kwargs["text_encoder"], self.m["auto_model"].from_pretrained.return_value)
        self.assertIs(kwargs["tokenizer"], self.m["auto_tok"].from_pretrained.return_value)

    def test_only_one_of_encoder_tokenizer_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pl.get_stable_diffusion_pipeline("some/sd-model", None, None, text_encoder_id="enc")
        self.assertTrue(any("tokenizer_id" in str(w.message) for w in caught))
        self.m["auto_model"].from_pretrained.assert_not_called()
        _, kwargs = self.m["sd_cls"].from_pretrained.call_args
        self.assertNotIn("text_encoder", kwargs)

    def test_vae_single_file(self):
        vae = self._touch("vae.safetensors")
        pipe = pl.get_stable_diffusion_pipeline("some/sd-model", None, Path(vae))
        self.m["vae_cls"].from_single_file.assert_called_once_with(vae, torch_dtype=torch.float32)
        self.assertIs(pipe.vae, self.m["vae_cls"].from_single_file.return_value)

    def test_vae_single_file_fallback_to_state_dict(self):
        vae = self._touch("vae.safetensors")
        self.m["vae_cls"].from_single_file.side_effect = RuntimeError("bad")
        original_vae = self.sd_pipe.vae
        pl.get_stable_diffusion_pipeline("some/sd-model", None, vae)
        self.m["load_file"].assert_called_once_with(vae)
        original_vae.load_state_dict.assert_called_once_with(self.m["load_file"].return_value, strict=False)

    def test_vae_pretrained(self):
        pipe = pl.get_stable_diffusion_pipeline("some/sd-model", None, "some/vae-repo")
        self.m["vae_cls"].from_pretrained.assert_called_once_with("some/vae-repo", torch_dtype=torch.float32)
        self.assertIs(pipe.vae, self.m["vae_cls"].from_pretrained.return_value)

    def test_empty_vae_string_skipped(self):
        pl.get_stable_diffusion_pipeline("some/sd-model", None, "")
        self.m["vae_cls"].from_pretrained.assert_not_called()
        self.m["vae_cls"].from_single_file.assert_not_called()

    def test_lora_file(self):
        lora = self._touch("my_lora.safetensors")
        pl.get_stable_diffusion_pipeline("some/sd-model", Path(lora), None)
        self.sd_pipe.load_lora_weights.assert_called_once_with(self.tmp.name, weight_name="my_lora.safetensors")

    def test_lora_dir_with_default_weights(self):
        self._touch("pytorch_lora_weights.safetensors")
        pl.get_stable_diffusion_pipeline("some/sd-model", self.tmp.name, None)
        self.sd_pipe.load_lora_weights.assert_called_once_with(
            self.tmp.name, weight_name="pytorch_lora_weights.safetensors"
        )

    def test_lora_repo_id(self):
        pl.get_stable_diffusion_pipeline("some/sd-model", "user/lora-repo", None)
        self.sd_pipe.load_lora_weights.assert_called_once_with("user/lora-repo")

    def test_empty_lora_string_skipped(self):
        pl.get_stable_diffusion_pipeline("some/sd-model", "", None)
        self.sd_pipe.load_lora_weights.assert_not_called()

    def test_textual_inversion(self):
        pl.get_stable_diffusion_pipeline("some/sd-model", None, None, text_inversion_embedding_path="emb.pt")
        self.sd_pipe.load_textual_inversion.assert_called_once_with("emb.pt", token="ng_deepnegative_v1_75t")

    def test_no_textual_inversion_by_default(self):
        pl.get_stable_diffusion_pipeline("some/sd-model", None, None)
        self.sd_pipe.load_textual_inversion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
