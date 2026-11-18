import types
import unittest
from unittest import mock

from diffusers import StableDiffusionXLPipeline

from tinysd.utils import _ensure_text_model_attr, _is_sdxl_pipeline


class TestIsSdxlPipeline(unittest.TestCase):
    def test_sdxl_instance(self):
        pipe = mock.MagicMock(spec=StableDiffusionXLPipeline)
        self.assertTrue(_is_sdxl_pipeline(pipe))

    def test_duck_typed_sdxl(self):
        pipe = types.SimpleNamespace(tokenizer_2=object(), text_encoder_2=object())
        self.assertTrue(_is_sdxl_pipeline(pipe))

    def test_missing_second_encoder(self):
        self.assertFalse(_is_sdxl_pipeline(types.SimpleNamespace(tokenizer_2=object(), text_encoder_2=None)))
        self.assertFalse(_is_sdxl_pipeline(types.SimpleNamespace(tokenizer_2=None, text_encoder_2=object())))
        self.assertFalse(_is_sdxl_pipeline(types.SimpleNamespace()))


class TestEnsureTextModelAttr(unittest.TestCase):
    def test_none_is_noop(self):
        _ensure_text_model_attr(None)  # should not raise

    def test_adds_self_property(self):
        class Encoder:
            pass

        enc = Encoder()
        _ensure_text_model_attr(enc)
        self.assertIs(enc.text_model, enc)

    def test_existing_attr_kept(self):
        class Encoder:
            def __init__(self):
                self.text_model = "inner"

        enc = Encoder()
        _ensure_text_model_attr(enc)
        self.assertEqual(enc.text_model, "inner")


if __name__ == "__main__":
    unittest.main()
