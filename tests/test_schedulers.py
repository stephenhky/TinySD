import types
import unittest

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)

from tinysd.schedulers import SCHEDULERS, set_scheduler


def _make_pipe():
    return types.SimpleNamespace(scheduler=DDIMScheduler())


class TestSchedulers(unittest.TestCase):
    def test_registry_entries_are_valid(self):
        for name, (cls, kwargs) in SCHEDULERS.items():
            with self.subTest(name=name):
                self.assertTrue(hasattr(cls, "from_config"))
                self.assertIsInstance(kwargs, dict)

    def test_all_schedulers_can_be_built(self):
        for name, (cls, _) in SCHEDULERS.items():
            with self.subTest(name=name):
                pipe = set_scheduler(_make_pipe(), name)
                self.assertIsInstance(pipe.scheduler, cls)

    def test_default_is_euler(self):
        pipe = set_scheduler(_make_pipe())
        self.assertIs(type(pipe.scheduler), EulerDiscreteScheduler)

    def test_returns_same_pipe(self):
        pipe = _make_pipe()
        self.assertIs(set_scheduler(pipe, "Euler a"), pipe)
        self.assertIsInstance(pipe.scheduler, EulerAncestralDiscreteScheduler)

    def test_kwargs_applied(self):
        pipe = set_scheduler(_make_pipe(), "DPM++ 2M SDE Karras")
        self.assertIsInstance(pipe.scheduler, DPMSolverMultistepScheduler)
        self.assertTrue(pipe.scheduler.config.use_karras_sigmas)
        self.assertEqual(pipe.scheduler.config.algorithm_type, "sde-dpmsolver++")

    def test_config_preserved(self):
        pipe = types.SimpleNamespace(scheduler=DDIMScheduler(beta_start=0.001, num_train_timesteps=500))
        set_scheduler(pipe, "Euler")
        self.assertEqual(pipe.scheduler.config.beta_start, 0.001)
        self.assertEqual(pipe.scheduler.config.num_train_timesteps, 500)

    def test_unknown_scheduler_raises(self):
        with self.assertRaises(ValueError) as ctx:
            set_scheduler(_make_pipe(), "Nope")
        self.assertIn("Nope", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
