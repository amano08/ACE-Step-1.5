"""Tests for cover-noise initialisation parity with the PyTorch DiT path.

The reference is ``generate_audio`` in
``acestep/models/xl_sft/modeling_acestep_v15_xl_base.py``:

    effective_noise_level = 1.0 - cover_noise_strength
    nearest_t = min(t[:-1], key=lambda x: abs(x - effective_noise_level))
    start_idx = t[:-1].index(nearest_t)
    xt = nearest_t * noise + (1 - nearest_t) * src_latents
    t = t[start_idx:]
    infer_steps = len(t) - 1

``get_timestep_schedule`` returns the descending schedule without the trailing
zero, so it is PyTorch's ``t[:-1]`` and ``len(t) - 1`` becomes
``len(schedule[start_idx:])``.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from acestep.models.mlx.dit_generate import (
    get_timestep_schedule,
    mlx_generate_diffusion,
)

SHIFT = 3.0
INFER_STEPS = 8
SHAPE = (1, 4, 4)
SEED = 42


def _zero_velocity_decoder():
    """Decoder predicting v=0, so ``xt`` never moves and the output is ``xt_init``."""
    import mlx.core as mx

    def _forward(hidden_states, timestep, timestep_r,
                 encoder_hidden_states, context_latents,
                 cache=None, use_cache=False):
        return mx.zeros_like(hidden_states), cache

    decoder = MagicMock()
    decoder.side_effect = _forward
    return decoder


def _seeded_noise(shape, seed):
    """Reproduce the noise draw performed inside ``mlx_generate_diffusion``."""
    import mlx.core as mx

    return np.array(mx.random.normal(shape, key=mx.random.key(int(seed))))


def _expected_pytorch(cover_noise_strength):
    """Return (nearest_t, remaining_steps) using the PyTorch reference formula."""
    schedule = get_timestep_schedule(SHIFT, None, infer_steps=INFER_STEPS)
    effective = 1.0 - cover_noise_strength
    nearest = min(schedule, key=lambda x: abs(x - effective))
    return nearest, len(schedule) - schedule.index(nearest)


def _run(decoder, cover_noise_strength, src_latents_np, **overrides):
    kwargs = dict(
        mlx_decoder=decoder,
        encoder_hidden_states_np=np.zeros((1, 2, 4), dtype=np.float32),
        context_latents_np=np.zeros(SHAPE, dtype=np.float32),
        src_latents_shape=SHAPE,
        seed=SEED,
        infer_method="ode",
        sampler_mode="euler",
        shift=SHIFT,
        infer_steps=INFER_STEPS,
        cover_noise_strength=cover_noise_strength,
        src_latents_np=src_latents_np,
        dcw_enabled=False,
        disable_tqdm=True,
    )
    kwargs.update(overrides)
    return mlx_generate_diffusion(**kwargs)


class CoverNoiseScheduleTests(unittest.TestCase):
    """The truncated schedule must match the PyTorch step count exactly."""

    def test_step_count_matches_pytorch_for_each_strength(self):
        src = np.ones(SHAPE, dtype=np.float32)
        for cns in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            with self.subTest(cover_noise_strength=cns):
                decoder = _zero_velocity_decoder()
                _run(decoder, cns, src)
                _, expected_steps = _expected_pytorch(cns)
                self.assertEqual(decoder.call_count, expected_steps)

    def test_higher_strength_means_fewer_steps(self):
        src = np.ones(SHAPE, dtype=np.float32)
        counts = []
        for cns in (0.1, 0.5, 0.9):
            decoder = _zero_velocity_decoder()
            _run(decoder, cns, src)
            counts.append(decoder.call_count)
        self.assertEqual(counts, sorted(counts, reverse=True))


class CoverNoiseInitialStateTests(unittest.TestCase):
    """``xt`` must start at ``nearest_t * noise + (1 - nearest_t) * src``."""

    def test_initial_state_matches_renoise(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=SHAPE).astype(np.float32)
        noise = _seeded_noise(SHAPE, SEED)
        for cns in (0.25, 0.5, 0.9):
            with self.subTest(cover_noise_strength=cns):
                result = _run(_zero_velocity_decoder(), cns, src)
                nearest_t, _ = _expected_pytorch(cns)
                expected = nearest_t * noise + (1.0 - nearest_t) * src
                np.testing.assert_allclose(
                    result["target_latents"], expected, rtol=1e-5, atol=1e-5
                )

    def test_stronger_cover_noise_stays_closer_to_source(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=SHAPE).astype(np.float32)
        distances = []
        for cns in (0.25, 0.5, 0.9):
            result = _run(_zero_velocity_decoder(), cns, src)
            distances.append(float(np.linalg.norm(result["target_latents"] - src)))
        self.assertEqual(distances, sorted(distances, reverse=True))


class CoverNoiseDisabledTests(unittest.TestCase):
    """Zero strength and missing source must preserve the pure-noise behaviour."""

    def test_zero_strength_starts_from_pure_noise(self):
        decoder = _zero_velocity_decoder()
        result = _run(decoder, 0.0, np.ones(SHAPE, dtype=np.float32))
        np.testing.assert_allclose(
            result["target_latents"], _seeded_noise(SHAPE, SEED), rtol=1e-5, atol=1e-5
        )
        self.assertEqual(decoder.call_count, INFER_STEPS)

    def test_missing_src_latents_falls_back_to_pure_noise(self):
        decoder = _zero_velocity_decoder()
        result = _run(decoder, 0.9, None)
        np.testing.assert_allclose(
            result["target_latents"], _seeded_noise(SHAPE, SEED), rtol=1e-5, atol=1e-5
        )
        self.assertEqual(decoder.call_count, INFER_STEPS)


if __name__ == "__main__":
    unittest.main()
