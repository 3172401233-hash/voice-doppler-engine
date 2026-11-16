import math
import unittest

from doppler import DopplerEngine, constant_velocity, piecewise_velocity

SR = 8000
C = 343.0
FREQ = 440.0


def sine(n, freq=FREQ, sr=SR, amp=1.0):
    return [amp * math.sin(2.0 * math.pi * freq * i / sr) for i in range(n)]


def run_blocks(engine, pcm, block_size):
    """Feed pcm in blocks; return (output, per-block pitch, per-block delay)."""
    out = []
    pitches = []
    delays = []
    for i in range(0, len(pcm), block_size):
        block, pitch = engine.process(pcm[i:i + block_size])
        out.extend(block)
        pitches.append(pitch)
        delays.append(engine.delay)
    return out, pitches, delays


def measure_freq(samples, sr=SR):
    first = last = None
    crossings = 0
    for i in range(1, len(samples)):
        if samples[i - 1] < 0.0 <= samples[i]:
            crossings += 1
            if first is None:
                first = i
            last = i
    if crossings < 2:
        return 0.0
    return (crossings - 1) * sr / (last - first)


def assert_all_finite(test, samples):
    test.assertTrue(all(math.isfinite(x) for x in samples), "non-finite output")


class TestApproach(unittest.TestCase):
    """Source closing at constant 50 m/s from 200 m away."""

    def setUp(self):
        self.v = 50.0
        self.engine = DopplerEngine(constant_velocity((200.0, 0.0, 0.0),
                                                      (-self.v, 0.0, 0.0)),
                                    speed_of_sound=C, sample_rate=SR)
        self.out, self.pitches, self.delays = run_blocks(
            self.engine, sine(2 * SR), 100)

    def test_pitch_factor_matches_theory(self):
        expected = C / (C - self.v)  # ~1.1706
        self.assertAlmostEqual(self.pitches[-1], expected, delta=0.01 * expected)

    def test_output_frequency_shifted_up(self):
        steady = self.out[SR:2 * SR - 100]
        measured = measure_freq(steady)
        expected = FREQ * C / (C - self.v)
        self.assertAlmostEqual(measured, expected, delta=0.03 * expected)

    def test_delay_decreases_monotonically(self):
        # Retarded delay: D = (200 + v*D)/C, i.e. D = 200/(C-v).
        self.assertAlmostEqual(self.delays[0], 200.0 / (C - self.v), delta=0.01)
        for a, b in zip(self.delays, self.delays[1:]):
            self.assertLess(b, a)

    def test_closing_speed(self):
        self.assertAlmostEqual(self.engine.closing_speed(), self.v, delta=0.5)


class TestRecede(unittest.TestCase):
    """Source receding at constant 40 m/s."""

    def test_pitch_factor_below_unity(self):
        v = 40.0
        engine = DopplerEngine(constant_velocity((50.0, 0.0, 0.0), (v, 0.0, 0.0)),
                               speed_of_sound=C, sample_rate=SR)
        _, pitches, delays = run_blocks(engine, sine(SR), 100)
        expected = C / (C + v)  # ~0.8956
        self.assertAlmostEqual(pitches[-1], expected, delta=0.01 * expected)
        self.assertLess(pitches[-1], 1.0)
        for a, b in zip(delays, delays[1:]):
            self.assertGreater(b, a)  # delay grows while receding


class TestFlyby(unittest.TestCase):
    """Source passing the listener at 40 m/s, closest distance 10 m."""

    def setUp(self):
        self.engine = DopplerEngine(constant_velocity((-100.0, 10.0, 0.0),
                                                      (40.0, 0.0, 0.0)),
                                    speed_of_sound=C, sample_rate=SR)
        self.out, self.pitches, self.delays = run_blocks(
            self.engine, sine(5 * SR), 64)

    def test_pitch_sweeps_down_through_unity(self):
        self.assertGreater(self.pitches[0], 1.05)
        self.assertLess(self.pitches[-1], 0.95)
        below = [i for i, p in enumerate(self.pitches) if p < 1.0]
        self.assertTrue(below)
        first_below = below[0]
        self.assertTrue(all(p < 1.0 for p in self.pitches[first_below:]),
                        "pitch should cross 1.0 exactly once")

    def test_pitch_and_delay_are_continuous(self):
        for a, b in zip(self.pitches, self.pitches[1:]):
            self.assertAlmostEqual(a, b, delta=0.02)
        for a, b in zip(self.delays, self.delays[1:]):
            self.assertAlmostEqual(a, b, delta=0.005)

    def test_output_waveform_continuous(self):
        assert_all_finite(self, self.out)
        bound = 2.0 * math.pi * FREQ / SR * 1.2 * 1.3
        worst = max(abs(b - a) for a, b in zip(self.out, self.out[1:]))
        self.assertLess(worst, bound)


class TestCloseEncounter(unittest.TestCase):
    """Near-sonic pass at 0.5 m distance: the Mach clamp must engage."""

    def test_clamped_and_finite(self):
        v = 320.0
        max_mach = 0.9
        engine = DopplerEngine(constant_velocity((-v, 0.5, 0.0), (v, 0.0, 0.0)),
                               speed_of_sound=C, sample_rate=SR,
                               max_mach=max_mach)
        out, pitches, delays = run_blocks(engine, sine(2 * SR), 64)
        out.extend(engine.flush())
        assert_all_finite(self, out)
        max_rate = 1.0 / (1.0 - max_mach)
        min_rate = 1.0 / (1.0 + max_mach)
        for p in pitches:
            self.assertLessEqual(p, max_rate + 1e-9)
            self.assertGreaterEqual(p, min_rate - 1e-9)
        self.assertGreater(max(pitches), 1.0)
        self.assertLess(min(pitches), 1.0)
        # Delay never jumps: per-block change is bounded by the rate clamp.
        limit = (max_rate - 1.0) / SR * 64 * 1.01
        for a, b in zip(delays, delays[1:]):
            self.assertLessEqual(abs(b - a), limit)
        # Linear interpolation can never overshoot the input amplitude.
        self.assertLessEqual(max(abs(x) for x in out), 1.0 + 1e-9)


class TestVelocityStep(unittest.TestCase):
    """Velocity flips sign mid-stream; position (and delay) stay continuous."""

    def test_smooth_delay_across_kink(self):
        traj = piecewise_velocity((100.0, 5.0, 0.0),
                                  [(1.0, (-60.0, 0.0, 0.0)),
                                   (1.0, (60.0, 0.0, 0.0))])
        engine = DopplerEngine(traj, speed_of_sound=C, sample_rate=SR)
        out, pitches, delays = run_blocks(engine, sine(2 * SR), 50)
        assert_all_finite(self, out)
        half = len(pitches) // 2
        self.assertGreater(sum(pitches[:half // 2]) / (half // 2), 1.0)
        self.assertLess(sum(pitches[-half // 2:]) / (half // 2), 1.0)
        # Position is continuous, so the delay must not jump at the kink.
        for a, b in zip(delays, delays[1:]):
            self.assertAlmostEqual(a, b, delta=0.01)
        self.assertLessEqual(max(abs(x) for x in out), 1.0 + 1e-9)


class TestBlockStateAndTail(unittest.TestCase):
    """Static source at 100 m: initial silence, tail, block-size invariance."""

    def make_engine(self):
        return DopplerEngine(constant_velocity((100.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                             speed_of_sound=C, sample_rate=SR)

    def test_initial_silence_and_tail(self):
        engine = self.make_engine()
        pcm = sine(SR // 2)
        out, _, _ = run_blocks(engine, pcm, 400)
        tail = engine.flush()
        delay_samples = 100.0 / C * SR  # ~2332
        first_voiced = next(i for i, x in enumerate(out) if x != 0.0)
        self.assertAlmostEqual(first_voiced, delay_samples, delta=2)
        self.assertAlmostEqual(len(tail), delay_samples, delta=4)
        self.assertGreater(max(abs(x) for x in tail), 0.5)
        self.assertAlmostEqual(engine.pending_tail, 0.0, delta=1.0 / SR)

    def test_block_size_invariance(self):
        pcm = sine(2000)
        out_a, _, _ = run_blocks(self.make_engine(), pcm, 500)
        out_b, _, _ = run_blocks(self.make_engine(), pcm, 2000)
        out_c, _, _ = run_blocks(self.make_engine(), pcm, 73)
        self.assertEqual(out_a, out_b)
        self.assertEqual(out_a, out_c)


if __name__ == "__main__":
    unittest.main()
