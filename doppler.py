"""Pure-python Doppler audio engine for games.

Physical model
--------------
All quantities use SI units: positions in meters, time in seconds,
velocities in m/s, ``speed_of_sound`` in m/s (default 343.0, air at 20 C),
``sample_rate`` in Hz. PCM blocks are plain sequences of floats.

A listener sample at listener time ``t`` reproduces the source sample that
was emitted at the retarded (emission) time ``tau`` solving::

    tau + |L(t) - S(tau)| / c = t

The engine keeps a continuous emission clock ``tau`` that is advanced
sample by sample, so the propagation delay glides smoothly and never
jumps by whole samples at block boundaries. The instantaneous pitch /
resample factor is ``d tau / dt``; for a stationary listener and a source
with radial (closing) speed ``u`` it equals ``c / (c - u)``.

Speed limit
-----------
``max_mach`` (0 < max_mach < 1) bounds the allowed rate to
``[1/(1+max_mach), 1/(1-max_mach)]``. Near-sonic or supersonic geometries
(where the retarded-time equation has no unique subsonic solution) are
clamped instead of producing singularities or backwards reads, so the
emission clock always advances monotonically.

Block state
-----------
The engine is a stateful stream processor. Across ``process()`` calls it
keeps: the listener clock ``t``, the emission clock ``tau``, the last
resample rate, and the not-yet-consumed emitted PCM. Output therefore
depends only on the sample stream, never on how the stream was split
into blocks. Sound still in flight after the input ends is drained with
``flush()`` (the "tail", roughly ``delay * sample_rate`` samples).
"""

import math


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a):
    return math.sqrt(_dot(a, a))


def constant_velocity(position, velocity):
    """Trajectory: ``pos(t) = position + velocity * t``."""
    x0, y0, z0 = position
    vx, vy, vz = velocity

    def pos(t):
        return (x0 + vx * t, y0 + vy * t, z0 + vz * t)

    return pos


def piecewise_velocity(position, segments):
    """Trajectory with piecewise-constant velocity and continuous position.

    ``segments`` is a list of ``(duration, (vx, vy, vz))`` pairs. Before
    ``t = 0`` the start position is held; after the last segment the final
    velocity is kept. Velocity may change abruptly between segments.
    """
    legs = []
    t0 = 0.0
    p0 = (float(position[0]), float(position[1]), float(position[2]))
    for duration, velocity in segments:
        vx, vy, vz = velocity
        legs.append((t0, t0 + duration, p0, (float(vx), float(vy), float(vz))))
        p0 = (p0[0] + vx * duration, p0[1] + vy * duration, p0[2] + vz * duration)
        t0 += duration
    if not legs:
        raise ValueError("segments must not be empty")

    def pos(t):
        if t <= 0.0:
            return legs[0][2]
        for t0, t1, p0, v in legs:
            if t <= t1:
                dt = t - t0
                return (p0[0] + v[0] * dt, p0[1] + v[1] * dt, p0[2] + v[2] * dt)
        t0, _, p0, v = legs[-1]
        dt = t - t0
        return (p0[0] + v[0] * dt, p0[1] + v[1] * dt, p0[2] + v[2] * dt)

    return pos


class DopplerEngine:
    """Streaming Doppler resampler for one moving source and one listener.

    ``source_pos`` and ``listener_pos`` are callables mapping time (s) to
    ``(x, y, z)`` in meters. ``listener_pos`` defaults to a static listener
    at the origin.
    """

    def __init__(self, source_pos, listener_pos=None, *,
                 speed_of_sound=343.0, sample_rate=48000, max_mach=0.9):
        if speed_of_sound <= 0.0:
            raise ValueError("speed_of_sound must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 < max_mach < 1.0:
            raise ValueError("max_mach must be in (0, 1)")
        self.source_pos = source_pos
        self.listener_pos = listener_pos or (lambda t: (0.0, 0.0, 0.0))
        self.speed_of_sound = float(speed_of_sound)
        self.sample_rate = int(sample_rate)
        self.max_mach = float(max_mach)
        self._min_rate = 1.0 / (1.0 + self.max_mach)
        self._max_rate = 1.0 / (1.0 - self.max_mach)
        # Stream state carried across blocks.
        self._buf = []          # emitted source samples not yet consumed
        self._base = 0          # absolute sample index of _buf[0]
        self._emitted = 0       # total samples pushed so far
        self._time = 0.0        # listener clock (s)
        self._tau = None        # emission clock (s), None until first sample
        self._rate = 1.0        # instantaneous d tau / dt
        self._delay = 0.0       # current propagation delay (s)

    @property
    def pitch_factor(self):
        """Current pitch multiplier (d tau / dt); >1 means higher pitch."""
        return self._rate

    @property
    def resample_rate(self):
        """Alias of ``pitch_factor``: input samples consumed per output sample."""
        return self._rate

    @property
    def delay(self):
        """Current propagation delay in seconds."""
        return self._delay

    @property
    def time(self):
        """Current listener clock in seconds."""
        return self._time

    @property
    def pending_tail(self):
        """Approximate seconds of emitted audio still in flight."""
        remaining = (self._emitted - 1) - self._tau * self.sample_rate
        return max(0.0, remaining) / self.sample_rate

    def closing_speed(self):
        """Radial closing speed in m/s at the current state.

        Positive means the source-listener distance is shrinking.
        """
        h = 1e-4
        t = self._time
        tau = self._tau if self._tau is not None else self._solve_tau(t)
        direction = _sub(self.listener_pos(t), self.source_pos(tau))
        dist = _length(direction)
        if dist == 0.0:
            return 0.0
        n = (direction[0] / dist, direction[1] / dist, direction[2] / dist)
        vs = _sub(self.source_pos(tau + h), self.source_pos(tau - h))
        vl = _sub(self.listener_pos(t + h), self.listener_pos(t - h))
        vs = (vs[0] / (2 * h), vs[1] / (2 * h), vs[2] / (2 * h))
        vl = (vl[0] / (2 * h), vl[1] / (2 * h), vl[2] / (2 * h))
        return _dot(_sub(vs, vl), n)

    def process(self, pcm_block):
        """Push one PCM block, return ``(output_block, pitch_factor)``.

        The output block has the same length as the input and is aligned to
        the listener clock; it starts with ``delay`` seconds of silence for
        a freshly created engine.
        """
        samples = [float(x) for x in pcm_block]
        self._buf.extend(samples)
        self._emitted += len(samples)
        out = []
        for _ in samples:
            out.append(self._step())
        self._trim()
        return out, self._rate

    def flush(self):
        """Drain the in-flight tail after the input stream has ended."""
        out = []
        end = self._emitted - 1
        guard = int(self._delay * self.sample_rate) + 2 * self.sample_rate
        while self._tau is not None and self._tau * self.sample_rate < end:
            if guard <= 0:
                break
            guard -= 1
            out.append(self._step())
        self._trim()
        return out

    def _solve_tau(self, t):
        """Retarded emission time for listener time ``t`` (fixed-point)."""
        if self._tau is None:
            tau = t
        else:
            tau = self._tau + self._rate / self.sample_rate
        for _ in range(6):
            dist = _length(_sub(self.listener_pos(t), self.source_pos(tau)))
            tau = t - dist / self.speed_of_sound
        return tau

    def _step(self):
        t = self._time
        tau = self._solve_tau(t)
        if self._tau is None:
            self._tau = tau
            self._rate = 1.0
        else:
            raw_rate = (tau - self._tau) * self.sample_rate
            self._rate = min(max(raw_rate, self._min_rate), self._max_rate)
            self._tau += self._rate / self.sample_rate
        self._delay = t - self._tau
        sample = self._read(self._tau * self.sample_rate)
        self._time = t + 1.0 / self.sample_rate
        return sample

    def _read(self, pos):
        """Linear-interpolated read at absolute emission sample index ``pos``."""
        if pos < 0.0:
            return 0.0
        i = int(pos)
        frac = pos - i
        k = i - self._base
        if k < 0:
            return 0.0
        if k + 1 < len(self._buf):
            a = self._buf[k]
            b = self._buf[k + 1]
            return a + (b - a) * frac
        if k < len(self._buf):
            return self._buf[k]
        return 0.0

    def _trim(self):
        """Drop buffer samples the (monotonic) emission clock has passed."""
        if self._tau is None:
            return
        drop = int(self._tau * self.sample_rate) - 1 - self._base
        if drop > 0:
            del self._buf[:drop]
            self._base += drop
