"""Offline tests for replay_evaluate (GATE 4 evaluator)."""

import math

import numpy as np
from rclpy.serialization import serialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from deforestation_monitoring.replay_evaluate import accumulate, scan_times


class FakeReader:
    """Minimal rosbag2 SequentialReader stand-in."""

    def __init__(self, messages):
        self._msgs = list(messages)

    def has_next(self):
        return bool(self._msgs)

    def read_next(self):
        return self._msgs.pop(0)


def _odom(x, y):
    msg = Odometry()
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.orientation.w = 1.0
    return msg


def _nadir_scan(range_m=8.0):
    """A fan whose middle beam points straight down after the 90° pitch."""
    scan = LaserScan()
    scan.angle_min = -0.05
    scan.angle_increment = 0.01
    scan.range_max = 40.0
    scan.ranges = [range_m] * 11
    return scan


def _bag():
    """Pass A flies over x=-5, pass B over x=+5; 1 s apart in bag time."""
    msgs = []
    t = 0
    for x in (-5.0, 5.0):
        for _ in range(3):
            msgs.append(('/parrot1/odometry', serialize_message(_odom(x, 0.0)), t))
            msgs.append(('/parrot1/scan', serialize_message(_nadir_scan()), t + 1))
            t += 10**9 // 10
        t += 10**9
    return msgs


def test_time_split_separates_passes():
    times = scan_times(FakeReader(_bag()))
    assert len(times) == 6
    split = int(np.median(times))
    _, hits, _, half_hits, _, _ = accumulate(
        FakeReader(_bag()), altitude=10.0, split_ns=split)
    ix_a = int((-5.0 + 40.0) / 0.25)
    ix_b = int((5.0 + 40.0) / 0.25)
    # Pass A only saw x=-5, pass B only saw x=+5.
    assert half_hits[0][ix_a - 2:ix_a + 3, :].sum() > 0
    assert half_hits[0][ix_b - 2:ix_b + 3, :].sum() == 0
    assert half_hits[1][ix_b - 2:ix_b + 3, :].sum() > 0
    assert half_hits[1][ix_a - 2:ix_a + 3, :].sum() == 0
    assert hits.sum() == half_hits[0].sum() + half_hits[1].sum()


def test_interleave_split_mixes_passes():
    """The old behaviour: both 'halves' contain both passes."""
    _, _, _, half_hits, _, _ = accumulate(
        FakeReader(_bag()), altitude=10.0, split_ns=None)
    ix_a = int((-5.0 + 40.0) / 0.25)
    assert half_hits[0][ix_a - 2:ix_a + 3, :].sum() > 0
    assert half_hits[1][ix_a - 2:ix_a + 3, :].sum() > 0


def test_scan_height_uses_altitude():
    """8 m range straight down from 10 m altitude -> 2 m canopy."""
    height, hits, _, _, _, _ = accumulate(
        FakeReader(_bag()[:2]), altitude=10.0)
    top = height[hits > 0].max()
    assert math.isclose(top, 2.0, abs_tol=0.05)


def test_flicker_ignores_centimetre_jitter():
    """Live bag: both passes found the same 5 trees within 0.13 m, but
    exact 0.1 m-rounded comparison reported 3 lost + 3 gained."""
    from deforestation_monitoring.replay_evaluate import match_passes
    a = [(-11.99, 0.0), (-7.95, -8.55), (-0.12, 7.4), (8.0, -8.01),
         (12.28, -0.35)]
    b = [(-12.01, 0.0), (-8.01, -8.48), (0.0, 7.29), (8.0, -8.0),
         (12.36, -0.32)]
    lost, gained, shifts = match_passes(a, b)
    assert (lost, gained) == (0, 0)
    assert max(shifts) < 0.2


def test_flicker_counts_a_real_removal():
    from deforestation_monitoring.replay_evaluate import match_passes
    a = [(-12.0, 0.0), (-8.0, -8.0), (0.0, 8.0)]
    b = [(-12.0, 0.0), (0.0, 8.0)]           # oak_1 removed
    assert match_passes(a, b)[:2] == (1, 0)


def test_recent_grid_is_latest_reading():
    """Parity must compare the live map (latest reading) with a latest-
    reading grid: a crown seen first, then ground, reads ground."""
    msgs = []
    t = 0
    for rng in (4.0, 10.0):          # first 6 m canopy, then 0 m ground
        msgs.append(('/parrot1/odometry', serialize_message(_odom(0, 0)), t))
        msgs.append(('/parrot1/scan', serialize_message(_nadir_scan(rng)), t + 1))
        t += 10**8
    height, hits, _, _, _, recent = accumulate(FakeReader(msgs), altitude=10.0)
    ix = iy = int(40.0 / 0.25)
    assert math.isclose(height[ix, iy], 6.0, abs_tol=0.05)
    assert math.isclose(recent[ix, iy], 0.0, abs_tol=0.05)


def test_recent_grid_returns_to_ground_below_zero():
    """Ground projects to about -0.2 m; it must still overwrite a tree."""
    msgs = []
    t = 0
    for rng in (4.0, 10.2):          # 6 m canopy, then ground at -0.2 m
        msgs.append(('/parrot1/odometry', serialize_message(_odom(0, 0)), t))
        msgs.append(('/parrot1/scan', serialize_message(_nadir_scan(rng)), t + 1))
        t += 10**8
    _, _, _, _, _, recent = accumulate(FakeReader(msgs), altitude=10.0)
    ix = iy = int(40.0 / 0.25)
    assert recent[ix, iy] == 0.0
