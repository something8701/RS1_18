"""Counting the drone's full survey loops from ``/survey_status``.

Shared by removal_test (how long to watch after the cut), scan_mapper and
parrot_tree_tracker (how many loops to fly before freezing the baseline).
"""
import re
from typing import Optional, Tuple

_WP = re.compile(r'STATE=PATROL wp=(\d+)/(\d+)')


def parse_waypoint(status: str) -> Optional[Tuple[int, int]]:
    """``(wp, total)`` from a demo_patrol status string, or None while the
    drone is not patrolling (flying to a flag, orbiting, returning)."""
    m = _WP.search(status)
    return (int(m.group(1)), int(m.group(2))) if m else None


class LoopCounter:
    """Counts full survey loops from ``/survey_status`` waypoint indices.

    A wrap (wp index going down) only counts as a completed loop if at
    least half of the waypoints were visited since the previous wrap, so a
    counter started just before a wrap does not count a near-empty loop.
    """

    def __init__(self):
        self.loops = 0
        self._last = None
        self._visited = set()

    def update(self, wp: int, total: int) -> bool:
        wrapped = self._last is not None and wp < self._last
        counted = False
        if wrapped:
            counted = len(self._visited) >= total / 2
            self.loops += int(counted)
            self._visited = set()
        self._visited.add(wp)
        self._last = wp
        return counted

    def update_from_status(self, status: str) -> bool:
        """Feed one ``/survey_status`` string; True if a loop just completed."""
        wp = parse_waypoint(status)
        return self.update(*wp) if wp else False
