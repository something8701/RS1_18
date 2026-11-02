# survey_loops.py: Counting the drone's full survey loops from ``/survey_status``
import re
from typing import Optional, Tuple

_WP = re.compile(r'STATE=PATROL wp=(\d+)/(\d+)')


def parse_waypoint(status: str) -> Optional[Tuple[int, int]]:
    m = _WP.search(status)
    return (int(m.group(1)), int(m.group(2))) if m else None


class LoopCounter:

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
        wp = parse_waypoint(status)
        return self.update(*wp) if wp else False
