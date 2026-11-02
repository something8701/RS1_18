# test_radio.py: Radio_model (link budgets through the trees) and radio_link's store and forward
import math

import numpy as np
import pytest
from std_msgs.msg import String

from deforestation_monitoring.radio_link import DOWN, UP, Channel
from deforestation_monitoring.radio_model import (foliage_db, foliage_depth, fspl_db,
                                                  link_margin)

RES, O = 0.25, -20.0


def _forest(x0, x1, height=5.0):
    h = np.zeros((160, 160), np.float32)
    h[:, int((x0 - O) / RES):int((x1 - O) / RES)] = height
    return h


def test_free_space_loss():
    assert fspl_db(10.0, 2.4) == pytest.approx(60.05, abs=0.05)
    assert fspl_db(20.0, 2.4) - fspl_db(10.0, 2.4) == pytest.approx(6.02, abs=0.01)
    assert fspl_db(0.1, 2.4) == fspl_db(1.0, 2.4)


def test_weissberger_foliage_loss():
    assert foliage_db(0.0, 2.4) == 0.0
    assert foliage_db(10.0, 2.4) == pytest.approx(0.45 * 2.4 ** 0.284 * 10.0)
    assert foliage_db(30.0, 2.4) == pytest.approx(1.33 * 2.4 ** 0.284 * 30.0 ** 0.588)
    assert abs(foliage_db(14.01, 2.4) - foliage_db(14.0, 2.4)) < 0.5
    assert foliage_db(40.0, 2.4) > foliage_db(20.0, 2.4) > foliage_db(5.0, 2.4)


def test_foliage_depth_follows_the_path_height():
    h = _forest(0.0, 5.0)
    ground = foliage_depth(h, RES, O, O, (-10.0, 0.0, 1.0), (10.0, 0.0, 1.0))
    assert ground == pytest.approx(5.0, abs=0.3)
    above = foliage_depth(h, RES, O, O, (-10.0, 0.0, 10.0), (10.0, 0.0, 10.0))
    assert above == 0.0
    down = foliage_depth(h, RES, O, O, (-10.0, 0.0, 10.0), (2.5, 0.0, 1.0))
    assert down == pytest.approx(0.2 * math.hypot(12.5, 9.0), abs=0.3)
    unscanned = np.full_like(h, -1.0)
    assert foliage_depth(unscanned, RES, O, O, (-10.0, 0.0, 1.0), (10.0, 0.0, 1.0)) == 0.0
    assert foliage_depth(None, RES, O, O, (-10.0, 0.0, 1.0), (10.0, 0.0, 1.0)) == 0.0


def test_the_drone_above_the_canopy_has_the_better_link():
    h = _forest(-15.0, 15.0)
    budget = dict(f_ghz=2.4, tx_dbm=-6.0, gains_dbi=(0.0, 3.0), sensitivity_dbm=-85.0,
                  fade_margin_db=10.0)
    ground, d, foliage = link_margin((12.0, 0.0, 0.8), (-12.0, 0.0, 2.0), h, RES, O, O, **budget)
    assert ground < 0 and d == pytest.approx(math.dist((12, 0, 0.8), (-12, 0, 2)))
    assert foliage > 20.0
    air, _, _ = link_margin((12.0, 0.0, 0.8), (6.0, 0.0, 10.0), h, RES, O, O, **budget)
    assert air > 0


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, m):
        self.sent.append(m)


def _ch(keep_all, news):
    c = Channel(UP, '/x', keep_all, news)
    c.pub = _Pub()
    return c


def test_events_wait_in_order_state_keeps_the_newest():
    events, state = _ch(True, True), _ch(False, False)
    for i in range(3):
        events.push(String(data=f'e{i}'), float(i))
        state.push(String(data=f's{i}'), float(i))
    assert len(events.waiting) == 3 and len(state.waiting) == 1
    assert events.oldest_news() == 0.0 and state.oldest_news() is None
    assert events.flush() == 3 and [m.data for m in events.pub.sent] == ['e0', 'e1', 'e2']
    assert state.flush() == 1 and state.pub.sent[0].data == 's2'
    assert not events.waiting and events.delivered == 3


def test_news_is_a_changed_key_and_keeps_its_age():
    c = _ch(False, lambda m: m.data[0])
    c.push(String(data='a1'), 0.0)
    assert c.oldest_news() == 0.0
    c.flush()
    c.push(String(data='a2'), 5.0)
    assert c.oldest_news() is None
    c.push(String(data='b1'), 6.0)
    c.push(String(data='b2'), 9.0)
    assert c.oldest_news() == 6.0 and len(c.waiting) == 1
    assert c.flush() == 1 and c.last_key == 'b' and c.oldest_news() is None
    assert DOWN != UP


def test_prior_drops_trees_whose_crown_is_gone():
    from nav_msgs.msg import OccupancyGrid
    from deforestation_monitoring.drone_prior_map import crown_gone
    g = OccupancyGrid()
    g.info.resolution, g.info.width, g.info.height = RES, 160, 160
    g.info.origin.position.x = g.info.origin.position.y = O
    d = np.zeros((160, 160), np.int8)
    d[76:84, 76:84] = -100
    d[100:104, 100:104] = -30
    d[0:20, 0:20] = -1
    g.data = d.ravel().tolist()
    assert crown_gone(g, 0.0, 0.0)
    assert not crown_gone(g, 6.1, 6.1)
    assert not crown_gone(g, -19.0, -19.0)
    assert not crown_gone(g, 10.0, -10.0)
    assert not crown_gone(None, 0.0, 0.0)


def _spec(topic):
    from deforestation_monitoring.radio_link import TOPICS
    d, t, _, _, keep_all, news = next(x for x in TOPICS if x[1] == topic)
    return _ch(keep_all, news)


def test_r7_the_baseline_freezing_is_news_but_waiting_updates_are_not():
    c = _spec('/drone_baseline_status')
    c.push(String(data='waiting: coverage 40%'), 0.0)
    assert c.oldest_news() is None
    c.push(String(data='frozen: 68 trees'), 1.0)
    assert c.oldest_news() == 1.0
    c.flush()
    c.push(String(data='frozen: 68 trees'), 2.0)
    assert c.oldest_news() is None


def test_r10_a_full_queue_counts_what_it_drops():
    c = _ch(True, True)
    for i in range(c.waiting.maxlen + 3):
        c.push(String(data=str(i)), float(i))
    assert c.dropped == 3 and len(c.waiting) == c.waiting.maxlen
    assert c.waiting[0][0].data == '3'


@pytest.fixture(scope='module')
def ros_ctx():
    import rclpy
    rclpy.init()
    yield
    rclpy.shutdown()


def _radio():
    from deforestation_monitoring.radio_link import RadioLink
    n = RadioLink()
    n.relay_after = n.relay_cooldown = 0.0
    n.heights = np.full((320, 320), 6.0, np.float32)
    n.grid_geo = (0.25, -40.0, -40.0)
    return n


def _news(n):
    ch = next(c for c in n.channels if c.topic == '/inspection_reports')
    ch.push(String(data='SITE INSPECTED'), n._now() - 1.0)


def test_r9_relay_called_only_where_it_closes_both_links(ros_ctx):
    n = _radio()
    try:
        n.husky, n.drone, n.drone_t = (25.0, 0.0), (5.0, 5.0), n._now()
        _news(n)
        n.tick()
        assert n.route is None and n.relay['active'] and n.relay['count'] == 1
        assert 0.0 < n.relay['x'] < 25.0
        m = _radio()
        try:
            m.husky, m.drone, m.drone_t = (39.0, 39.0), (5.0, 5.0), m._now()
            _news(m)
            m.tick()
            assert m.route is None and not m.relay['active'] and m.no_relay_logged
        finally:
            m.destroy_node()
    finally:
        n.destroy_node()


def test_r8_a_drone_gone_quiet_relays_nothing(ros_ctx):
    n = _radio()
    try:
        n.husky, n.drone, n.drone_t = (25.0, 0.0), (-30.0, 30.0), n._now() - 60.0
        _news(n)
        n.tick()
        assert not n.drone_alive() and n.route is None and not n.relay['active']
        assert 'husky-drone' not in n.margins and not n.up['drone-base']
        n.drone_t = n._now()
        n.tick()
        assert n.route is None and n.relay['active']
        n.drone_t = n._now() - 60.0
        n.tick()
        assert not n.relay['active']
    finally:
        n.destroy_node()
