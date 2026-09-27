"""Offline tests for the live removal-test scoring."""

import math
from pathlib import Path

from deforestation_monitoring.removal_test import (
    baseline_quality, format_report, parse_canopy_events, score_removal,
    select_removal_targets)
from deforestation_monitoring.tree_detection import parse_tree_truth

WORLDS = Path(__file__).resolve().parents[2] / "41068_ignition_bringup" / "worlds"
TRUTH = [("a", 0.0, 0.0), ("b", 10.0, 0.0), ("c", 20.0, 0.0), ("d", 0.0, 10.0)]
BASE = [(1, 0.1, 0.0), (2, 10.0, 0.2), (3, 20.0, 0.0), (4, 0.0, 10.0)]


def test_dense_targets_are_inside_and_separated():
    truth = parse_tree_truth(str(WORLDS / "dense_forest.sdf"))
    box = (-30.0, 30.0, -30.0, 30.0)
    chosen = select_removal_targets(truth, 10, box, margin=5.0,
                                    min_separation=10.0)
    assert len(chosen) == 10
    for _, x, y in chosen:
        assert -25.0 <= x <= 25.0 and -25.0 <= y <= 25.0
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            assert math.hypot(a[1] - b[1], a[2] - b[2]) >= 10.0
    assert chosen == select_removal_targets(truth, 10, box)  # deterministic


def test_all_found_no_false_is_pass():
    events = [{'type': 'LOST', 'id': 1, 'x': 0.1, 'y': 0.0, 't': 30.0},
              {'type': 'LOST', 'id': 2, 'x': 10.0, 'y': 0.2, 't': 40.0}]
    r = score_removal(TRUTH[:2], TRUTH, BASE, events)
    assert r['passed'] and r['true_positives'] == 2 and r['false_events'] == 0


def test_missed_tree_is_diagnosed():
    # 'b' is not in the baseline as its own tree, so it can never be LOST
    base = [(1, 0.1, 0.0), (3, 20.0, 0.0)]
    events = [{'type': 'LOST', 'id': 1, 'x': 0.1, 'y': 0.0, 't': 30.0}]
    r = score_removal(TRUTH[:2], TRUTH, base, events)
    assert not r['passed'] and r['missed'] == 1
    b = [t for t in r['trees'] if t['name'] == 'b'][0]
    assert 'not in the tracker baseline' in b['why_missed']


def test_missed_tree_without_canopy_loss():
    r = score_removal(TRUTH[:1], TRUTH, BASE, [], canopy_events=[])
    assert 'no canopy loss' in r['trees'][0]['why_missed']


def test_false_lost_and_gained_are_counted():
    events = [{'type': 'LOST', 'id': 1, 'x': 0.1, 'y': 0.0, 't': 30.0},
              {'type': 'LOST', 'id': 4, 'x': 0.0, 'y': 10.0, 't': 35.0},
              {'type': 'GAINED', 'id': 9, 'x': 5.0, 'y': 5.0, 't': 50.0}]
    r = score_removal(TRUTH[:1], TRUTH, BASE, events)
    assert r['true_positives'] == 1 and r['false_events'] == 2
    whys = {e['why'] for e in r['false_event_list']}
    assert 'LOST for a tree that is still standing' in whys
    report = format_report(r, {'world': 'test'})
    assert 'FAIL' in report and 'd ' in report


def test_one_event_cannot_count_for_two_trees():
    truth = [("a", 0.0, 0.0), ("b", 1.0, 0.0)]
    events = [{'type': 'LOST', 'id': 1, 'x': 0.5, 'y': 0.0, 't': 10.0}]
    r = score_removal(truth, truth, [(1, 0.0, 0.0), (2, 1.0, 0.0)], events)
    assert r['true_positives'] == 1 and r['missed'] == 1


def test_parse_canopy_events():
    s = ("CANOPY LOST: ~40 cells (3m²) near (-8.4, 0.5); "
         "NEW CANOPY: ~9 cells (1m²) near (13.2, 2.3)")
    assert parse_canopy_events(s) == [('LOST', -8.4, 0.5), ('NEW', 13.2, 2.3)]


def test_baseline_quality():
    q = baseline_quality(TRUTH, BASE[:3] + [(5, 30.0, 30.0)],
                         (-5.0, 25.0, -5.0, 15.0))
    assert (q['tp'], q['fp'], q['fn']) == (3, 1, 1)


def test_visible_only_filter_skips_pines_near_oaks():
    from deforestation_monitoring.removal_test import occluded_from_above
    truth = parse_tree_truth(str(WORLDS / "dense_forest.sdf"))
    hidden = occluded_from_above(truth)
    chosen = select_removal_targets(truth, 10, (-30.0, 30.0, -30.0, 30.0),
                                    visible_only=True)
    assert not hidden & {n for n, _, _ in chosen}


def test_default_targets_are_the_original_ten():
    """The default targets stay the same so results can be compared."""
    truth = parse_tree_truth(str(WORLDS / "dense_forest.sdf"))
    chosen = [n for n, _, _ in select_removal_targets(
        truth, 10, (-30.0, 30.0, -30.0, 30.0))]
    assert chosen == ["pine_75", "pine_111", "oak_78", "oak_137", "pine_117",
                      "oak_60", "pine_53", "oak_153", "oak_37", "pine_163"]


def test_loop_counter_ignores_partial_loop_after_removal():
    """Removal at waypoint 18/18 and a wrap two seconds later."""
    from deforestation_monitoring.removal_test import LoopCounter
    c = LoopCounter()
    counted = [c.update(wp, 18) for wp in [18, 1, 2]]   # partial loop
    assert counted == [False, False, False] and c.loops == 0
    for wp in range(3, 19):
        c.update(wp, 18)
    assert c.update(1, 18) is True and c.loops == 1     # a real loop


def test_balanced_targets_alternate_species():
    truth = parse_tree_truth(str(WORLDS / "showcase_forest.sdf"))
    chosen = select_removal_targets(truth, 10, (-30.0, 30.0, -30.0, 30.0),
                                    balanced=True)
    species = [n.split("_")[0] for n, _, _ in chosen]
    assert len(chosen) == 10
    assert species.count("oak") == 5 and species.count("pine") == 5
