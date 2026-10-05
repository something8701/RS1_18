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
    # 'b' is not in the baseline as its own tree -> can never be LOST
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
    assert any(w.startswith('LOST for a tree that is still standing') for w in whys)
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
    """Same targets as removal tests 1-2 so results stay comparable."""
    truth = parse_tree_truth(str(WORLDS / "dense_forest.sdf"))
    chosen = [n for n, _, _ in select_removal_targets(
        truth, 10, (-30.0, 30.0, -30.0, 30.0))]
    assert chosen == ["pine_75", "pine_111", "oak_78", "oak_137", "pine_117",
                      "oak_60", "pine_53", "oak_153", "oak_37", "pine_163"]


def test_loop_counter_ignores_partial_loop_after_removal():
    """Removal test 7: removal at wp 18/18, wrap two seconds later."""
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


def _tiers(**classes):
    return {n: dict(zip(("class", "top_before", "top_after"), v))
            for n, v in classes.items()}


def test_two_tier_understory_with_area_alert_passes():
    events = [{'type': 'LOST', 'id': 1, 'x': 0.1, 'y': 0.0, 't': 30.0}]
    tiers = _tiers(a=("visible", 6.2, 0.0), b=("understory", 5.3, 3.9))
    r = score_removal(TRUTH[:2], TRUTH, BASE, events, tiers=tiers,
                      area_alerts=[(10.5, 1.0)])
    assert r['passed'] and r['canopy_found'] == 1 and r['understory_ok'] == 1
    report = format_report(r, {})
    assert "Tiers:" in report and "AREA ALERT" in report


def test_two_tier_understory_unchanged_is_not_observable():
    tiers = _tiers(b=("understory", 5.2, 5.1))
    r = score_removal(TRUTH[1:2], TRUTH, BASE, [], tiers=tiers)
    assert r['passed'] and r['trees'][0]['not_observable']
    assert "NOT OBSERVABLE" in format_report(r, {})


def test_understory_tolerance_is_the_measured_noise_one_sided():
    # Dense run 8, pine_53 under an oak: 5.5 -> 5.2 m is within the noise
    # of standing understory trunks (p95 0.84 m, max 1.2 m); a rise is
    # never a removal signature.
    for before, after in ((5.5, 5.2), (5.0, 5.9), (5.4, 4.5)):
        tiers = _tiers(b=("understory", before, after))
        r = score_removal(TRUTH[1:2], TRUTH, BASE, [], tiers=tiers)
        assert r['passed'] and r['trees'][0]['not_observable'], (before, after)


def test_two_tier_understory_changed_without_alert_fails():
    tiers = _tiers(b=("understory", 5.3, 3.6))
    r = score_removal(TRUTH[1:2], TRUTH, BASE, [], tiers=tiers)
    assert not r['passed']


def test_two_tier_canopy_tree_still_needs_tree_level_lost():
    tiers = _tiers(a=("visible", 6.2, 0.0))
    r = score_removal(TRUTH[:1], TRUTH, BASE, [], tiers=tiers,
                      area_alerts=[(0.0, 0.0)])
    assert not r['passed']          # an area alert is not enough for canopy


def test_crown_shared_tree_passes_with_area_alert():
    tiers = _tiers(b=("crown-shared", 4.6, 3.6))
    ok = score_removal(TRUTH[1:2], TRUTH, BASE, [], tiers=tiers,
                       area_alerts=[(10.0, 1.5)])
    bad = score_removal(TRUTH[1:2], TRUTH, BASE, [], tiers=tiers)
    assert ok['passed'] and not bad['passed']
    assert "crown-shared" in format_report(ok, {})


def test_identity_credits_an_offset_oak_detection():
    """Dense oaks are detected ~1-2 m off their trunk; the baseline tree
    paired with the removed trunk going LOST is the right tree."""
    truth = [("oak_1", 0.0, 0.0), ("oak_2", 8.0, 0.0)]
    base = [(1, 1.8, 0.3), (2, 8.2, 0.0)]           # oak_1 detected 1.8 m off
    events = [{'type': 'LOST', 'id': 1, 'x': 1.8, 'y': 0.3, 't': 40.0}]
    r = score_removal(truth[:1], truth, base, events)
    assert r['passed'] and r['trees'][0]['baseline_id'] == 1
    legacy = score_removal(truth[:1], truth, base, events, identity_radius=0)
    assert not legacy['passed']                      # 1.5 m position match misses


def test_identity_never_credits_a_neighbours_lost():
    """A neighbour's LOST 1.2 m from the removed trunk is still a false
    report when that baseline tree is paired with the neighbour."""
    truth = [("oak_1", 0.0, 0.0), ("pine_2", 2.4, 0.0)]
    base = [(1, 0.3, 0.0), (2, 1.2, 0.0)]            # pine_2 detected 1.2 m off
    events = [{'type': 'LOST', 'id': 2, 'x': 1.2, 'y': 0.0, 't': 40.0}]
    r = score_removal(truth[:1], truth, base, events)
    assert r['true_positives'] == 0 and r['false_events'] == 1
    assert 'pine_2' in r['false_event_list'][0]['why']


def test_area_alert_far_from_every_removal_is_false():
    """An area alert > 6 m from every removed tree is a
    false event; one near a removal is not (repeats within 1 m count once)."""
    events = [{'type': 'LOST', 'id': 1, 'x': 0.1, 'y': 0.0, 't': 30.0}]
    tiers = _tiers(a=("visible", 6.2, 0.0))
    near = score_removal(TRUTH[:1], TRUTH, BASE, events, tiers=tiers,
                         area_alerts=[(2.0, 1.0)])
    assert near['passed'] and near['false_events'] == 0
    far = score_removal(TRUTH[:1], TRUTH, BASE, events, tiers=tiers,
                        area_alerts=[(2.0, 1.0), (20.0, 20.0), (20.3, 20.2)])
    assert not far['passed'] and far['false_events'] == 1
    assert "AREA alert at (20.0, 20.0)" in format_report(far, {})


def test_loop_counter_reads_patrol_status_strings():
    """survey_loops: the same counter feeds scan_mapper / the tracker
    (baseline_min_loops, freeze_min_loops) from /survey_status."""
    from deforestation_monitoring.survey_loops import LoopCounter, parse_waypoint
    assert parse_waypoint('STATE=PATROL wp=3/18 coverage=11% diverted=0') == (3, 18)
    assert parse_waypoint('STATE=ORBITING target=(1.0,2.0) diverted=1') is None
    c = LoopCounter()
    for loop in range(2):
        for wp in range(1, 19):
            c.update_from_status(f'STATE=PATROL wp={wp}/18 coverage=50% diverted=0')
        c.update_from_status('STATE=ORBITING target=(1.0,2.0) diverted=1')   # ignored
    assert c.loops == 1                          # the second wrap has not come yet
    assert c.update_from_status('STATE=PATROL wp=1/18 coverage=0% diverted=0') is True
    assert c.loops == 2
