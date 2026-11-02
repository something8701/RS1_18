# test_waste.py: Land waste: red test, bag placement, red blobs, site tracking
import math

import numpy as np

from deforestation_monitoring import camera_species as cs
from deforestation_monitoring.waste import (CONFIRMED, NOT_FOUND, RedBlob, SiteTracker,
                                            bag_sdf, bag_spot_near, find_red_blobs,
                                            red_fraction_bgr, red_mask)

TREES = [('oak_1', 0.0, 0.0), ('pine_2', 3.0, 0.0), ('oak_3', -2.0, 2.5)]


def test_red_mask_takes_bag_red_not_bark_or_leaves():
    px = np.array([[[200, 20, 20],
                    [110, 70, 40],
                    [90, 140, 40],
                    [60, 90, 70],
                    [80, 10, 10]]],
                  np.uint8)
    assert red_mask(px).tolist() == [[True, False, False, False, False]]
    bgr = np.zeros((10, 10, 3), np.uint8)
    bgr[:2, :5] = (20, 20, 200)
    assert red_fraction_bgr(bgr) == 0.1


def test_bag_goes_next_to_the_cut_tree_clear_of_others():
    x, y = bag_spot_near(TREES[0], TREES)
    d = math.hypot(x, y)
    assert 1.5 <= d <= 3.0
    assert all(math.hypot(x - tx, y - ty) >= 1.2 for _, tx, ty in TREES[1:])
    assert bag_spot_near(TREES[0], TREES) == (x, y)
    assert '<ellipsoid>' in bag_sdf('b', x, y) and '<static>true' in bag_sdf('b', x, y)


def test_red_blobs_need_enough_pixels_and_a_bag_sized_area():
    grid = np.zeros((40, 40), np.float32)
    grid[10:13, 10:13] = 5.0
    grid[30, 30] = 3.0
    blobs = find_red_blobs(grid, 0.25, -5.0, -5.0)
    assert len(blobs) == 1
    b = blobs[0]
    assert abs(b.x - (-5.0 + 11.5 * 0.25)) < 1e-6 and abs(b.y - (-5.0 + 11.5 * 0.25)) < 1e-6
    big = np.zeros((40, 40), np.float32)
    big[:, :] = 5.0
    assert find_red_blobs(big, 0.25, 0.0, 0.0) == []


def test_sites_keep_their_ids_and_take_ground_inspections():
    t = SiteTracker(match_dist=2.0)
    new = t.update([RedBlob(1.0, 1.0, 30, 0.4), RedBlob(10.0, 0.0, 20, 0.3)])
    assert [s.id for s in new] == [1, 2]
    assert t.update([RedBlob(1.5, 1.2, 40, 0.5)]) == []
    assert t.sites[0].pixels == 40
    assert t.inspected(1.2, 1.0, found=True).status == CONFIRMED
    assert t.inspected(9.0, 0.5, found=False).status == NOT_FOUND
    assert t.inspected(30.0, 30.0, found=True) is None


def test_ground_projection_puts_the_nadir_pixel_under_the_drone():
    mask = np.zeros((cs.HEIGHT, cs.WIDTH), bool)
    mask[cs.HEIGHT // 2, cs.WIDTH // 2] = True
    xy = cs.plane_points(mask, np.eye(3), 5.0, -3.0, 10.0, 0.2)
    assert xy.shape == (1, 2)
    assert abs(xy[0, 0] - 5.25) < 0.1 and abs(xy[0, 1] + 3.0) < 0.1


def _cloud(points_rgb):
    from sensor_msgs.msg import PointCloud2
    msg = PointCloud2()
    msg.point_step, msg.height, msg.width = 24, 1, len(points_rgb)
    buf = bytearray()
    for (x, y, z), (r, g, b) in points_rgb:
        buf += np.array([x, y, z, 0.0], np.float32).tobytes() + bytes([b, g, r, 0]) + bytes(4)
    msg.data = bytes(buf)
    return msg


def test_husky_cloud_red_points_and_clusters():
    from deforestation_monitoring.waste_detector import cloud_red_points, cluster_xy
    bag = [((3.0 + 0.01 * i, -0.5 + 0.01 * (i % 7), -0.3), (210, 25, 20)) for i in range(40)]
    grass = [((2.0, 1.0, -0.4), (40, 90, 35))] * 30
    far_red = [((12.0, 0.0, -0.3), (210, 25, 20))] * 40
    red = cloud_red_points(_cloud(bag + grass + far_red), max_range=8.0)
    assert len(red) == 40
    groups = cluster_xy(red, min_points=30)
    assert len(groups) == 1 and abs(groups[0][0] - 3.2) < 0.2 and groups[0][3] == 40
    assert cluster_xy(red[:10], min_points=30) == []


def test_ground_check_statuses():
    from deforestation_monitoring.tree_refine import (CUT_CONFIRMED, FALSE_LOST, HIDDEN,
                                                      NO_TRUNK, STANDING, UNCHECKED,
                                                      RefineConfig, Trunk, refine)
    drone = [(1, 0.0, 0.0), (2, 10.0, 0.0), (3, 20.0, 0.0), (4, 30.0, 0.0), (5, 40.0, 0.0),
             (6, 50.0, 0.0), (7, 45.0, 8.0)]
    trunks = [Trunk(0.8, 0.3, 9),
              Trunk(20.5, 0.0, 9, last_seen=200.0),
              Trunk(50.2, 0.0, 30, last_seen=50.0),
              Trunk(15.0, 4.0, 20, spread2=0.01),
              Trunk(25.0, 4.0, 20, spread2=0.25),
              Trunk(30.2, 0.0, 2)]
    views = {2: 20, 4: 20, 5: 3, 6: 15}
    lost = {2: 100.0, 3: 100.0, 6: 100.0}
    out = {(r.id, r.status) for r in refine(drone, lost, trunks, views, RefineConfig())}
    assert (1, STANDING) in out and (3, FALSE_LOST) in out
    assert (2, CUT_CONFIRMED) in out
    assert (6, CUT_CONFIRMED) in out
    assert (4, NO_TRUNK) in out
    assert (5, UNCHECKED) in out
    assert sum(1 for i, st in out if st == HIDDEN) == 1
    far = [Trunk(15.0, 30.0, 20, spread2=0.01)]
    assert all(r.status != HIDDEN for r in refine(drone, lost, far, views, RefineConfig()))


def test_trunks_are_forgotten_when_beams_pass_through():
    from deforestation_monitoring.tree_refine import TrunkTracker
    tr = TrunkTracker(forget=3)
    tr.add([(1.0, 1.0), (5.0, 5.0)], t=1.0)
    tr.add([(1.1, 1.0)], t=2.0)
    assert len(tr.trunks) == 2 and tr.trunks[0].hits == 2 and tr.trunks[0].last_seen == 2.0
    for _ in range(3):
        tr.seen_through(np.array([False, True]))
    assert [(round(k.x, 2), k.hits) for k in tr.trunks] == [(1.05, 2)]


def test_clear_views_need_the_beam_to_pass_the_spot():
    from deforestation_monitoring.tree_refine import clear_views
    n = 360
    ranges = np.full(n, 30.0, np.float32)
    ranges[90] = 4.0
    spots = np.array([[5.0, 0.0], [0.0, 5.0], [0.0, 3.0]])
    seen = clear_views((0.0, 0.0), 0.0, 0.0, 2 * math.pi / n, ranges, spots)
    assert seen.tolist() == [True, False, True]


def test_loss_box_covers_the_clearing_around_the_flag():
    from deforestation_monitoring.area_search import loss_box
    grid = np.zeros((80, 80), np.int8)
    grid[40:56, 44:52] = -100
    grid[10:12, 10:12] = -100
    b = loss_box(grid, 0.25, -10.0, -10.0, 2.0, 2.0)
    assert b == (0.0, 0.0, 4.0, 4.0)
    assert loss_box(grid, 0.25, -10.0, -10.0, -5.0, 5.0) == (-7.0, 3.0, -3.0, 7.0)


def test_sweep_covers_the_area_and_a_margin_from_the_nearest_corner():
    from deforestation_monitoring.area_search import free_spot, sweep_path
    path = sweep_path((0.0, 0.0, 10.0, 4.0), start=(15.0, -5.0), margin=2.0, spacing=3.0)
    xs, ys = [p[0] for p in path], [p[1] for p in path]
    assert (min(xs), max(xs), min(ys), max(ys)) == (-2.0, 12.0, -2.0, 6.0)
    assert path[0][:2] == (12.0, -2.0)
    assert len(path) == 8
    assert abs(path[0][2] - math.pi) < 1e-9
    cost = np.zeros((40, 40), np.int8)
    cost[18:22, 18:22] = 100
    assert free_spot(cost, 0.5, 0.0, 0.0, 3.0, 3.0) == (3.0, 3.0)
    x, y = free_spot(cost, 0.5, 0.0, 0.0, 10.0, 10.0)
    assert math.hypot(x - 10, y - 10) <= 1.5 and cost[int(y / 0.5), int(x / 0.5)] < 70
    cost[:, :] = 100
    assert free_spot(cost, 0.5, 0.0, 0.0, 10.0, 10.0) is None


def test_a_grown_clearing_overlaps_the_area_already_searched():
    from deforestation_monitoring.area_search import box_overlap
    old, grown = (5.5, 2.2, 9.5, 8.5), (6.2, 2.0, 15.2, 9.8)
    assert 0.25 < box_overlap(grown, old) < 0.5
    assert box_overlap(old, grown) > 0.8
    assert box_overlap((20, 20, 24, 24), old) == 0.0


def test_depth_obstacles_thin_the_camera_cloud():
    from sensor_msgs.msg import PointCloud2
    from deforestation_monitoring.depth_obstacles import thin_cloud
    h, w = 8, 12
    xyz = np.zeros((h, w, 4), np.float32)
    xyz[..., 0] = np.arange(w)[None, :] * 0.5
    xyz[0, 0, 0] = np.nan
    msg = PointCloud2()
    msg.height, msg.width, msg.point_step = h, w, 32
    buf = np.zeros((h, w, 32), np.uint8)
    buf[..., :16] = xyz.view(np.uint8).reshape(h, w, 16)
    msg.data = buf.tobytes()
    pts = thin_cloud(msg, stride=2, max_range=3.0)
    assert len(pts) == 4 * 4 - 1 and pts[:, 0].max() <= 3.0


def test_a_sighting_must_be_bag_sized_for_its_distance():
    from deforestation_monitoring.waste_detector import bag_pixels
    assert 2000 < bag_pixels(2.0) < 2500
    assert abs(bag_pixels(4.0) - bag_pixels(2.0) / 4) < 1
    assert 37 < 0.2 * bag_pixels(2.0)
