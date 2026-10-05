"""Camera pixel placement and per-cell colour (camera_species)."""

import numpy as np

from deforestation_monitoring.camera_species import (
    CX, CY, HEIGHT, WIDTH, cell_colour, cell_counts, open_rays, project)

LEVEL = np.eye(3)


def _frame(depth_m, colour=(60, 70, 30)):
    rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    rgb[:] = colour
    return rgb, np.full((HEIGHT, WIDTH), depth_m, np.float32)


def test_centre_pixel_lands_under_the_camera():
    rgb, depth = _frame(5.0)
    pts, _ = project(rgb, depth, LEVEL, 10.0, -3.0, 10.0)
    assert len(pts) == WIDTH * HEIGHT   # a flat surface 5.25 m high is all canopy
    # The centre pixel lies under the lens (mount 0.25 m forward and up),
    # 10.25 - 5 = 5.25 m high.
    x, y, z = pts.reshape(HEIGHT, WIDTH, 3)[int(round(CY)), int(round(CX))]
    assert abs(x - 10.25) < 0.05 and abs(y + 3.0) < 0.05
    assert abs(z - 5.25) < 0.05


def test_image_top_is_ahead_and_image_right_is_starboard():
    rgb, depth = _frame(5.0)
    pts, _ = project(rgb, depth, LEVEL, 0.0, 0.0, 10.0)
    grid = pts.reshape(HEIGHT, WIDTH, 3)
    assert grid[0, WIDTH // 2, 0] > grid[-1, WIDTH // 2, 0]      # top = +x
    assert grid[HEIGHT // 2, -1, 1] < grid[HEIGHT // 2, 0, 1]    # right = -y


def test_drone_body_and_ground_are_dropped():
    rgb, depth = _frame(0.6)            # the drone's own body, < 1 m away
    assert len(project(rgb, depth, LEVEL, 0.0, 0.0, 10.0)[0]) == 0
    rgb, depth = _frame(9.5)            # 0.75 m above ground: not canopy
    assert len(project(rgb, depth, LEVEL, 0.0, 0.0, 10.0)[0]) == 0


def test_yaw_rotates_the_footprint():
    rgb, depth = _frame(5.0)
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pts, _ = project(rgb, depth, yaw90, 0.0, 0.0, 10.0)
    grid = pts.reshape(HEIGHT, WIDTH, 3)
    # Facing +y now: the top of the image lies further along +y.
    assert grid[0, WIDTH // 2, 1] > grid[-1, WIDTH // 2, 1]


def test_cell_colour_counts_and_blue_green():
    pts = np.array([[0.1, 0.1, 5.0], [0.2, 0.1, 5.0], [1.1, 0.1, 5.0]])
    cols = np.array([[60, 70, 28], [60, 70, 35], [70, 80, 56]], np.float32)
    cells, count, bg = cell_colour(pts, cols, 0.25, -40.0, -40.0, 320, 320)
    assert list(count) == [2.0, 1.0]
    assert np.allclose(bg / count, [(0.4 + 0.5) / 2, 0.7])
    ix, iy = divmod(int(cells[0]), 320)
    assert (ix, iy) == (160, 160)


def test_open_rays_only_where_nothing_was_hit_above_the_height():
    _, canopy = _frame(5.0)                 # canopy top at 5.25 m everywhere
    assert len(open_rays(canopy, LEVEL, 0.0, 0.0, 10.0, 3.0)) == 0
    _, low = _frame(8.0)                    # everything hit below 3 m
    assert len(open_rays(low, LEVEL, 0.0, 0.0, 10.0, 3.0)) == WIDTH * HEIGHT
    _, empty = _frame(np.inf)               # ground is past the far clip
    xy = open_rays(empty, LEVEL, 2.0, -1.0, 10.0, 3.0)
    assert len(xy) == WIDTH * HEIGHT
    centre = xy.reshape(HEIGHT, WIDTH, 2)[int(round(CY)), int(round(CX))]
    assert abs(centre[0] - 2.25) < 0.05 and abs(centre[1] + 1.0) < 0.05


def test_cell_counts():
    xy = np.array([[0.1, 0.1], [0.2, 0.1], [1.1, 0.1], [99.0, 0.0]])
    cells, count = cell_counts(xy, 0.25, -40.0, -40.0, 320, 320)
    assert list(count) == [2.0, 1.0]           # the point off the grid is dropped
    assert divmod(int(cells[0]), 320) == (160, 160)
