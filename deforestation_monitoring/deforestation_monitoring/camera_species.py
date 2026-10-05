"""Place drone camera pixels on the canopy grid and score their colour.

The Parrot's RGB-D camera looks straight down (parrot.urdf.xacro: xyz
0.25 0 0.25, rpy 0 1.57 0; 320x240, hfov 2.0944 rad). Each pixel with a
valid depth is placed in the world from the drone pose. The drone's
odometry reports z = 0, so the camera height is the survey altitude, as in
scan_mapper; on recorded dense runs the resulting heights match the LiDAR
canopy map to a median of 0.04 m.

Colour cue: blue / green. Rendered oak leaves are saturated yellow-green
(B/G ~0.39 at the treetop), pine needles a duller green (~0.7). Ground
has a pine-like B/G too, so only pixels 1.5-8.5 m above ground count. The
drone's own body fills the bottom-centre of every frame < 1 m from the
lens, so pixels closer than 2 m are dropped.
"""

import math

import numpy as np

WIDTH, HEIGHT, HFOV = 320, 240, 2.0944
FX = (WIDTH / 2) / math.tan(HFOV / 2)
FY = FX
CX, CY = (WIDTH - 1) / 2, (HEIGHT - 1) / 2
MOUNT_T = np.array([0.25, 0.0, 0.25])
PITCH = 1.57
MIN_DEPTH, MAX_DEPTH = 2.0, 9.9
MIN_HEIGHT, MAX_HEIGHT = 1.5, 8.5
MIN_GREEN = 20                     # near-black shadow pixels carry no colour

# camera_link -> base_link (pitch about y), optical -> camera_link
R_BC = np.array([[math.cos(PITCH), 0, math.sin(PITCH)],
                 [0, 1, 0],
                 [-math.sin(PITCH), 0, math.cos(PITCH)]])
R_CO = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
_U, _V = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
RAY = np.stack([(_U - CX) / FX, (_V - CY) / FY, np.ones(_U.shape)], -1)


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(rgb, depth, rotation, x, y, altitude):
    """World points (N, 3) and colours (N, 3) of the canopy pixels.

    ``rgb`` is (H, W, 3) uint8, ``depth`` (H, W) float32 metres along the
    optical axis, ``rotation`` the 3x3 world-from-body matrix.
    """
    ok = np.isfinite(depth) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    ok &= rgb[..., 1] >= MIN_GREEN
    p_opt = RAY[ok] * depth[ok][:, None]
    p_body = (R_BC @ (R_CO @ p_opt.T)).T + MOUNT_T
    p_world = (rotation @ p_body.T).T + np.array([x, y, altitude])
    keep = (p_world[:, 2] >= MIN_HEIGHT) & (p_world[:, 2] <= MAX_HEIGHT)
    return p_world[keep], rgb[ok][keep].astype(np.float32)


def open_rays(depth, rotation, x, y, altitude, height):
    """Where camera rays cross ``height`` without having hit anything above it.

    Returns (N, 2) world x, y. A ray whose depth is invalid (ground is past
    the 10 m far clip) or deeper than the point where it reaches ``height``
    saw no canopy above ``height`` at that spot. Counted against the canopy
    pixels there, this gives the fraction of the camera's view of a spot
    that is canopy, independent of how often the drone flew over it.
    """
    rays = (rotation @ (R_BC @ (R_CO @ RAY.reshape(-1, 3).T))).T      # per unit depth
    origin = rotation @ MOUNT_T + np.array([x, y, altitude])
    vz = rays[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        d_h = (height - origin[2]) / vz                            # depth at `height`
    d = depth.reshape(-1)
    passed = (vz < 0) & (d_h > MIN_DEPTH) & (~np.isfinite(d) | (d > d_h))
    return origin[:2] + rays[passed, :2] * d_h[passed, None]


def cell_counts(xy, res, origin_x, origin_y, dim_x, dim_y):
    """Per-cell counts of world points: ``(flat_index, count)`` on an [ix, iy] grid."""
    ix = ((xy[:, 0] - origin_x) / res).astype(np.int64)
    iy = ((xy[:, 1] - origin_y) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < dim_x) & (iy >= 0) & (iy < dim_y)
    cells, count = np.unique(ix[ok] * dim_y + iy[ok], return_counts=True)
    return cells, count.astype(np.float32)


def cell_colour(points, colours, res, origin_x, origin_y, dim_x, dim_y):
    """Per-cell pixel count and blue/green sum for one frame.

    Returns ``(flat_index, count, bg_sum)`` for the cells the frame hit;
    ``flat_index`` indexes an [ix, iy] grid of shape (dim_x, dim_y).
    """
    ix = ((points[:, 0] - origin_x) / res).astype(np.int64)
    iy = ((points[:, 1] - origin_y) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < dim_x) & (iy >= 0) & (iy < dim_y)
    flat = ix[ok] * dim_y + iy[ok]
    bg = colours[ok, 2] / np.maximum(colours[ok, 1], 1.0)
    cells, inverse = np.unique(flat, return_inverse=True)
    count = np.bincount(inverse).astype(np.float32)
    bg_sum = np.bincount(inverse, weights=bg).astype(np.float32)
    return cells, count, bg_sum
