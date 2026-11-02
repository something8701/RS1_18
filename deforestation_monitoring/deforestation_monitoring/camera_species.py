# camera_species.py: Place drone camera pixels on the canopy grid and score their colour
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
MIN_GREEN = 20

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
    ok = np.isfinite(depth) & (depth > MIN_DEPTH) & (depth < MAX_DEPTH)
    ok &= rgb[..., 1] >= MIN_GREEN
    p_opt = RAY[ok] * depth[ok][:, None]
    p_body = (R_BC @ (R_CO @ p_opt.T)).T + MOUNT_T
    p_world = (rotation @ p_body.T).T + np.array([x, y, altitude])
    keep = (p_world[:, 2] >= MIN_HEIGHT) & (p_world[:, 2] <= MAX_HEIGHT)
    return p_world[keep], rgb[ok][keep].astype(np.float32)


def open_rays(depth, rotation, x, y, altitude, height):
    rays = (rotation @ (R_BC @ (R_CO @ RAY.reshape(-1, 3).T))).T
    origin = rotation @ MOUNT_T + np.array([x, y, altitude])
    vz = rays[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        d_h = (height - origin[2]) / vz
    d = depth.reshape(-1)
    passed = (vz < 0) & (d_h > MIN_DEPTH) & (~np.isfinite(d) | (d > d_h))
    return origin[:2] + rays[passed, :2] * d_h[passed, None]


def plane_points(mask, rotation, x, y, altitude, height):
    rays = (rotation @ (R_BC @ (R_CO @ RAY[mask].T))).T
    origin = rotation @ MOUNT_T + np.array([x, y, altitude])
    vz = rays[:, 2]
    down = vz < -1e-6
    d_h = (height - origin[2]) / vz[down]
    return origin[:2] + rays[down, :2] * d_h[:, None]


def cell_counts(xy, res, origin_x, origin_y, dim_x, dim_y):
    ix = ((xy[:, 0] - origin_x) / res).astype(np.int64)
    iy = ((xy[:, 1] - origin_y) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < dim_x) & (iy >= 0) & (iy < dim_y)
    cells, count = np.unique(ix[ok] * dim_y + iy[ok], return_counts=True)
    return cells, count.astype(np.float32)


def cell_colour(points, colours, res, origin_x, origin_y, dim_x, dim_y):
    ix = ((points[:, 0] - origin_x) / res).astype(np.int64)
    iy = ((points[:, 1] - origin_y) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < dim_x) & (iy >= 0) & (iy < dim_y)
    flat = ix[ok] * dim_y + iy[ok]
    bg = colours[ok, 2] / np.maximum(colours[ok, 1], 1.0)
    cells, inverse = np.unique(flat, return_inverse=True)
    count = np.bincount(inverse).astype(np.float32)
    bg_sum = np.bincount(inverse, weights=bg).astype(np.float32)
    return cells, count, bg_sum
