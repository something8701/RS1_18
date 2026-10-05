#!/usr/bin/env python3
"""Individual tree detection (ITD) from a fused canopy height model (CHM).

This module implements the classic forestry ITD pipeline, adapted for this
project's two sources of height data:

  * the drone's 3D depth point cloud (``/drone_terrain``), and
  * the overall canopy height map (``/forest_canopy_map``).

Pipeline
--------
1. Fuse the two sources into one CHM. The point cloud supplies fine-grained
   heights where it has coverage; the canopy map fills the remaining scanned
   cells so segmentation works on the *complete* map, not just cloud hits.
2. Separate ground from canopy automatically with Otsu thresholding of the
   height distribution. This addresses "variations with the height": ground
   sits in a low mode near 0 m and vegetation sits in a higher mode.
3. Find treetops with a *height-dependent* search window (bigger trees get a
   bigger window). This keeps two closely spaced trees as separate maxima
   while still smoothing over the crown of one large tree.
4. Segment each crown from its treetop with a priority-flood watershed that
   stops at crown-height valleys and at a height-dependent crown radius.
   That is what splits clusters of adjacent trees instead of merging them.

Nothing here requires ROS, so the algorithm can be calibrated and tested
offline against the exact tree positions stored in the Gazebo world SDF.
"""

from __future__ import annotations

import heapq
import json
import math
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


# ── Small containers ────────────────────────────────────────────────

@dataclass
class DetectedTree:
    """One segmented tree crown."""
    id: int
    x: float
    y: float
    height: float
    area_m2: float
    radius_m: float
    label: int = 0      # crown label id in the segmentation grid
    peak_ix: int = -1   # grid cell of the crown's highest point
    peak_iy: int = -1

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DetectionParams:
    """Tunable parameters for the ITD pipeline."""
    chm_resolution: float = 0.2
    smooth_sigma: float = 0.15
    min_height: float = 2.0
    use_auto_threshold: bool = True
    ground_floor: float = 0.2
    ground_percentile: float = 0.10
    ground_margin: float = 1.0
    auto_threshold_cap: float = 4.0
    window_scale: float = 0.06
    window_offset: float = 0.25
    min_window: float = 0.3
    max_window: float = 1.2
    min_sep: float = 0.5
    saddle_ratio: float = 0.7
    valley_ratio: float = 0.25
    valley_offset: float = 0.4
    crown_scale: float = 1.5
    crown_offset: float = 2.0
    min_tree_height: float = 2.0
    min_crown_cells: int = 3
    mask_close_iterations: int = 2
    mask_open_iterations: int = 1
    pit_fill_size: int = 3
    centroid_band: float = 0.5
    # Tree position: "crown" = height-weighted centroid of the whole
    # (merged) crown; "band" = the same over the top centroid_band metres;
    # "auto" = crown for isolated crowns, band for crowns touching a
    # neighbour. Big oaks have off-centre tops, so the whole crown sits
    # closer to the trunk when its full footprint is known.
    position_mode: str = "crown"
    # A suppressed crown is merged into its suppressor only while the merged
    # extent (from the suppressor's peak) stays <= this; an oak with its
    # lobes reaches ~7.4 m, neighbouring trees merged together much more.
    # Pieces that would exceed it are dropped.
    merge_max_radius: float = 8.0
    # Two-scale treetops: smooth_sigma blurs narrow pine spires away. If
    # narrow_sigma > 0, local maxima of a lightly smoothed CHM (window
    # narrow_window) are added as markers where no broad marker lies within
    # min_sep; the raw-CHM saddle test still merges crown bumps back into
    # their tree.
    narrow_sigma: float = 0.0
    narrow_window: float = 1.0
    # A narrow peak must stand this far above the broad-smoothed surface:
    # a spire loses metres to heavy smoothing, a bump on an oak crown almost
    # nothing.
    narrow_prominence: float = 1.0
    # It must also show a real valley (raw-CHM saddle below saddle_ratio of
    # the lower peak) towards every broad marker within this range; oak
    # crowns reach beyond the normal saddle range.
    narrow_saddle_range: float = 8.0
    # ... and must not lie within narrow_exclusion_radius of a broad marker
    # at least narrow_tall_height high (a big oak crown, whose branch clumps
    # are as tall as its top). 0 = off.
    narrow_exclusion_radius: float = 0.0
    narrow_tall_height: float = 5.2
    # Watershed seeds. Markers are maxima of the smoothed CHM but crowns
    # flood the raw CHM, so a marker sitting in a gap between branch clumps
    # can be flooded by a taller neighbour first and two oaks become one
    # crown. seed_lock reserves each marker cell for its own crown;
    # seed_snap moves the seed to the raw maximum within this radius (m).
    seed_lock: bool = False
    seed_snap: float = 0.0
    # Overlap suppression treats a small crown inside a big crown's radius
    # as a lobe of it. If > 0, the small crown is kept when the raw CHM dips
    # below overlap_valley_ratio x the lower peak between the two peaks
    # (a real valley means a separate tree, e.g. a pine beside an oak).
    overlap_valley_ratio: float = 0.0
    # Surface for the "band" position: "smoothed" (the peak-finding surface)
    # or "raw". Smoothing flattens a pine spire below its neighbours' crown
    # edges, which moves a small crown's smoothed top band off its spire.
    band_surface: str = "smoothed"
    # With band_surface "raw": only crowns whose (smoothed) top is below
    # this height use the raw surface (spires); taller, broad crowns keep
    # the smoothed band. 0 = all crowns.
    band_raw_below: float = 0.0
    # Band width (m) on the raw surface; 0 = centroid_band. The raw top of a
    # spire is a single noisy LiDAR return, so a wider band averages it.
    band_raw_width: float = 0.0
    merge_small_area: float = 0.0
    merge_radius: float = 4.0
    overlap_threshold: float = 0.5

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "DetectionParams":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ── CHM construction and fusion ─────────────────────────────────────

def rasterize_points(
    points: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    dim_x: int,
    dim_y: int,
    reducer: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize (x, y, z) points into a height grid and a hit-count grid.

    Returns ``(height, hits)``. Cells without points keep height 0 and
    hits 0, so ``hits > 0`` is the "observed by the point cloud" mask.
    """
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ix = ((xs - origin_x) / resolution).astype(np.int64)
    iy = ((ys - origin_y) / resolution).astype(np.int64)
    valid = (ix >= 0) & (ix < dim_x) & (iy >= 0) & (iy < dim_y) \
        & np.isfinite(zs)
    ix, iy, zs = ix[valid], iy[valid], zs[valid]

    height = np.zeros((dim_x, dim_y), dtype=np.float32)
    hits = np.zeros((dim_x, dim_y), dtype=np.int32)
    if len(ix) == 0:
        return height, hits
    if reducer == "max":
        np.maximum.at(height, (ix, iy), zs)
    else:  # average
        np.add.at(height, (ix, iy), zs)
        np.add.at(hits, (ix, iy), 1)
        count = hits.astype(np.float32)
        height = np.divide(height, count, out=np.zeros_like(height),
                           where=count > 0)
        hits = (count > 0).astype(np.int32)
        return height, hits
    np.add.at(hits, (ix, iy), 1)
    return height, hits


def fuse_height_maps(
    cloud_height: np.ndarray,
    cloud_hits: np.ndarray,
    map_data: Optional[np.ndarray],
    map_scanned: Optional[np.ndarray],
    height_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse the point-cloud heights with the overall canopy map.

    Point-cloud cells win where they exist (unquantised, uncapped heights);
    the canopy map fills the remaining scanned cells. ``height_scale``
    converts the map's 0-100 encoding back to metres (10 = 0-10 m).
    Returns ``(fused_height, scanned_mask)``.
    """
    cloud_seen = cloud_hits > 0
    if map_data is None or map_scanned is None:
        return cloud_height.copy(), cloud_seen

    fused = np.where(cloud_seen, cloud_height,
                     np.where(map_scanned, map_data / height_scale, 0.0))
    scanned = cloud_seen | map_scanned
    return fused.astype(np.float32), scanned


def smooth_chm(chm: np.ndarray, scanned: np.ndarray,
               sigma_px: float) -> np.ndarray:
    """Gaussian-smooth the CHM while ignoring unscanned cells.

    Unknown (unscanned) cells must not be treated as ground: the smoothing
    kernel is renormalised by a smoothed version of the observed mask, so
    unknown cells contribute nothing instead of pulling nearby canopy down.
    """
    if sigma_px <= 0:
        return chm.copy()
    weights = ndimage.gaussian_filter(
        scanned.astype(np.float32), sigma=sigma_px, mode="constant")
    filled = np.where(scanned, chm, 0.0)
    smoothed = ndimage.gaussian_filter(filled, sigma=sigma_px, mode="constant")
    out = np.divide(smoothed, weights,
                    out=np.zeros_like(smoothed), where=weights > 1e-6)
    return np.where(scanned, out, 0.0).astype(np.float32)


# ── Ground / canopy separation ──────────────────────────────────────

def otsu_threshold(values: np.ndarray, bins: int = 256,
                   include_zero: bool = False) -> float:
    """Otsu's threshold over the histogram of height values (> 0, or >= 0
    with ``include_zero`` so exact-0 ground is part of the ground mode)."""
    keep = (values >= 0) if include_zero else (values > 0)
    values = values[np.isfinite(values) & keep]
    if values.size < 16:
        return 0.0
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return 0.0
    cum = np.cumsum(hist)
    cum_mean = np.cumsum(hist * centers)
    weight0 = cum
    weight1 = total - cum
    if np.any(weight0 == 0) or np.any(weight1 == 0):
        valid = (weight0 > 0) & (weight1 > 0)
        centers = centers[valid]
        weight0 = weight0[valid]
        weight1 = weight1[valid]
        cum_mean = cum_mean[valid]
        if centers.size == 0:
            return 0.0
    mean0 = cum_mean / weight0
    mean1 = (cum_mean[-1] - cum_mean) / weight1
    between = weight0 * weight1 * (mean0 - mean1) ** 2
    return float(centers[np.argmax(between)])


def canopy_mask_from_chm(
    chm: np.ndarray,
    scanned: np.ndarray,
    params: DetectionParams,
) -> Tuple[np.ndarray, float]:
    """Return the canopy mask and the effective ground/canopy threshold.

    The ground is *not* assumed to be exactly 0 m. Its reference height is
    estimated as a low percentile of the observed surface, and a cell only
    counts as canopy when it stands at least ``ground_margin`` above that
    reference and at least ``min_height`` in absolute terms. This keeps a
    slightly uneven ground (0.0-0.5 m relief plus sensor noise) from being
    labelled as trees.
    """
    # Ground and Otsu statistics use every scanned cell, including the
    # exact-0 m ground: without it, in dense forest the ground percentile
    # lands in the canopy and the threshold cuts off short pines.
    observed = chm[scanned]
    ground_ref = 0.0
    if observed.size >= 16:
        ground_ref = float(np.percentile(
            observed, params.ground_percentile * 100.0))
        # The ground mode must be *low*: if the percentile lands in canopy
        # (canopy cover > 1 - ground_percentile), fall back to the floor.
        if ground_ref > params.ground_floor + params.ground_margin:
            ground_ref = params.ground_floor
    threshold = max(float(params.min_height), ground_ref + params.ground_margin)
    if params.use_auto_threshold:
        if observed.size >= 16:
            # Otsu's valley can drift into the canopy tail on a smooth CHM,
            # which would erase short trees. Cap it so auto-thresholding can
            # only ever refine the ground boundary, never define "tree".
            otsu = float(otsu_threshold(observed, include_zero=True))
            threshold = max(threshold, min(otsu, params.auto_threshold_cap))
    mask = scanned & (chm >= threshold)
    return mask, threshold


def refine_canopy_mask(
    canopy: np.ndarray,
    params: DetectionParams,
) -> np.ndarray:
    """Clean the tree/non-tree mask before crown localisation.

    Mirrors the graph-cuts refinement step in Yang et al. (2009): a noisy
    surface can shatter one crown into many disconnected cells. A small
    morphological closing reconnects those fragments, and components smaller
    than a crown are dropped. Individual crown localisation then runs on one
    connected region per tree instead of on speckles.
    """
    mask = canopy.copy()
    # Opening first removes 1-cell-wide edge-ray streaks while keeping the
    # compact body of a crown. The closing pass afterwards reconnects small
    # gaps inside the crown.
    if params.mask_open_iterations > 0:
        mask = ndimage.binary_opening(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(params.mask_open_iterations))
    if params.mask_close_iterations > 0:
        mask = ndimage.binary_closing(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(params.mask_close_iterations))
    labels, num = ndimage.label(mask)
    if num:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, num + 1))
        keep = sizes >= params.min_crown_cells
        remove = np.arange(1, num + 1)[~keep]
        for label_id in remove:
            mask[labels == label_id] = False
    return mask


# ── Treetop detection ───────────────────────────────────────────────

def treetop_markers(
    chm: np.ndarray,
    canopy: np.ndarray,
    params: DetectionParams,
    raw_chm: Optional[np.ndarray] = None,
    narrow_chm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find treetops as local maxima using a height-dependent window.

    Returns an ``(N, 2)`` array of ``(ix, iy)`` marker cells. A small window
    is used so neighbouring crowns can produce separate maxima; a saddle
    test then re-merges the spurious double peaks that a small window can
    create on one noisy crown. Peaks are found on the (smoothed) ``chm``,
    but valley depth is measured on ``raw_chm`` so smoothing cannot fill a
    narrow gap between two closely spaced trees.
    """
    res = params.chm_resolution
    heights = np.where(canopy, chm, 0.0)

    window_m = np.clip(
        params.window_scale * heights + params.window_offset,
        params.min_window,
        params.max_window,
    )
    radius_px = np.ceil(window_m / res).astype(np.int32)
    radius_px = np.clip(radius_px, 1, 32)

    candidates = np.zeros(chm.shape, dtype=bool)
    for scale in np.unique(radius_px):
        size = int(2 * scale + 1)
        local_max = ndimage.maximum_filter(
            heights, size=size, mode="constant")
        group = radius_px == scale
        candidates |= group & (heights == local_max)
    candidates &= canopy
    if not np.any(candidates):
        return np.zeros((0, 2), dtype=np.int32)

    # Non-maximum suppression: keep the tallest peak when two candidate
    # maxima sit closer than min_sep.
    idx = np.argwhere(candidates)
    order = np.argsort(-heights[idx[:, 0], idx[:, 1]])
    kept: List[np.ndarray] = []
    min_sep_px = max(1.0, params.min_sep / res)
    for i in idx[order]:
        if all(math.hypot(i[0] - k[0], i[1] - k[1]) >= min_sep_px
               for k in kept):
            kept.append(i)
    if narrow_chm is not None:
        kept = _add_narrow_markers(kept, narrow_chm, heights, canopy,
                                   params, min_sep_px,
                                   chm if raw_chm is None else raw_chm)
    markers = np.asarray(kept, dtype=np.int32).reshape(-1, 2)
    return _merge_markers_by_saddle(
        chm if raw_chm is None else raw_chm, markers, params)


def _peak(chm, k, r=2):
    """Max of ``chm`` within r cells of marker ``k`` (its local peak)."""
    return float(chm[max(0, k[0] - r):k[0] + r + 1,
                     max(0, k[1] - r):k[1] + r + 1].max())


def _add_narrow_markers(kept, narrow_chm, broad, canopy, params, min_sep_px,
                        raw_chm=None):
    """Add prominent local maxima of the lightly smoothed CHM that are at
    least ``min_sep`` from every existing (broad) marker. Never replaces
    one. Prominence = narrow height - broad-smoothed height."""
    res = params.chm_resolution
    hn = np.where(canopy, narrow_chm, 0.0)
    size = int(2 * max(1, int(math.ceil(params.narrow_window / res))) + 1)
    local_max = ndimage.maximum_filter(hn, size=size, mode="constant")
    cand = (canopy & (hn == local_max) & (hn >= params.min_tree_height)
            & (hn - broad >= params.narrow_prominence))
    idx = np.argwhere(cand)
    if len(idx) == 0:
        return kept
    out = list(kept)
    range_px = params.narrow_saddle_range / res
    for i in idx[np.argsort(-hn[idx[:, 0], idx[:, 1]])]:
        if not all(math.hypot(i[0] - k[0], i[1] - k[1]) >= min_sep_px
                   for k in out):
            continue
        if raw_chm is not None:
            hi = float(raw_chm[i[0], i[1]])
            if params.narrow_exclusion_radius > 0 and any(
                    math.hypot(i[0] - k[0], i[1] - k[1])
                    < params.narrow_exclusion_radius / res
                    and _peak(raw_chm, k) >= params.narrow_tall_height
                    for k in kept):
                continue
            same_crown = False
            for k in kept:              # broad markers only
                if math.hypot(i[0] - k[0], i[1] - k[1]) > range_px:
                    continue
                hk = float(raw_chm[k[0], k[1]])
                if _line_min(raw_chm, k, i) >= params.saddle_ratio * min(hi, hk):
                    same_crown = True
                    break
            if same_crown:
                continue
        out.append(i)
    return out


def _line_min(chm: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Minimum CHM height along the straight line between two markers."""
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(math.ceil(dist)))
    ts = np.linspace(0.0, 1.0, n)
    rows = np.round(a[0] + ts * (b[0] - a[0])).astype(np.int32)
    cols = np.round(a[1] + ts * (b[1] - a[1])).astype(np.int32)
    rows = np.clip(rows, 0, chm.shape[0] - 1)
    cols = np.clip(cols, 0, chm.shape[1] - 1)
    return float(np.min(chm[rows, cols]))


def _merge_markers_by_saddle(
    chm: np.ndarray,
    markers: np.ndarray,
    params: DetectionParams,
) -> np.ndarray:
    """Merge adjacent markers whose intervening valley is too shallow.

    Two peaks are the same tree when the saddle between them is still a large
    fraction of the shorter peak; they are distinct trees when the surface
    dips meaningfully between them.
    """
    if len(markers) <= 1:
        return markers
    heights = chm[markers[:, 0], markers[:, 1]]
    order = np.argsort(-heights)
    max_dist_px = 2.0 * params.max_window / params.chm_resolution

    kept = [markers[order[0]]]
    for idx in order[1:]:
        cand = markers[idx]
        hc = float(heights[idx])
        merged = False
        for k in kept:
            hk = float(chm[k[0], k[1]])
            if math.hypot(cand[0] - k[0], cand[1] - k[1]) > max_dist_px:
                continue
            saddle = _line_min(chm, k, cand)
            if saddle >= params.saddle_ratio * min(hk, hc):
                merged = True
                break
        if not merged:
            kept.append(cand)
    return np.asarray(kept, dtype=np.int32).reshape(-1, 2)


# ── Crown segmentation ──────────────────────────────────────────────

def segment_crowns(
    chm: np.ndarray,
    canopy: np.ndarray,
    markers: np.ndarray,
    params: DetectionParams,
    flood_chm: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Marker-controlled watershed from treetop markers.

    All markers flood simultaneously on the inverted height surface, so each
    crown grows into whatever shape the surface supports and two fronts meet
    at the saddle between trees. There is no circular radius assumption: a
    crown can be big, lop-sided or have a branch out to one side. The only
    bounds are the canopy mask (ground), a very low relative height floor
    (so a tall crown cannot pour down into a distant short tree), and a
    generous height-dependent sanity radius.

    Returns ``(labels, treetop_cells)`` where labels is an int grid and
    treetop_cells maps label id -> the marker (ix, iy).
    """
    res = params.chm_resolution
    heights = chm if flood_chm is None else flood_chm
    labels = np.zeros(chm.shape, dtype=np.int32)
    treetop_cells: Dict[int, np.ndarray] = {}

    if len(markers) == 0:
        return labels, treetop_cells

    if params.seed_snap > 0:
        # Seed each crown at the raw top next to its (smoothed) marker.
        k = max(1, int(round(params.seed_snap / res)))
        snapped = []
        for mi, mj in markers:
            x0, x1 = max(0, mi - k), min(chm.shape[0], mi + k + 1)
            y0, y1 = max(0, mj - k), min(chm.shape[1], mj + k + 1)
            sub = np.where(canopy[x0:x1, y0:y1], heights[x0:x1, y0:y1], -1.0)
            a, b = np.unravel_index(int(np.argmax(sub)), sub.shape)
            snapped.append((x0 + a, y0 + b) if sub[a, b] > heights[mi, mj]
                           else (mi, mj))
        markers = np.asarray(snapped, dtype=np.int32).reshape(-1, 2)
    order = np.argsort(-heights[markers[:, 0], markers[:, 1]])
    reserved = np.zeros(chm.shape, dtype=np.int32)
    info: Dict[int, Tuple[int, int, int, float]] = {}
    heap: List[Tuple[float, int, int, int]] = []
    for label_id, idx in enumerate(order, start=1):
        mi, mj = int(markers[idx][0]), int(markers[idx][1])
        if params.seed_lock and reserved[mi, mj] == 0:
            reserved[mi, mj] = label_id
        peak_h = float(heights[mi, mj])
        crown_m = params.crown_scale * peak_h + params.crown_offset
        crown_px = max(1, int(math.ceil(crown_m / res)))
        # No per-crown height drop: the watershed fronts themselves define
        # the boundary at the saddle. Only the canopy threshold stops a
        # crown from spreading onto ground.
        floor_h = params.min_height
        info[label_id] = (mi, mj, crown_px * crown_px, floor_h)
        heapq.heappush(heap, (-peak_h, mi, mj, label_id))

    while heap:
        _, i, j, label_id = heapq.heappop(heap)
        if labels[i, j] != 0:
            continue
        if reserved[i, j] not in (0, label_id):
            continue      # another crown's seed cell
        labels[i, j] = label_id
        mi, mj, crown_px2, floor_h = info[label_id]
        for ni in range(max(0, i - 1), min(chm.shape[0], i + 2)):
            for nj in range(max(0, j - 1), min(chm.shape[1], j + 2)):
                if ni == i and nj == j:
                    continue
                if not canopy[ni, nj] or labels[ni, nj] != 0:
                    continue
                if heights[ni, nj] < floor_h:
                    continue
                if (ni - mi) ** 2 + (nj - mj) ** 2 > crown_px2:
                    continue
                heapq.heappush(
                    heap, (-float(heights[ni, nj]), ni, nj, label_id))

    for label_id, (mi, mj, _, _) in info.items():
        treetop_cells[label_id] = np.array([mi, mj], dtype=np.int32)

    return labels, treetop_cells


def trees_from_labels(
    labels: np.ndarray,
    chm: np.ndarray,
    params: DetectionParams,
    origin_x: float,
    origin_y: float,
    position_mode: Optional[str] = None,
    band_chm: Optional[np.ndarray] = None,
) -> List[DetectedTree]:
    """Convert a label grid into DetectedTree objects.

    ``band_chm`` (optional) is the surface used for the top-band position;
    height, radius and peak always come from ``chm``.
    """
    res = params.chm_resolution
    out: List[DetectedTree] = []
    touching = _touching_labels(labels)
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        cells = labels == label_id
        n = int(np.count_nonzero(cells))
        if n < params.min_crown_cells:
            continue
        heights = chm[cells]
        height = float(np.max(heights))
        if height < params.min_tree_height:
            continue
        # Grid is indexed [ix, iy] = [x, y], so rows carry x and columns y.
        rows, cols = np.nonzero(cells)
        mode = position_mode or params.position_mode
        if mode == "auto":
            # An isolated crown's full footprint is known, so its centroid
            # sits over the trunk; a crown cut by the watershed against a
            # neighbour has a truncated footprint, so use its top band.
            mode = "band" if label_id in touching else "crown"
        # "band" is stable from tick to tick but follows the highest lobe of
        # a big crown; "crown" sits over the trunk for broad, lobed crowns.
        use_raw = band_chm is not None and (
            params.band_raw_below <= 0 or height < params.band_raw_below)
        bh = band_chm[cells] if use_raw else heights
        btop = float(np.max(bh))
        width = (params.band_raw_width if use_raw and params.band_raw_width > 0
                 else params.centroid_band)
        band = bh >= btop - max(width, 0.0)
        if mode == "crown":
            wx = float(np.average(rows, weights=heights))
            wy = float(np.average(cols, weights=heights))
        elif np.any(band):
            weights = np.clip(bh[band] - (btop - max(width, 0.0)), 0.05, None)
            wx = float(np.average(rows[band], weights=weights))
            wy = float(np.average(cols[band], weights=weights))
        else:
            peak_flat = int(np.argmax(heights))
            wx = float(rows[peak_flat])
            wy = float(cols[peak_flat])
        # Crown radius is the true distance to the furthest crown cell from
        # the treetop, not the radius of a circle with the same area. A crown
        # can be big and lop-sided (a branch sticking out one side).
        peak_flat = int(np.argmax(heights))  # heights is the 1D crown values
        peak_row, peak_col = rows[peak_flat], cols[peak_flat]
        extent = np.hypot(rows - peak_row, cols - peak_col)
        radius_m = float(np.max(extent) * res)
        area_m2 = float(n * res * res)
        out.append(DetectedTree(
            id=0,
            x=origin_x + (wx + 0.5) * res,
            y=origin_y + (wy + 0.5) * res,
            height=height,
            area_m2=area_m2,
            radius_m=radius_m,
            label=int(label_id),
            peak_ix=int(peak_row),
            peak_iy=int(peak_col),
        ))
    # Stable sort by descending height (taller trees are the least ambiguous).
    out.sort(key=lambda t: -t.height)
    for i, tree in enumerate(out, start=1):
        tree.id = i
    return out


def _touching_labels(labels: np.ndarray) -> set:
    """Label ids whose crown shares an edge with a different crown."""
    out = set()
    for a, b in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):
        m = (a != 0) & (b != 0) & (a != b)
        out.update(np.unique(a[m]).tolist())
        out.update(np.unique(b[m]).tolist())
    return out


def merge_small_into_nearby(
    detections: List[DetectedTree],
    small_area_m2: float,
    merge_radius: float,
) -> List[DetectedTree]:
    """Absorb small crown fragments that sit next to a large crown.

    A noisy surface can split one big tree into its main crown plus small
    fringe pieces. Small detections near a larger tree are dropped; small
    detections standing alone (e.g. a genuinely small pine) are kept.
    """
    if small_area_m2 <= 0 or not detections:
        return detections
    # Tallest first. A candidate inside the merge radius of a taller crown
    # that is very small, or much smaller than that crown, is a fringe
    # fragment and is absorbed.
    ordered = sorted(detections, key=lambda t: -t.height)
    kept: List[DetectedTree] = []
    for candidate in ordered:
        absorbed = False
        for taller in kept:
            if math.hypot(candidate.x - taller.x, candidate.y - taller.y) > merge_radius:
                continue
            if candidate.area_m2 < small_area_m2 or \
                    taller.area_m2 >= 2.0 * candidate.area_m2:
                absorbed = True
                break
        if not absorbed:
            kept.append(candidate)
    kept.sort(key=lambda t: -t.height)
    return kept


def greedy_overlap_select(
    detections: List[DetectedTree],
    overlap_threshold: float,
    owner: Optional[Dict[int, int]] = None,
    valley_chm: Optional[np.ndarray] = None,
    valley_ratio: float = 0.0,
) -> List[DetectedTree]:
    """Greedy overlap suppression, adapted from Yang et al. (2009).

    Candidates are kept tallest-first and a candidate is dropped when its
    crown overlaps a taller, already-kept crown by more than
    ``overlap_threshold`` of the smaller radius:

        overlap = (R_i + R_j - distance(C_i, C_j)) / min(R_i, R_j)

    With a high threshold (default 2.0) this only removes fragments whose
    crown is essentially contained inside a bigger crown, so two separate
    trees that merely touch are preserved.

    If ``owner`` is given it is filled with ``{suppressed label: label of
    the crown that suppressed it}`` so the caller can merge the cells.

    With ``valley_chm`` and ``valley_ratio`` > 0, an overlapping crown is
    still kept when the surface between the two peaks dips below
    ``valley_ratio`` x the lower peak (a separate tree, not a lobe).
    """
    if overlap_threshold <= 0 or len(detections) < 2:
        return detections
    ordered = sorted(detections, key=lambda d: -d.height)
    kept: List[DetectedTree] = []
    for candidate in ordered:
        suppressed = False
        for taller in kept:
            distance = math.hypot(
                candidate.x - taller.x, candidate.y - taller.y)
            smaller_r = min(candidate.radius_m, taller.radius_m)
            if smaller_r <= 0.0:
                if distance < 0.5:
                    suppressed = True
                    break
                continue
            overlap = (candidate.radius_m + taller.radius_m - distance) \
                / smaller_r
            if overlap > overlap_threshold:
                if (valley_chm is not None and valley_ratio > 0
                        and candidate.peak_ix >= 0 and taller.peak_ix >= 0):
                    a = np.array([candidate.peak_ix, candidate.peak_iy])
                    b = np.array([taller.peak_ix, taller.peak_iy])
                    low = min(float(valley_chm[a[0], a[1]]),
                              float(valley_chm[b[0], b[1]]))
                    if _line_min(valley_chm, a, b) < valley_ratio * low:
                        continue
                suppressed = True
                break
        if suppressed and owner is not None:
            owner[candidate.label] = taller.label
        if not suppressed:
            kept.append(candidate)
    kept.sort(key=lambda d: -d.height)
    return kept


def select_crowns(
    detections: List[DetectedTree],
    params: DetectionParams,
) -> List[DetectedTree]:
    """Final crown selection: overlap suppression, then small-fragment merge.

    Overlap suppression runs first so a secondary lobe of a big crown is
    removed as contained in its parent; merging first can let such a lobe
    absorb a real small neighbour through ``merge_radius``.
    """
    dets = greedy_overlap_select(detections, params.overlap_threshold)
    return merge_small_into_nearby(
        dets, params.merge_small_area, params.merge_radius)


def detect_trees(
    chm: np.ndarray,
    scanned: np.ndarray,
    params: DetectionParams,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    return_debug: bool = False,
):
    """Run the full pipeline: threshold, treetops, crown segmentation.

    With ``return_debug=True`` also returns ``(labels, canopy_mask)`` so a
    caller can publish the crown segmentation for visual verification.
    """
    smooth_px = params.smooth_sigma / params.chm_resolution
    chm_s = smooth_chm(chm, scanned, smooth_px)
    if params.pit_fill_size > 0:
        # Fill pits inside crowns before treetop detection: a noisy dropout
        # inside a crown must not become a false local minimum that splits
        # one tree. Grey closing raises pits without touching the crown tops.
        size = max(1, int(params.pit_fill_size))
        chm_s = ndimage.grey_closing(chm_s, size=size)
        chm_s = np.where(scanned, chm_s, 0.0).astype(np.float32)
    canopy, _ = canopy_mask_from_chm(chm_s, scanned, params)
    canopy = refine_canopy_mask(canopy, params)
    narrow = None
    if params.narrow_sigma > 0:
        narrow = smooth_chm(chm, scanned,
                            params.narrow_sigma / params.chm_resolution)
    markers = treetop_markers(chm_s, canopy, params, raw_chm=chm,
                              narrow_chm=narrow)
    # Peaks are found on the smoothed surface, but crowns flood on the raw
    # surface so a narrow valley between two trees is never filled in.
    labels, _ = segment_crowns(chm_s, canopy, markers, params, flood_chm=chm)
    # Overlap suppression on treetop geometry ("band" positions). A crown
    # suppressed as contained in a taller one is a lobe of the same tree:
    # merge its cells into that crown instead of dropping them, so the
    # tree's extent and position come from the whole crown.
    dets = trees_from_labels(labels, chm_s, params, origin_x, origin_y,
                             position_mode="band")
    owner: Dict[int, int] = {}
    greedy_overlap_select(dets, params.overlap_threshold, owner=owner,
                          valley_chm=chm,
                          valley_ratio=params.overlap_valley_ratio)
    if owner:
        labels = labels.copy()
        res = params.chm_resolution
        for lobe, parent in owner.items():   # tallest suppressed first
            while parent in owner:
                parent = owner[parent]
            union = (labels == parent) | (labels == lobe)
            rows, cols = np.nonzero(union)
            if len(rows) == 0:
                continue
            peak = int(np.argmax(chm_s[union]))
            extent = float(np.max(np.hypot(
                rows - rows[peak], cols - cols[peak]))) * res
            # Merge a lobe of the same crown; drop a piece that would turn
            # the crown into an implausible multi-tree blob.
            labels[labels == lobe] = (
                parent if extent <= params.merge_max_radius else 0)
    dets = trees_from_labels(
        labels, chm_s, params, origin_x, origin_y,
        band_chm=chm if params.band_surface == "raw" else None)
    dets = merge_small_into_nearby(
        dets, params.merge_small_area, params.merge_radius)
    if return_debug:
        return dets, labels, canopy
    return dets


# ── Ground truth from SDF ───────────────────────────────────────────

def parse_tree_truth(world_sdf_path: str) -> List[Tuple[str, float, float]]:
    """Extract ``(name, x, y)`` for every fuel tree include in a world SDF."""
    tree = ET.parse(world_sdf_path)
    trees: List[Tuple[str, float, float]] = []
    for inc in tree.getroot().iter("include"):
        uri = (inc.findtext("uri", default="") or "").lower()
        name = inc.findtext("name", default="") or ""
        pose = inc.findtext("pose", default="") or ""
        if "tree" not in uri and "tree" not in name.lower():
            continue
        parts = pose.split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        trees.append((name, x, y))
    return trees


# ── Synthetic scenarios (offline calibration / tests) ───────────────

def deterministic_height(name: str, rng_seed: int = 7) -> float:
    """Deterministic pseudo-height in [3.0, 11.0] m from a tree name."""
    # zlib.crc32 is stable across processes, unlike Python's salted hash().
    h = zlib.crc32(name.encode("utf-8")) % 1000 / 1000.0
    return 3.0 + h * 8.0


def synthetic_chm(
    truth: Sequence[Tuple[str, float, float]],
    resolution: float,
    origin_x: float,
    origin_y: float,
    dim_x: int,
    dim_y: int,
    crown_factor: float = 0.16,
    crown_offset: float = 0.40,
    noise_sigma: float = 0.04,
    ground_relief: float = 0.4,
    rng_seed: int = 7,
) -> np.ndarray:
    """Build a synthetic CHM from ground-truth trunk positions.

    Each tree contributes a Gaussian crown bump whose full radius scales with
    height (``sigma = (crown_factor * height + crown_offset) / 3``), and the
    surface is the per-cell maximum (the canopy envelope). A smooth random
    ground relief of up to ``ground_relief`` metres is added so calibration
    learns that ground is not a perfect zero plane.
    This reproduces both isolated trees and merged clusters of close trees.
    """
    rng = np.random.default_rng(rng_seed)
    grid = np.zeros((dim_x, dim_y), dtype=np.float32)
    xs = origin_x + (np.arange(dim_x) + 0.5) * resolution
    ys = origin_y + (np.arange(dim_y) + 0.5) * resolution
    # indexing="ij" makes grid[ix, iy] = (x, y), matching the detector.
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    # Smooth low-frequency ground relief in [0, ground_relief].
    relief = ndimage.gaussian_filter(
        rng.standard_normal((dim_x, dim_y)), sigma=max(1.0, 2.0 / resolution))
    relief -= relief.min()
    relief = relief / max(relief.max(), 1e-9) * ground_relief

    for name, tx, ty in truth:
        h = deterministic_height(name)
        crown_radius = crown_factor * h + crown_offset
        sigma = crown_radius / 3.0
        bump = h * np.exp(-((gx - tx) ** 2 + (gy - ty) ** 2) / (2 * sigma ** 2))
        np.maximum(grid, bump, out=grid)

    grid = relief.astype(np.float32) + grid

    if noise_sigma > 0:
        grid += rng.normal(0.0, noise_sigma, size=grid.shape).astype(np.float32)
        grid = np.maximum(grid, 0.0)
    return grid


def scenario_layouts() -> Dict[str, List[Tuple[str, float, float]]]:
    """Named cluster scenarios used to calibrate and test the detector."""
    def make(names: Iterable[str], pts: Iterable[Tuple[float, float]]):
        return list(zip(names, *zip(*pts)))  # -> [(name, x, y), ...]

    return {
        "isolated_5": make(
            (f"t{i}" for i in range(1, 6)),
            [(-8, -8), (8, -8), (-8, 8), (8, 8), (0, 0)]),
        "pair_1.2m": make(
            ("a", "b"), [(0.0, 0.0), (1.2, 0.0)]),
        "pair_1.8m": make(
            ("a", "b"), [(0.0, 0.0), (1.8, 0.0)]),
        "pair_2.5m": make(
            ("a", "b"), [(0.0, 0.0), (2.5, 0.0)]),
        "triplet_1.5m": make(
            ("a", "b", "c"), [(0.0, 0.0), (1.5, 0.0), (0.75, 1.3)]),
        "quad_1.3m": make(
            ("a", "b", "c", "d"),
            [(0.0, 0.0), (1.3, 0.0), (0.0, 1.3), (1.3, 1.3)]),
        "mixed": make(
            ("a", "b", "c", "d", "e", "f", "g", "h"),
            [(-10, -10), (-4, -4), (-2.4, -4), (4, -4), (5.5, -4),
             (4, 4), (5.3, 5.3), (6.6, 5.3)]),
    }


# ── Matching and scoring ────────────────────────────────────────────

def match_detections(
    detections: Sequence[DetectedTree],
    truth: Sequence[Tuple[str, float, float]],
    match_radius: float,
) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    """Greedy one-to-one matching. Returns (tp, fp, fn, matched pairs)."""
    matched_truth = [False] * len(truth)
    pairs: List[Tuple[int, int, float]] = []
    tp = 0
    for di, det in enumerate(detections):
        best_i, best_d = None, match_radius
        for ti, (_, tx, ty) in enumerate(truth):
            if matched_truth[ti]:
                continue
            d = math.hypot(det.x - tx, det.y - ty)
            if d <= best_d:
                best_i, best_d = ti, d
        if best_i is not None:
            matched_truth[best_i] = True
            tp += 1
            pairs.append((di, best_i, best_d))
    fp = len(detections) - tp
    fn = sum(1 for m in matched_truth if not m)
    return tp, fp, fn, pairs


def score_detections(
    detections: Sequence[DetectedTree],
    truth: Sequence[Tuple[str, float, float]],
    match_radius: float,
) -> Dict:
    """Precision / recall / F1 for a detection set against ground truth."""
    tp, fp, fn, _ = match_detections(detections, truth, match_radius)
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "detected": len(detections),
        "truth": len(truth),
        "true_positives": tp,
        "false_positives": fp,
        "missed": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ── Automatic calibration ───────────────────────────────────────────

def _sample_param(rng: np.random.Generator, name: str, values: Sequence):
    return float(rng.choice(np.asarray(values, dtype=np.float64))) \
        if name not in ("use_auto_threshold", "min_crown_cells") \
        else rng.choice(values)


def _random_config(rng: np.random.Generator) -> DetectionParams:
    grid = {
        "chm_resolution": (0.2,),
        "smooth_sigma": (0.1, 0.15, 0.25, 0.35, 0.5),
        "min_height": (1.5, 2.0, 2.5),
        "use_auto_threshold": (True,),
        "ground_floor": (0.1, 0.2, 0.3),
        "ground_percentile": (0.05, 0.10, 0.20),
        "ground_margin": (0.5, 1.0, 1.5),
        "auto_threshold_cap": (3.0, 4.0, 5.0),
        "window_scale": (0.04, 0.06, 0.08, 0.10),
        "window_offset": (0.2, 0.25, 0.3),
        "min_window": (0.3, 0.4),
        "max_window": (0.9, 1.2, 1.5),
        "min_sep": (0.4, 0.5, 0.6),
        "saddle_ratio": (0.5, 0.6, 0.7),
        "valley_ratio": (0.15, 0.25, 0.35),
        "valley_offset": (0.3, 0.4, 0.6),
        "crown_scale": (0.20, 0.25, 0.30),
        "crown_offset": (0.3, 0.4, 0.5),
        "min_tree_height": (1.8, 2.0, 2.5),
        "min_crown_cells": (2, 3, 4),
    }
    values = {}
    for key, options in grid.items():
        sampled = _sample_param(rng, key, options)
        if key == "use_auto_threshold":
            values[key] = bool(sampled)
        elif key == "min_crown_cells":
            values[key] = int(sampled)
        else:
            values[key] = float(sampled)
    return DetectionParams.from_dict(values)


def calibrate(
    chm: np.ndarray,
    scanned: np.ndarray,
    truth: Sequence[Tuple[str, float, float]],
    origin_x: float,
    origin_y: float,
    match_radius: float = 1.0,
    n_trials: int = 60,
    seed: int = 7,
) -> Dict:
    """Random-search the parameter space, scoring F1 against SDF truth."""
    rng = np.random.default_rng(seed)
    best: Dict = {"params": None, "metrics": None}
    results: List[Dict] = []
    for _ in range(n_trials):
        config = _random_config(rng)
        try:
            dets = detect_trees(chm, scanned, config, origin_x, origin_y)
            metrics = score_detections(dets, truth, match_radius)
        except Exception:
            continue
        results.append({"params": config.to_dict(), "metrics": metrics})
        f1 = metrics["f1"]
        if best["metrics"] is None or f1 > best["metrics"]["f1"]:
            best = {"params": config.to_dict(), "metrics": metrics}

    results.sort(key=lambda r: -r["metrics"]["f1"])
    return {
        "best": best,
        "all_results": results,
        "n_trials": n_trials,
        "match_radius": match_radius,
    }


def params_to_ros_yaml(params: DetectionParams, node_name: str = "/**") -> str:
    """Serialize calibrated params as a ROS 2 parameter file (YAML)."""
    lines = [f"{node_name}:", "  ros__parameters:"]
    for key, value in params.to_dict().items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, int):
            rendered = str(value)
        elif isinstance(value, str):
            rendered = json.dumps(value)   # quoted YAML string
        else:
            rendered = repr(float(value))
        lines.append(f"    {key}: {rendered}")
    return "\n".join(lines) + "\n"


def save_calibration_json(path: str, calibration_result: Dict) -> None:
    with open(path, "w") as fh:
        json.dump(calibration_result, fh, indent=2)
