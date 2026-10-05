#!/usr/bin/env python3
"""
Pattern Scanner Node: turns deforestation signatures into SuspiciousArea
flags for the mission coordinator (the drone to Husky link).

Two detection modes (parameter `detection_mode`):

  change (default): connected regions of lost canopy on /canopy_change_map
      (scan_mapper's diff against its baseline; -100 = lost) become
      'CLEARING' flags. Height-map noise does not trigger them.

  height: gap / anomaly / line heuristics on /forest_canopy_map. The launch
      files keep them off (gap_threshold 0, anomaly_height_drop 60,
      line_min_cells 1000): on the height map any positive gap threshold
      floods flags.

Flags are deduplicated spatially (grid buckets) and temporally (time + radius
window) so the queue cannot flood. Markers are published for RViz.
"""

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from deforestation_interfaces.msg import SuspiciousArea
from scipy import ndimage
from collections import deque
import math


class PatternScanner(Node):
    """Detects deforestation signatures (canopy loss / gaps / anomalies)."""

    def __init__(self):
        super().__init__('pattern_scanner')

        # --- Detection mode ---
        self.declare_parameter('detection_mode', 'change',
            descriptor=ParameterDescriptor(
                description="Detection source: 'change' (default) — canopy-loss "
                            "regions on /canopy_change_map; 'height' — legacy "
                            "gap/anomaly/line heuristics on /forest_canopy_map "
                            "(needs retuning for height maps)"))
        # --- Change-mode parameters ---
        self.declare_parameter('lost_value_threshold', -50.0,
            descriptor=ParameterDescriptor(
                description='Change-map value below which a cell counts as '
                            'canopy LOST (-100 = lost, 0 = unchanged)'))
        self.declare_parameter('min_lost_cells', 8,
            descriptor=ParameterDescriptor(
                description='Minimum connected lost cells to flag a clearing'))
        self.declare_parameter('max_change_flags', 5,
            descriptor=ParameterDescriptor(
                description='Maximum CLEARING flags per scan tick'))
        # --- Height-mode parameters (legacy, gated in launches) ---
        self.declare_parameter('gap_threshold', 15,
            descriptor=ParameterDescriptor(
                description='Occupancy value (0-100) below which cells are considered gaps'))
        self.declare_parameter('min_gap_area', 10,
            descriptor=ParameterDescriptor(
                description='Minimum number of cells to flag as a gap'))
        self.declare_parameter('anomaly_height_drop', 8.0,
            descriptor=ParameterDescriptor(
                description='Height drop in metres that triggers anomaly detection'))
        self.declare_parameter('scan_rate', 0.5,
            descriptor=ParameterDescriptor(
                description='Detection scan rate in Hz'))
        self.declare_parameter('line_min_cells', 20,
            descriptor=ParameterDescriptor(
                description='Minimum connected edge cells to flag as a linear feature'))
        self.declare_parameter('detection_frame', 'parrot1_odom',
            descriptor=ParameterDescriptor(
                description='Frame to publish suspicious areas in — must match Nav2 map frame'))
        self.declare_parameter('dedup_radius', 3.0,
            descriptor=ParameterDescriptor(
                description='Minimum distance (m) between flags to avoid duplicates'))
        self.declare_parameter('dedup_window', 300.0,
            descriptor=ParameterDescriptor(
                description='Time window (s) for temporal dedup — skip flags within '
                'dedup_radius of a recent flag published within this window'))

        self.detection_mode = self.get_parameter('detection_mode').value
        if self.detection_mode not in ('change', 'height'):
            self.get_logger().warn(
                f'Unknown detection_mode "{self.detection_mode}" — using "change".')
            self.detection_mode = 'change'
        self.lost_threshold = self.get_parameter('lost_value_threshold').value
        self.min_lost_cells = self.get_parameter('min_lost_cells').value
        self.max_change_flags = self.get_parameter('max_change_flags').value
        self.gap_threshold = self.get_parameter('gap_threshold').value
        self.min_gap_area = self.get_parameter('min_gap_area').value
        self.anomaly_drop = self.get_parameter('anomaly_height_drop').value
        self.line_min_cells = self.get_parameter('line_min_cells').value
        self.detection_frame = self.get_parameter('detection_frame').value
        self.dedup_radius = self.get_parameter('dedup_radius').value
        self.dedup_window = self.get_parameter('dedup_window').value

        # Internal state
        self.latest_map = None          # /forest_canopy_map (height mode)
        self.latest_change = None       # /canopy_change_map (change mode)
        self.resolution = None
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.flag_counter = 0
        self.flagged_cells = set()
        # Temporal dedup: (x, y, timestamp) for recent flags
        self._recent_flags = deque(maxlen=500)

        # -- QoS profiles --
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        alert_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/forest_canopy_map', self.map_callback, map_qos
        )
        self.change_sub = self.create_subscription(
            OccupancyGrid, '/canopy_change_map', self.change_callback, map_qos
        )

        # --- Publishers ---
        self.flag_pub = self.create_publisher(
            SuspiciousArea, '/suspicious_areas', alert_qos
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, '/suspicious_markers', alert_qos
        )

        # --- Timer ---
        period = 1.0 / max(self.get_parameter('scan_rate').value, 0.1)
        self.timer = self.create_timer(period, self.scan)

        self.get_logger().info(
            f'Pattern Scanner ready. Mode: {self.detection_mode}. '
            f'Publishing SuspiciousArea messages to /suspicious_areas.'
        )

    def map_callback(self, msg: OccupancyGrid):
        """Store the latest canopy map (height mode)."""
        self.latest_map = msg
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

    def change_callback(self, msg: OccupancyGrid):
        """Store the latest canopy change map (change mode)."""
        self.latest_change = msg
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

    # ── Scan dispatcher ───────────────────────────────────────────────

    def scan(self):
        """Run the active detection mode."""
        if self.detection_mode == 'change':
            self._scan_change()
        else:
            self._scan_height()

    def _scan_change(self):
        """Flag connected regions of canopy LOSS from the change map.

        Change-map semantics (from scan_mapper): -100 = canopy lost,
        +100 = new canopy, 0 = unchanged (including unscanned cells), so a
        threshold well below zero isolates genuine loss.
        """
        if self.latest_change is None:
            return

        width = self.latest_change.info.width
        height = self.latest_change.info.height
        data = np.array(self.latest_change.data, dtype=np.float32).reshape(height, width)

        lost_mask = data < self.lost_threshold
        if not np.any(lost_mask):
            return

        labels, num_features = ndimage.label(lost_mask)
        flags = []
        for label_id in range(1, num_features + 1):
            if len(flags) >= self.max_change_flags:
                break
            region = labels == label_id
            cells = int(np.sum(region))
            if cells < self.min_lost_cells:
                continue

            cy, cx = ndimage.center_of_mass(region)
            wx = self.origin_x + cx * self.resolution
            wy = self.origin_y + cy * self.resolution

            # Loss magnitude 0-100 (cells at the threshold boundary count
            # partial loss, full-loss cells read -100).
            avg_loss = float(np.mean(-data[region]))

            rows, cols = np.where(region)
            bbox_area = (rows.max() - rows.min()) * (cols.max() - cols.min())
            compactness = cells / max(bbox_area, 1)
            area_m2 = cells * self.resolution**2

            # Confidence from loss magnitude + shape compactness; severity
            # from area scaled by loss fraction (full 40 m² loss ≈ 100).
            confidence = min(95.0, avg_loss * 0.85 * compactness + 15.0)
            severity = max(5.0, min(100.0, area_m2 / 40.0 * (avg_loss / 100.0) * 100.0))

            frame = self.latest_change.header.frame_id or self.detection_frame
            flags.append({
                'x': wx, 'y': wy,
                'area_m2': area_m2,
                'type': 'CLEARING',
                'confidence': confidence,
                'severity': severity,
                'frame': frame,
                'desc': f'Canopy loss: {area_m2:.0f}m² cleared, {avg_loss:.0f}% loss',
            })

        self._publish_flags(flags)

    def _scan_height(self):
        """Run the legacy gap/anomaly/line heuristics on the height map."""
        if self.latest_map is None:
            return

        width = self.latest_map.info.width
        height = self.latest_map.info.height
        data = np.array(self.latest_map.data, dtype=np.float32).reshape(height, width)

        data[data < 0] = np.nan

        gaps = self._detect_gaps(data, width, height)
        anomalies = self._detect_anomalies(data, width, height)
        lines = self._detect_lines(data, width, height)

        all_flags = gaps + anomalies + lines
        self._publish_flags(all_flags)

    def _detect_gaps(self, data, width, height):
        """Find connected regions where canopy height is below threshold."""
        low_mask = (data >= 0) & (data < self.gap_threshold)
        if not np.any(low_mask):
            return []

        labels, num_features = ndimage.label(low_mask)
        flags = []
        for label_id in range(1, num_features + 1):
            region = labels == label_id
            area = np.sum(region)
            if area < self.min_gap_area:
                continue

            cy, cx = ndimage.center_of_mass(region)
            wx = self.origin_x + cx * self.resolution
            wy = self.origin_y + cy * self.resolution

            rows, cols = np.where(region)
            bbox_area = (rows.max() - rows.min()) * (cols.max() - cols.min())
            rectangularity = area / max(bbox_area, 1)

            if rectangularity > 0.4:
                confidence = min(rectangularity * 100, 95)
                area_m2 = area * self.resolution**2
                flags.append({
                    'x': wx, 'y': wy,
                    'area_m2': area_m2,
                    'type': 'GAP',
                    'confidence': confidence,
                    'severity': min(confidence * area_m2 / 500, 100),
                    'frame': self.detection_frame,
                    'desc': f'Canopy gap: {area_m2:.0f}m², rect={rectangularity:.2f}',
                })

        return flags

    def _detect_anomalies(self, data, width, height):
        """Find cells where height drops sharply compared to local median."""
        flags = []
        if np.sum(~np.isnan(data)) < 100:
            return flags

        filled = data.copy()
        nan_mask = np.isnan(filled)
        if np.any(nan_mask):
            filled[nan_mask] = 0

        local_median = ndimage.median_filter(filled, size=5)
        local_median[local_median == 0] = np.nan

        diff = local_median - data
        anomaly_mask = (diff > self.anomaly_drop) & ~np.isnan(data)

        if np.any(anomaly_mask):
            labels, num_features = ndimage.label(anomaly_mask)
            for label_id in range(1, min(num_features + 1, 10)):
                region = labels == label_id
                area = np.sum(region)
                if area < 5:
                    continue
                cy, cx = ndimage.center_of_mass(region)
                wx = self.origin_x + cx * self.resolution
                wy = self.origin_y + cy * self.resolution
                avg_drop = np.mean(diff[region])
                confidence = min(avg_drop / 15.0 * 100, 90)
                area_m2 = area * self.resolution**2
                flags.append({
                    'x': wx, 'y': wy,
                    'area_m2': area_m2,
                    'type': 'ANOMALY',
                    'confidence': confidence,
                    'severity': min(avg_drop * area_m2 / 100, 100),
                    'frame': self.detection_frame,
                    'desc': f'Height anomaly: {avg_drop:.1f}m drop, {area_m2:.0f}m²',
                })

        return flags

    def _detect_lines(self, data, width, height):
        """Detect linear features (logging roads) using Sobel edge detection."""
        flags = []
        if np.sum(~np.isnan(data)) < 500:
            return flags

        filled = data.copy()
        filled[np.isnan(filled)] = 0

        gy = ndimage.sobel(filled, axis=0)
        gx = ndimage.sobel(filled, axis=1)
        edges = np.sqrt(gx**2 + gy**2)

        edge_vals = edges[edges > 0]
        if len(edge_vals) == 0:
            return flags
        strong_edges = edges > np.percentile(edge_vals, 90)

        if np.any(strong_edges):
            labels, num_features = ndimage.label(strong_edges)
            for label_id in range(1, min(num_features + 1, 5)):
                region = labels == label_id
                area = np.sum(region)
                if area < self.line_min_cells:
                    continue
                cy, cx = ndimage.center_of_mass(region)
                wx = self.origin_x + cx * self.resolution
                wy = self.origin_y + cy * self.resolution
                area_m2 = area * self.resolution**2
                flags.append({
                    'x': wx, 'y': wy,
                    'area_m2': area_m2,
                    'type': 'LINE',
                    'confidence': 70,
                    'severity': min(70 * area_m2 / 500, 90),
                    'frame': self.detection_frame,
                    'desc': f'Linear clearing (possible road): {area_m2:.0f}m²',
                })

        return flags

    def _publish_flags(self, flags):
        """Publish flags as SuspiciousArea messages and MarkerArray.

        Applies two levels of deduplication:
          1. Temporal: skip flags within dedup_radius and dedup_window of a recent flag
          2. Spatial: skip flags in grid buckets already flagged this session
        """
        marker_array = MarkerArray()
        # Sim clock, not wall time: dedup_window is in sim seconds.
        now_ts = self.get_clock().now().nanoseconds / 1e9

        for flag in flags:
            # --- Temporal dedup: skip if too close in space AND time to a recent flag ---
            too_recent = False
            for rx, ry, rt in self._recent_flags:
                if now_ts - rt > self.dedup_window:
                    continue  # outside time window, ignore this old entry
                if math.hypot(flag['x'] - rx, flag['y'] - ry) < self.dedup_radius:
                    too_recent = True
                    break
            if too_recent:
                continue

            # --- Spatial dedup: skip if in a previously flagged grid bucket ---
            cell_ix = int((flag['x'] - self.origin_x) / self.resolution)
            cell_iy = int((flag['y'] - self.origin_y) / self.resolution)
            if (cell_ix // 10, cell_iy // 10) in self.flagged_cells:
                continue
            self.flagged_cells.add((cell_ix // 10, cell_iy // 10))

            # Record this flag for temporal dedup
            self._recent_flags.append((flag['x'], flag['y'], now_ts))

            self.flag_counter += 1
            now = self.get_clock().now().to_msg()

            # --- SuspiciousArea message (carries full metadata + position) ---
            # Cast to native floats: min(np.float64, int) can return a plain
            # int, and the generated message setters reject non-float types.
            area_msg = SuspiciousArea()
            area_msg.header = Header(stamp=now, frame_id=flag['frame'])
            area_msg.position = Point(x=float(flag['x']), y=float(flag['y']), z=0.0)
            area_msg.type = flag['type']
            area_msg.confidence = float(flag['confidence'])
            area_msg.area_m2 = float(flag['area_m2'])
            area_msg.severity = float(flag['severity'])
            area_msg.description = flag['desc']
            self.flag_pub.publish(area_msg)

            # --- Marker for RViz ---
            marker = Marker()
            marker.header = Header(stamp=now, frame_id=flag['frame'])
            marker.ns = 'suspicious'
            marker.id = self.flag_counter
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose = Pose(
                position=Point(x=flag['x'], y=flag['y'], z=10.0),
                orientation=Quaternion(w=1.0),
            )
            marker.scale.x = max(flag['area_m2'] ** 0.5, 2.0)
            marker.scale.y = max(flag['area_m2'] ** 0.5, 2.0)
            marker.scale.z = 2.0

            colours = {
                'CLEARING': (1.0, 0.0, 0.0, 0.8),
                'GAP': (1.0, 0.3, 0.0, 0.7),
                'ANOMALY': (1.0, 0.0, 0.0, 0.7),
                'LINE': (0.8, 0.0, 0.8, 0.7),
            }
            r, g, b, a = colours.get(flag['type'], (1.0, 1.0, 0.0, 0.7))
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = a
            marker_array.markers.append(marker)

            self.get_logger().info(
                f'FLAG #{self.flag_counter}: {flag["type"]} '
                f'S={flag["severity"]:.0f} C={flag["confidence"]:.0f}% '
                f'area={flag["area_m2"]:.0f}m² at ({flag["x"]:.1f}, {flag["y"]:.1f})'
            )

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def destroy_node(self):
        self.get_logger().info('Pattern Scanner shutting down.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PatternScanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
