#!/usr/bin/env python3
"""
Deforestation Simulator: Waits for baseline tree map to be established, then
injects suspicious area flags at known clearing sites to test the pipeline.

Phase 1: Baseline mapping (no flags). Tree trunks are being mapped by Husky.
Phase 2: Monitoring (flags fire at pre-defined clearing sites).

Publishes SuspiciousArea messages with full metadata (type, confidence,
area, severity) instead of bare PoseStamped.

The staged sites are published ONLY on /ground_truth_events. That topic is
the ground truth the evaluation node scores against; /suspicious_areas must
come from the pattern scanner, otherwise the injected test flags would also
be counted as "detections" and the evaluation would score the simulator
against itself instead of scoring perception.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
from deforestation_interfaces.msg import SuspiciousArea


class SimulateDeforestation(Node):
    """Injects test flags at known clearing sites after baseline is established."""

    def __init__(self):
        super().__init__('simulate_deforestation')


        self.sites = [
            {'x': 15.0, 'y': 10.0, 'type': 'GAP', 'confidence': 92.0,
             'area_m2': 200.0, 'severity': 85.0,
             'desc': 'Site A: Large clearing (~200m², 8m radius)'},
            {'x': -20.0, 'y': -15.0, 'type': 'GAP', 'confidence': 85.0,
             'area_m2': 110.0, 'severity': 70.0,
             'desc': 'Site B: Medium clearing (~110m², 6m radius)'},
            {'x': 5.0, 'y': -25.0, 'type': 'ANOMALY', 'confidence': 78.0,
             'area_m2': 50.0, 'severity': 55.0,
             'desc': 'Site C: Small clearing (~50m², 4m radius)'},
        ]
        self.site_index = 0
        self.baseline_ready = False
        self.trees_mapped = 0

        # -- QoS profiles --
        alert_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.truth_pub = self.create_publisher(
            SuspiciousArea, '/ground_truth_events', alert_qos
        )

        self.create_subscription(
            String, '/baseline_status', self.baseline_callback, alert_qos
        )

        self.get_logger().info(
            f'Deforestation simulator ready. {len(self.sites)} sites staged '
            f'with full SuspiciousArea metadata. Waiting for baseline...'
        )

    def baseline_callback(self, msg: String):
        """Handle baseline establishment notification."""
        if self.baseline_ready:
            return
        self.baseline_ready = True
        try:
            self.trees_mapped = int(msg.data.split(':')[1])
        except (IndexError, ValueError):
            self.trees_mapped = 0
            self.get_logger().warn(f'Could not parse tree count from: {msg.data}')

        self.get_logger().info(
            f'Baseline received: {self.trees_mapped} trees mapped. '
            'Starting monitoring phase in 30s...'
        )
        # Humble note: rclpy create_timer has no one_shot kwarg (Iron+ only),
        # so a periodic 30 s timer drives the flag cadence.
        self.create_timer(30.0, self.publish_next_flag)

    def publish_next_flag(self):
        """Publish the next staged clearing as a SuspiciousArea message."""
        if not self.baseline_ready:
            return
        if self.site_index == 0:
            self.get_logger().info(
                'MONITORING ACTIVE. Flags will fire at clearing sites every 30s.')
        site = self.sites[self.site_index % len(self.sites)]
        self.site_index += 1

        flag = SuspiciousArea()
        flag.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='husky1_map',
        )
        flag.position = Point(x=site['x'], y=site['y'], z=0.0)
        flag.type = site['type']
        flag.confidence = site['confidence']
        flag.area_m2 = site['area_m2']
        flag.severity = site['severity']
        flag.description = site['desc']

        self.truth_pub.publish(flag)
        self.get_logger().info(
            f'GROUND TRUTH: {flag.type} S={flag.severity:.0f} '
            f'C={flag.confidence:.0f}% area={flag.area_m2:.0f}m² '
            f'at ({site["x"]:.1f}, {site["y"]:.1f}) — {site["desc"]}'
        )

    def destroy_node(self):
        self.get_logger().info(
            f'Deforestation simulator shutting down. {self.site_index} flags fired.'
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimulateDeforestation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
