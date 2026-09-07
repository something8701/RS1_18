#!/usr/bin/env python3
"""
Mission Evaluation Node — scores the detection pipeline against ground truth.

The proposal promises measurable evaluation: detection accuracy, false
positives, decision outcomes and end-to-end mission time. This node:

  - matches /suspicious_areas detections against /ground_truth_events
    (published by simulate_deforestation) within match_radius metres
    → true positives, false positives, missed events
  - listens to /inspection_reports for decision outcomes (inspected /
    cancelled / unreachable) and end-to-end mission time (flag → report
    latency, sim clock)
  - publishes a /evaluation_summary String every summary_period and
    appends CSV rows to eval_dir/evaluation.csv

Detection frames (parrot1_odom) and truth frames (husky1_map) are both
anchored at the sim world origin, so coordinates compare directly.
"""

import csv
import math
import os
import re
from collections import deque

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from deforestation_interfaces.msg import SuspiciousArea


class EvaluateMission(Node):
    """Matches detections to ground truth and reports mission metrics."""

    def __init__(self):
        super().__init__('evaluate_mission')

        self.declare_parameter('match_radius', 8.0,
            descriptor=ParameterDescriptor(
                description='Max distance (m) between a detection and a ground-truth '
                            'event for the detection to count as a true positive'))
        self.declare_parameter('summary_period', 10.0,
            descriptor=ParameterDescriptor(
                description='Seconds between /evaluation_summary publishes and CSV rows'))
        self.declare_parameter('eval_dir', '/tmp/deforestation_eval',
            descriptor=ParameterDescriptor(
                description='Directory for the evaluation CSV'))

        self.match_radius = self.get_parameter('match_radius').value
        self.summary_period = self.get_parameter('summary_period').value
        self.eval_dir = self.get_parameter('eval_dir').value
        os.makedirs(self.eval_dir, exist_ok=True)
        self.csv_path = os.path.join(self.eval_dir, 'evaluation.csv')
        self._csv_header_written = os.path.exists(self.csv_path)

        # Ground-truth events waiting for a matching detection
        self.events = []          # [{'x', 'y', 'stamp_ns', 'matched'}]
        self.flags = []           # [{'x', 'y', 'is_tp'}]
        self.outcomes = {'inspected': 0, 'cancelled': 0, 'unreachable': 0, 'other': 0}
        self.latencies = []       # flag → report seconds (sim clock)
        self._pending_flags = deque(maxlen=100)  # (x, y, stamp_ns) awaiting reports

        # -- QoS --
        alert_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.truth_sub = self.create_subscription(
            SuspiciousArea, '/ground_truth_events', self.truth_callback, alert_qos
        )
        self.flag_sub = self.create_subscription(
            SuspiciousArea, '/suspicious_areas', self.flag_callback, alert_qos
        )
        self.report_sub = self.create_subscription(
            String, '/inspection_reports', self.report_callback, alert_qos
        )
        self.summary_pub = self.create_publisher(
            String, '/evaluation_summary', alert_qos
        )

        self.create_timer(self.summary_period, self.publish_summary)
        self.get_logger().info(
            f'Evaluation node ready. match_radius={self.match_radius:.0f}m, '
            f'CSV: {self.csv_path}'
        )

    # ── Ground truth / detection matching ─────────────────────────────

    def truth_callback(self, msg: SuspiciousArea):
        """Record a known clearing event from the simulator."""
        self.events.append({
            'x': msg.position.x,
            'y': msg.position.y,
            'stamp_ns': msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec,
            'matched': False,
        })
        self.get_logger().debug(
            f'Ground truth #{len(self.events)} at ({msg.position.x:.1f},{msg.position.y:.1f})')

    def flag_callback(self, msg: SuspiciousArea):
        """Score a detection: TP if it matches an unmatched event, else FP."""
        fx, fy = msg.position.x, msg.position.y

        best = None
        for ev in self.events:
            if ev['matched']:
                continue
            d = math.hypot(fx - ev['x'], fy - ev['y'])
            if d <= self.match_radius and (best is None or d < best[0]):
                best = (d, ev)

        is_tp = best is not None
        if is_tp:
            best[1]['matched'] = True
            self.get_logger().info(
                f'Detection @ ({fx:.1f},{fy:.1f}) matched truth '
                f'({best[0]:.1f}m) — TRUE POSITIVE')
        else:
            self.get_logger().warn(
                f'Detection @ ({fx:.1f},{fy:.1f}) has no truth event within '
                f'{self.match_radius:.0f}m — FALSE POSITIVE')

        self.flags.append({'x': fx, 'y': fy, 'is_tp': is_tp})
        self._pending_flags.append(
            (fx, fy, msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec))

    def _pop_pending_for_report(self, text):
        """Pop the pending flag nearest to the coordinates in a report.

        Reports end with "@ (x, y)". Matching by location is more robust than
        blind FIFO when /suspicious_areas carries both real and duplicate
        detections. Falls back to FIFO if the coordinates cannot be parsed.
        """
        if not self._pending_flags:
            return None
        m = re.search(r'@\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)', text)
        if m is None:
            _, _, flag_ns = self._pending_flags.popleft()
            return flag_ns

        rx, ry = float(m.group(1)), float(m.group(2))
        best_idx, best_dist = None, None
        for i, (fx, fy, _) in enumerate(self._pending_flags):
            d = math.hypot(rx - fx, ry - fy)
            if best_dist is None or d < best_dist:
                best_dist, best_idx = d, i
        if best_idx is None:
            return None

        _, _, flag_ns = self._pending_flags[best_idx]
        del self._pending_flags[best_idx]
        return flag_ns
    # ── Decision outcomes ─────────────────────────────────────────────

    def report_callback(self, msg: String):
        """Parse an inspection report for outcome + end-to-end mission time."""
        data = msg.data

        if 'INSPECTED' in data:
            outcome = 'inspected'
        elif 'CANCELLED' in data:
            outcome = 'cancelled'
        elif 'UNREACHABLE' in data:
            outcome = 'unreachable'
        else:
            outcome = 'other'
        self.outcomes[outcome] += 1

        # End-to-end mission time: flag stamp → report arrival (sim clock),
        # matched back to the flag by its coordinates.
        flag_ns = self._pop_pending_for_report(data)
        if flag_ns is not None:
            now = self.get_clock().now()
            now_ns = now.nanoseconds
            latency = (now_ns - flag_ns) / 1e9
            if latency >= 0.0:
                self.latencies.append(latency)
                self.get_logger().info(
                    f'Mission time {outcome}: {latency:.1f}s (flag→report, sim clock)')
            else:
                self.get_logger().debug(f'Negative latency {latency:.1f}s ignored '
                                        '(clock/order artefact)')

    # ── Summary ───────────────────────────────────────────────────────

    def _metrics(self):
        n_events = len(self.events)
        n_flags = len(self.flags)
        tp = sum(1 for f in self.flags if f['is_tp'])
        missed = sum(1 for ev in self.events if not ev['matched'])
        accuracy = tp / float(n_events) if n_events else 0.0
        precision = tp / float(n_flags) if n_flags else 0.0
        avg_latency = sum(self.latencies) / float(len(self.latencies)) \
            if self.latencies else 0.0
        return n_events, n_flags, tp, missed, accuracy, precision, avg_latency

    def publish_summary(self):
        """Publish /evaluation_summary and append a CSV row."""
        (n_events, n_flags, tp, missed,
         accuracy, precision, avg_latency) = self._metrics()

        summary = (
            f'EVALUATION — events={n_events} detections={n_flags} '
            f'TP={tp} FP={n_flags - tp} missed={missed} | '
            f'accuracy={accuracy*100:.0f}% precision={precision*100:.0f}% | '
            f'avg mission time={avg_latency:.1f}s | '
            f'inspected={self.outcomes["inspected"]} '
            f'cancelled={self.outcomes["cancelled"]} '
            f'unreachable={self.outcomes["unreachable"]}'
        )
        self.summary_pub.publish(String(data=summary))
        self.get_logger().info(summary)

        row = [
            self.get_clock().now().nanoseconds / 1e9,
            n_events, n_flags, tp, n_flags - tp, missed,
            f'{accuracy:.3f}', f'{precision:.3f}', f'{avg_latency:.1f}',
            self.outcomes['inspected'], self.outcomes['cancelled'],
            self.outcomes['unreachable'],
        ]
        with open(self.csv_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if not self._csv_header_written:
                writer.writerow([
                    'sim_time_s', 'events', 'detections', 'true_positives',
                    'false_positives', 'missed', 'accuracy', 'precision',
                    'avg_mission_time_s', 'inspected', 'cancelled', 'unreachable',
                ])
                self._csv_header_written = True
            writer.writerow(row)

    def destroy_node(self):
        (n_events, n_flags, tp, missed,
         accuracy, precision, avg_latency) = self._metrics()
        self.get_logger().info(
            f'Evaluation complete — events={n_events} detections={n_flags} '
            f'TP={tp} FP={n_flags - tp} missed={missed} '
            f'accuracy={accuracy*100:.0f}% avg mission time={avg_latency:.1f}s '
            f'| CSV: {self.csv_path}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EvaluateMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
