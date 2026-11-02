#!/usr/bin/env python3
# husky_cmd.py: JSON | queue: send one operator command to the mission coordinator and print its answer (used by …
import json
import sys
import time

import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

rclpy.init()
node = rclpy.create_node('husky_cmd_cli')
qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
                 history=HistoryPolicy.KEEP_LAST, depth=10)


def mission_queue(wait=5.0):
    from deforestation_interfaces.msg import MissionQueue
    got = []
    sub = node.create_subscription(MissionQueue, '/mission_queue', got.append,
                                   QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
    t = time.time()
    while not got and time.time() - t < wait:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return got[-1] if got else None


if sys.argv[1] == 'queue':
    q = mission_queue()
    if q is None:
        print('no /mission_queue (is a run started?)')
        sys.exit(2)
    print(f'Husky: {q.state}' + (f' · now: {q.current.label}' if q.current.label else ''))
    print('Waiting areas (use the number: demo husky choose N / skip N / proceed N):'
          if q.queue else 'No area waiting.')
    for i, m in enumerate(q.queue, 1):
        print(f'  {i}. {m.label}  at ({m.position.x:.1f}, {m.position.y:.1f})')
    if q.done:
        print(f'Done: {len(q.done)} (' + ', '.join(m.label for m in q.done) + ')')
    sys.exit(0)

cmd = json.loads(sys.argv[1])
cmd.setdefault('id', f'cli{int(time.time() * 1000)}')
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
if str(cmd.get('label', '')).isdigit():
    q = mission_queue()
    n = int(cmd['label'])
    if q is None or not 1 <= n <= len(q.queue):
        print(f'REFUSED there is no waiting area {n} (demo husky queue lists them)')
        sys.exit(1)
    cmd['label'] = q.queue[n - 1].label
answer = []
node.create_subscription(String, '/mission_command_ack',
                         lambda m: answer.append(m.data) if cmd['id'] in m.data else None, qos)
pub = node.create_publisher(String, '/mission_command', qos)
t0 = time.time()
while pub.get_subscription_count() == 0 and time.time() - t0 < 5.0:
    rclpy.spin_once(node, timeout_sec=0.1)
if pub.get_subscription_count() == 0:
    print('NO_SUBSCRIBER')
    sys.exit(2)
for _ in range(5):
    rclpy.spin_once(node, timeout_sec=0.1)
pub.publish(String(data=json.dumps(cmd)))
while not answer and time.time() - t0 < timeout:
    rclpy.spin_once(node, timeout_sec=0.1)
if answer:
    a = json.loads(answer[0])
    print(('OK ' if a.get('ok') else 'REFUSED ') + a.get('msg', ''))
    sys.exit(0 if a.get('ok') else 1)
print('NO_ANSWER')
sys.exit(3)
