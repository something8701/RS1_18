"""Keep the unit tests off the live simulation's ROS network.

The tests create real tracker / mapper nodes that publish on the live topic
names (/parrot_tree_baseline, /parrot_tree_change_events, ...). Run from a
shell that still has a live run's ROS_DOMAIN_ID set, they once fed a fake
3-tree baseline into a live removal test (run 35, 2026-10-03). rclpy.init()
reads the environment, so pin every test to a private domain here, before any
test module starts rclpy.
"""
import os

os.environ['ROS_DOMAIN_ID'] = '142'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
