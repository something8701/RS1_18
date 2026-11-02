# conftest.py: Keep the unit tests off the live simulation's ROS network
import os

os.environ['ROS_DOMAIN_ID'] = '142'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
