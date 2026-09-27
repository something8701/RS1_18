import os
from glob import glob
from setuptools import setup

package_name = 'deforestation_monitoring'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ali Coruk',
    maintainer_email='ali.c.coruk@student.uts.edu.au',
    description='Multi-robot deforestation monitoring system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tree_mapper = deforestation_monitoring.tree_mapper:main',
            'scan_mapper = deforestation_monitoring.scan_mapper:main',
            'pattern_scanner = deforestation_monitoring.pattern_scanner:main',
            'mission_coordinator = deforestation_monitoring.mission_coordinator:main',
            'demo_patrol = deforestation_monitoring.demo_patrol:main',
            'simulate_deforestation = deforestation_monitoring.simulate_deforestation:main',
            'simulate_tree_removal = deforestation_monitoring.simulate_tree_removal:main',
            'husky_patrol = deforestation_monitoring.husky_patrol:main',
            'data_server = deforestation_monitoring.data_server:main',
            'drone_mapper = deforestation_monitoring.drone_mapper:main',
            'parrot_tree_tracker = deforestation_monitoring.parrot_tree_tracker:main',
            'calibrate_tree_detection = deforestation_monitoring.calibrate_tree_detection:main',
            'replay_evaluate = deforestation_monitoring.replay_evaluate:main',
            'removal_test = deforestation_monitoring.removal_test:main',
            'tune_on_recording = deforestation_monitoring.tune_on_recording:main',
            'screen_recorder = deforestation_monitoring.screen_recorder:main',
            'demo_replay = deforestation_monitoring.demo_replay:main',
            'tree_fusion = deforestation_monitoring.tree_fusion:main',
            'evaluate_mission = deforestation_monitoring.evaluate_mission:main',
            'camera_republisher = deforestation_monitoring.camera_republisher:main',
            'clock_watchdog = deforestation_monitoring.clock_watchdog:main',
        ],
    },
)
