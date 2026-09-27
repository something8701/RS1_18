# Setup and running

Tested on Ubuntu 22.04, ROS 2 Humble, Gazebo Fortress (Ignition 6).

## 1. Dependencies

```bash
sudo apt update
sudo apt install -y \
  ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
  ros-humble-robot-localization ros-humble-ros-gz ros-humble-cv-bridge \
  ros-humble-rosbridge-suite \
  python3-numpy python3-scipy python3-sklearn python3-opencv python3-websocket \
  mesa-utils
```

## 2. Get the code and build

The three packages (`41068_ignition_bringup`, `deforestation_interfaces`,
`deforestation_monitoring`) must sit directly inside the workspace `src/` folder.

```bash
mkdir -p ~/41068_ws/src && cd ~/41068_ws/src
git clone <REPO_URL> .          # or clone elsewhere and copy the three packages in
cd ~/41068_ws
colcon build --symlink-install
source install/setup.bash
colcon list                     # should list exactly the three packages above
```

Add `source ~/41068_ws/install/setup.bash` to `~/.bashrc` if you want it in every terminal.

## 3. Run the full demo

```bash
ros2 launch deforestation_monitoring demo_full.launch.py world:=simple_trees
```

`world:=dense_forest` also works (slower). Watch progress in a second terminal:

```bash
ros2 topic echo /change_detector/status     # BASELINE -> MONITORING
ros2 topic echo /canopy_change_events       # canopy-loss detections
ros2 topic echo /evaluation_summary
```

Expected sequence: the drone flies one lawnmower pass, `change_detector` logs
`BASELINE FROZEN ... N canopy cells` (N well above 0), `simulate_tree_removal`
removes trees, `/canopy_change_events` fires, and the Husky navigates to inspect.

## 4. Quick health checks (run while the demo is up)

```bash
# Lidar: should print a few dozen (beams that hit something). 0 = lidar broken.
ros2 topic echo /husky1/scan --once --field ranges | tr ',' '\n' | grep -vc inf

# Simulation speed: values should stay close to 1.0
ign topic -e -t /world/simple_trees/stats -n 5 | grep real_time_factor
```

## 5. Unit tests (no ROS needed)

```bash
cd ~/41068_ws && python3 -m pytest src/deforestation_monitoring/test
```

## Troubleshooting

**Leftover processes after Ctrl+C** (two simulations fighting over the same topics):

```bash
pkill -f "ign gazebo"; pkill -f "gz sim"; pkill -f ruby; pkill -f parameter_bridge; pkill -f rviz2
```

**Windows / WSL users.** WSL's GPU driver (Mesa D3D12, OpenGL 4.2) cannot render
Gazebo's lidars: every beam reads the minimum range (0.2 m), so SLAM, Nav2 and the
canopy map get no real data. Check with the lidar health check above. Workarounds,
in order of preference:

1. Use a machine with native Ubuntu and a working GPU driver (recommended for demos).
2. Force CPU rendering for Gazebo only. This gives correct lidars but runs slowly with
   freezes of up to ~50 s, because each lidar frame renders on a single CPU thread.
   Do **not** commit these edits; they are machine-specific.
   - In `41068_ignition.launch.py`, add `SetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE', '1')`
     immediately before Gazebo is started and `UnsetEnvironmentVariable('LIBGL_ALWAYS_SOFTWARE')`
     immediately after, and change the Gazebo args from `' -r'` to `' -r -s'` (headless).
   - Optionally change both cameras from `type="rgbd_camera"` to `type="camera"`
     (topic `/model/<name>/camera/image`) and the Husky camera to 320x240.
   - Make sure `~/.bashrc` does not globally export `LIBGL_ALWAYS_SOFTWARE=1`.
