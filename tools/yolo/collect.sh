#!/usr/bin/env bash
# collect.sh: OUT_DIR [collect_dataset.py args]: start a Gazebo server with a world (WORLD, default …
set -e
OUT=$1; shift
WORLD=${WORLD:-showcase_forest}
cd ~/rs1_18_ws
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-23} ROS_LOCALHOST_ONLY=1 DISPLAY=${DISPLAY:-:1}
export IGN_PARTITION=${IGN_PARTITION:-yolo_$WORLD}
source /opt/ros/humble/setup.bash && source install/setup.bash
SHARE=$(ros2 pkg prefix 41068_ignition_bringup)/share/41068_ignition_bringup
export IGN_GAZEBO_SERVER_CONFIG_PATH=$SHARE/config/ignition_server.config
mkdir -p "$OUT"
LOG=$OUT/gazebo.log
if ign service -l 2>/dev/null | grep "/create$" | grep -vq "/world/$WORLD/create"; then
  echo "another Gazebo world is running; stop it first (tools/yolo/stop_gz.sh)"; exit 1
fi
if ! ign service -l 2>/dev/null | grep -q "/world/$WORLD/create"; then
  setsid nohup ros2 launch ros_ign_gazebo ign_gazebo.launch.py \
    ign_args:="$SHARE/worlds/$WORLD.sdf -r -s" > "$LOG" 2>&1 < /dev/null &
  until ign service -l 2>/dev/null | grep -q "/world/$WORLD/create"; do sleep 2; done
  sleep 5
  STARTED=1
fi
setsid nohup ros2 run ros_ign_bridge parameter_bridge \
  /datacam/image@sensor_msgs/msg/Image[ignition.msgs.Image \
  /datacam/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock --ros-args -r __node:=datacam_bridge_$WORLD \
  > "$OUT/bridge.log" 2>&1 < /dev/null &
python3 "$(dirname "$0")/collect_dataset.py" --world "$WORLD" \
  --world_sdf "$SHARE/worlds/$WORLD.sdf" --out "$OUT" "$@"
pkill -f "[d]atacam_bridge_$WORLD" || true
ign service -s /world/$WORLD/remove --reqtype ignition.msgs.Entity --reptype ignition.msgs.Boolean \
  --timeout 2000 --req 'name: "datacam" type: MODEL' > /dev/null || true
if [ -n "$STARTED" ] && [ -z "$KEEP_GZ" ]; then
  P="[i]gn gazebo .*worlds/$WORLD\.sdf -r -s|ign_args:=.*worlds/$WORLD\.sdf"
  pkill -INT -f "$P"; for _ in 1 2 3 4 5 6 7 8; do pgrep -f "$P" > /dev/null || break; sleep 1; done
  pkill -9 -f "$P" || true
fi
