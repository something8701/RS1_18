#!/usr/bin/env bash
# stop_gz.sh: Stop the headless Gazebo servers and camera bridges of the YOLO tools
P='ign [g]azebo .*worlds/.*\.sdf -r -s|parameter_bridge /[d]atacam|ign_gazebo\.launch\.py'
pkill -INT -f "$P"
for _ in 1 2 3 4 5 6 7 8; do pgrep -f "$P" > /dev/null || exit 0; sleep 1; done
pkill -9 -f "$P"; sleep 1
