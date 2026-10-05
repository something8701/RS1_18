#!/usr/bin/env bash
# Stop a demo run: SIGINT bag + launch (+ its process group), then TERM/KILL every
# survivor, including ROS 2 system nodes (robot_state_publisher, ekf, slam, nav2, spawners).
# Bracket patterns never match this script itself.
PAT="[r]os2 launch deforestation_monitoring|[r]os2 bag record|[i]gn gazebo|[p]arameter_bridge|[r]viz2|[r]osbridge_websocket|[d]eforestation_monitoring/lib/|[r]obot_state_publisher|[e]kf_node|[s]lam_toolbox|[n]av2_|[c]ontroller_server|[p]lanner_server|[b]t_navigator|[b]ehavior_server|[w]aypoint_follower|[s]moother_server|[v]elocity_smoother|[l]ifecycle_manager|[c]omponent_container|ros_gz_sim/[c]reate|ros_ign_gazebo/[c]reate"
pids() { ps -eo pid,args | grep -E "$PAT" | grep -vE "offline_replay|eval_detector_variants" | awk '{print $1}'; }
for p in $(ps -eo pid,args | grep -E "[r]os2 bag record|[r]os2 launch deforestation_monitoring" | awk '{print $1}'); do
  kill -INT $p 2>/dev/null; kill -INT -- -$p 2>/dev/null
done
for i in $(seq 1 20); do [ -z "$(pids)" ] && { echo "stopped cleanly"; exit 0; }; sleep 1; done
pids | xargs -r kill -TERM; sleep 3; pids | xargs -r kill -KILL; sleep 1
[ -z "$(pids)" ] && echo "stopped (forced)" || { echo "STILL RUNNING:"; pids; }
