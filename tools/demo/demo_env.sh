# demo_env.sh: Shared settings of the demo scripts (sourced, not run)
WS=~/rs1_18_ws
SRC=$WS/src/RS1_18
DEMO_DIR=$WS/eval/demo
mkdir -p "$DEMO_DIR"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-7} ROS_LOCALHOST_ONLY=1 DISPLAY=${DISPLAY:-:1}
[ -z "$XAUTHORITY" ] && [ -f /run/user/$(id -u)/gdm/Xauthority ] && export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority
PC_IP=$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
[ -z "$PC_IP" ] && PC_IP=$(hostname -I | awk '{print $1}')
[ -n "$SSH_CONNECTION" ] && NO_BROWSER=${NO_BROWSER:-1}
export ROS_LOG_DIR=$DEMO_DIR/ros
YOLO_WEIGHTS=${YOLO_WEIGHTS:-$WS/eval/yolo/weights/evidence_v1.pt}
DASH=http://localhost:8081
current_run() { cat "$DEMO_DIR/current" 2>/dev/null; }
current_log() { echo "$DEMO_DIR/$(current_run).log"; }
ros_env() { source /opt/ros/humble/setup.bash; source "$WS/install/setup.bash"; }
ok()   { printf '  \e[32m✔\e[0m %s\n' "$*"; }
bad()  { printf '  \e[31m✘\e[0m %s\n' "$*"; }
warn() { printf '  \e[33m!\e[0m %s\n' "$*"; }
say()  { printf '\e[1;36m%s\e[0m\n' "$*"; }
