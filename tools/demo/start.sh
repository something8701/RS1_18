#!/usr/bin/env bash
# start.sh: [variations...]: start the live demo run and return
source "$(dirname "$0")/demo_env.sh"
WORLD=showcase_forest; TREES=4; HUSKY=true; YOLO=true; EXTRA=(); TAG=""
for v in "$@"; do
  case $v in
    radio)   EXTRA+=(radio:=true) ;;
    noprior) EXTRA+=(drone_prior:=false) ;;
    noyolo)  YOLO=false ;;
    lite)    EXTRA+=(lite:=true) ;;
    norviz)  EXTRA+=(rviz:=false) ;;
    dense)   WORLD=dense_forest; TREES=10; HUSKY=false; YOLO=false ;;
    *) echo "unknown variation: $v (see the top of this script)"; exit 1 ;;
  esac
  TAG="${TAG}_$v"
done
if pgrep -f "[r]os2 launch deforestation_monitoring|[i]gn gazebo" > /dev/null; then
  echo "A simulation is already running: tools/demo/stop.sh first (or preflight.sh --clean)."; exit 1
fi
[ "$YOLO" = true ] && [ ! -f "$YOLO_WEIGHTS" ] && { echo "no YOLO weights: $YOLO_WEIGHTS"; exit 1; }
NAME=demo_$(date +%H%M)$TAG
echo "$NAME" > "$DEMO_DIR/current"
L=$DEMO_DIR/$NAME.log
ros_env
ARGS=(world:=$WORLD n_trees:=$TREES balanced:=true settle_loops:=2)
if [ "$HUSKY" = true ]; then
  ARGS+=(use_husky:=true waste:=true evidence:=$YOLO yolo:=$YOLO yolo_weights:=$YOLO_WEIGHTS)
fi
cd "$WS"
say "Starting $NAME: ${ARGS[*]} ${EXTRA[*]}"
(setsid nohup ros2 launch deforestation_monitoring removal_test.launch.py "${ARGS[@]}" "${EXTRA[@]}" \
  > "$L" 2>&1 < /dev/null &)
echo "  log: $L"
printf '  waiting for the dashboard'
for _ in $(seq 1 90); do
  curl -s -o /dev/null "$DASH/api/data" && break; printf '.'; sleep 2
done
echo
if curl -s -o /dev/null "$DASH/api/data"; then
  ok "dashboard up: $DASH/?layout=focus   (from the Mac: http://$PC_IP:8081/?layout=focus)"
  [ -z "$NO_BROWSER" ] && (xdg-open "$DASH/?layout=focus" > /dev/null 2>&1 &)
else
  bad "the dashboard did not come up in 3 min: tail $L"
fi
if [ "$HUSKY" = true ]; then
  printf '  waiting for the Husky navigation (Nav2)'
  for _ in $(seq 1 40); do
    grep -q "husky1.lifecycle_manager_navigation.*Managed nodes are active" "$L" && break
    printf '.'; sleep 2
  done
  echo
  if grep -q "husky1.lifecycle_manager_navigation.*Managed nodes are active" "$L"; then
    ok "Husky navigation active"
  elif [ -z "$DEMO_RETRY" ]; then
    bad "Husky navigation did not start (a rare start-up race). Restarting the run once..."
    bash "$(dirname "$0")/stop.sh" > /dev/null 2>&1
    DEMO_RETRY=1 exec bash "$0" "$@"
  else
    bad "Husky navigation did not start twice. Stop and start by hand: tools/demo/stop.sh && tools/demo/start.sh"
  fi
fi
if [ -n "$RECORD" ]; then
  mkdir -p "$DEMO_DIR/$NAME"
  (setsid nohup python3 "$SRC/tools/eval/record_dashboard.py" --out "$DEMO_DIR/$NAME/dashboard.mp4" \
    --url "$DASH/?layout=focus" --fps 1 --speed 4 --duration 3600 --stop-file "$DEMO_DIR/$NAME/stop" \
    > "$DEMO_DIR/$NAME/recorder.log" 2>&1 < /dev/null &)
  ok "recording the dashboard: $DEMO_DIR/$NAME/dashboard.mp4 (stop.sh ends it)"
fi
say "Next: tools/demo/watch.sh   (live phases, with what to say next)"
