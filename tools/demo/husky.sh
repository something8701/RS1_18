#!/usr/bin/env bash
# husky.sh: COMMAND [AREA]: operator control of the Husky from the terminal (the same commands as the …
source "$(dirname "$0")/demo_env.sh"
ros_env
case $1 in
  queue) python3 "$(dirname "$0")/husky_cmd.py" queue; exit ;;
  ask)     [ "$2" = on ] && J='{"cmd":"ask","on":true}' || J='{"cmd":"ask","on":false}' ;;
  pause|resume) J="{\"cmd\":\"$1\"}" ;;
  proceed) J="{\"cmd\":\"proceed\",\"label\":\"$2\"}" ;;
  choose|skip|unskip)
    [ -z "$2" ] && { echo "usage: husky.sh $1 \"AREA LABEL\""; exit 1; }
    J="{\"cmd\":\"$1\",\"label\":\"$2\"}" ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
R=$(python3 "$(dirname "$0")/husky_cmd.py" "$J" 12)
case $R in
  OK*)        ok "Husky: ${R#OK }" ;;
  REFUSED*)   warn "Husky refused: ${R#REFUSED }" ;;
  NO_SUBSCRIBER) bad "the mission coordinator is not running (is a run started?)" ;;
  *)          warn "no answer in 12 s (out of radio contact? the command waits for it)" ;;
esac
