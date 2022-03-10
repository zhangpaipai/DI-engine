#!/usr/bin/env bash

BASEDIR=$(dirname "$0")/../entry

kill_descendant_processes() {
  local pid="$1"
  local and_self="${2:-false}"
  if children="$(pgrep -P "$pid")"; then
    for child in $children; do
      kill_descendant_processes "$child" true
    done
  fi
  if [[ "$and_self" == true ]]; then
    kill -9 "$pid"
  fi
}

trap "kill_descendant_processes $$" EXIT

ditask --package $BASEDIR \
  --main main_league.main \
  --parallel-workers 5 \
  --protocol tcp \
  --address 127.0.0.1 \
  --ports 50515 \
  --node-ids 0 \
  --topology mesh &

sleep 10000
