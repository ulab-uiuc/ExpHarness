#!/bin/bash
# ============================================================================
# Start the AppWorld episode servers (needed by train / eval / cold start).
# AppWorld has its own dependencies, so the servers run in a separate Python
# environment ($APPWORLD_PYTHON) and talk to the trainer over HTTP.
#
#   APPWORLD_ROOT=/path/to/appworld APPWORLD_PYTHON=/path/to/appworld-env/bin/python \
#       bash run/appworld/start_servers.sh            # ports 8006-8009
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults agentic
trap - EXIT   # keep the servers running after this script returns

: "${APPWORLD_ROOT:?Please export APPWORLD_ROOT (directory that contains AppWorld data/)}"
APPWORLD_PYTHON=${APPWORLD_PYTHON:-$PYTHON}
APPWORLD_PORT=${APPWORLD_PORT:-8006}
APPWORLD_NUM_SERVERS=${APPWORLD_NUM_SERVERS:-4}

export APPWORLD_EPISODE_PROCESS=1        # run each episode in a child process
export APPWORLD_EPISODE_TIMEOUT=${APPWORLD_EPISODE_TIMEOUT:-120}
export APPWORLD_KEEP_OUTPUTS=0
export GEMINI_MAX_WALL_TIME=${GEMINI_MAX_WALL_TIME:-10}
export GEMINI_REQUEST_TIMEOUT=${GEMINI_REQUEST_TIMEOUT:-8}

mkdir -p logs
CUDA_VISIBLE_DEVICES="" nohup "$APPWORLD_PYTHON" scripts/servers/appworld_server.py \
    --port "$APPWORLD_PORT" --num-servers "$APPWORLD_NUM_SERVERS" \
    > logs/appworld_servers.log 2>&1 &
echo "AppWorld servers launching (pid $!), log: logs/appworld_servers.log"

LAST=$((APPWORLD_PORT + APPWORLD_NUM_SERVERS - 1))
for port in $(seq "$APPWORLD_PORT" "$LAST"); do
    if wait_for_url "http://127.0.0.1:${port}/health" 120; then echo "  port ${port}: OK"
    else echo "  port ${port}: FAILED (see logs/appworld_servers.log)"; exit 1; fi
done
