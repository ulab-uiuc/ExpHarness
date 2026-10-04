#!/bin/bash
# ============================================================================
# (Optional) Rebuild the initial AppWorld experience graph with your own executor.
# A ready-to-use graph is shipped at data/appworld/cold_start_graph.json.
#
#   bash run/appworld/start_servers.sh
#   APPWORLD_ROOT=/path/to/appworld bash run/appworld/cold_start.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults agentic
: "${APPWORLD_ROOT:?Please export APPWORLD_ROOT (directory that contains AppWorld data/)}"

# Needs the `appworld` package (to list the train task ids) -> run with the AppWorld env
"${APPWORLD_PYTHON:-$PYTHON}" scripts/cold_start/cold_start_appworld.py \
    --server_url ${APPWORLD_SERVER_URL:-http://127.0.0.1:8006,8007,8008,8009} \
    --max_steps 40 \
    --retriever_device cpu \
    --output_graph ${OUTPUT_GRAPH:-data/appworld/cold_start_graph.json} \
    --seed 42
