#!/bin/bash
# ============================================================================
# (Optional) Rebuild the initial ALFWorld experience graph with your own executor.
# A ready-to-use graph is shipped at data/alfworld/cold_start_graph.json.
#
#   bash run/alfworld/cold_start.sh
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults agentic

"$PYTHON" scripts/cold_start/cold_start_alfworld.py \
    --num_episodes 200 \
    --max_steps 50 \
    --max_workers ${ALFWORLD_NUM_WORKERS:-16} \
    --device ${COLD_START_DEVICE:-cuda} \
    --output_graph ${OUTPUT_GRAPH:-data/alfworld/cold_start_graph.json} \
    --seed 42
