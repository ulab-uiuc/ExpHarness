#!/bin/bash
# ============================================================================
# (Optional) Rebuild the initial experience graph for a static domain.
# Ready-to-use graphs are shipped at data/static/<domain>/cold_start_graph.json.
#
#   bash run/static/cold_start.sh <all|reasoning|coding>
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults static

DOMAIN=${1:-all}
"$PYTHON" scripts/cold_start/cold_start_static.py \
    --train_data data/static/$DOMAIN/train.parquet \
    --num_samples 300 \
    --max_workers ${QA_NUM_WORKERS:-16} \
    --device ${COLD_START_DEVICE:-cuda} \
    --output_graph ${OUTPUT_GRAPH:-data/static/$DOMAIN/cold_start_graph.json} \
    --seed 42
