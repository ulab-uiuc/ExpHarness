#!/bin/bash
# ============================================================================
# Evaluate a trained ExpHarness copilot on the ExpSuite-Static test sets.
#
#   bash run/static/eval.sh <domain> <experiment_dir> <step>
#   e.g. bash run/static/eval.sh all checkpoints/expharness_static_all_42 500
#
# Per-benchmark accuracy (with / without experiences) is written to
# outputs/eval/<name>/eval_summary.json
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults static

DOMAIN=${1:?usage: eval.sh <all|reasoning|coding> <experiment_dir> <step>}
CKPT_DIR=${2:?usage: eval.sh <all|reasoning|coding> <experiment_dir> <step>}
STEP=${3:?usage: eval.sh <all|reasoning|coding> <experiment_dir> <step>}
ACTOR_CKPT=$CKPT_DIR/actor/global_step_$STEP
GRAPH=${GRAPH:-$CKPT_DIR/graph_step_$STEP.json}
NAME=$(basename "$CKPT_DIR")_step$STEP
DATA=data/static/$DOMAIN

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export QA_NUM_WORKERS=${QA_NUM_WORKERS:-16}
GRAPH_PORT=${GRAPH_PORT:-8013}

start_graph_server "$GRAPH_PORT" "$GRAPH" 2000
eval_common_args "$ACTOR_CKPT"
mkdir -p logs

PYTHONUNBUFFERED=1 "$PYTHON" -m verl.trainer.main_expharness \
    "${EVAL_COMMON[@]}" \
    data.train_files=$DATA/test.parquet \
    data.val_files=$DATA/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.project_name=ExpHarness-Static-Eval \
    trainer.experiment_name=$NAME \
    +output_context_dir=outputs/eval/$NAME \
    retriever.url=http://127.0.0.1:${GRAPH_PORT}/retrieve \
    retriever.topk=10 \
    +env.name=qa \
    2>&1 | tee logs/eval_$NAME.log

grep -h "^\[Eval\]" logs/eval_$NAME.log
