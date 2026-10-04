#!/bin/bash
# ============================================================================
# Evaluate a trained ExpHarness copilot on AppWorld Test-Normal / Test-Challenge.
#
#   bash run/appworld/start_servers.sh
#   bash run/appworld/eval.sh <experiment_dir> <step>
#
# Results are reported per split (appworld_test_n / appworld_test_c) in
# outputs/eval/<name>/eval_summary.json  (pass rate + avg. steps)
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults agentic

CKPT_DIR=${1:?usage: eval.sh <experiment_dir> <step>}
STEP=${2:?usage: eval.sh <experiment_dir> <step>}
ACTOR_CKPT=$CKPT_DIR/actor/global_step_$STEP
GRAPH=${GRAPH:-$CKPT_DIR/graph_step_$STEP.json}
NAME=$(basename "$CKPT_DIR")_step$STEP

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export APPWORLD_SERVER_URL=${APPWORLD_SERVER_URL:-http://127.0.0.1:8006,8007,8008,8009}
export APPWORLD_MAX_STEPS=40
export APPWORLD_POOL_WORKERS=${APPWORLD_POOL_WORKERS:-4}
export APPWORLD_REQUEST_TIMEOUT=${APPWORLD_REQUEST_TIMEOUT:-150}
GRAPH_PORT=${GRAPH_PORT:-8012}

start_graph_server "$GRAPH_PORT" "$GRAPH" 2000
eval_common_args "$ACTOR_CKPT"
mkdir -p logs

PYTHONUNBUFFERED=1 "$PYTHON" -m verl.trainer.main_expharness \
    "${EVAL_COMMON[@]}" \
    data.train_files=data/appworld/val.parquet \
    data.val_files=data/appworld/val.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.project_name=ExpHarness-AppWorld-Eval \
    trainer.experiment_name=$NAME \
    +output_context_dir=outputs/eval/$NAME \
    retriever.url=http://127.0.0.1:${GRAPH_PORT}/retrieve \
    retriever.topk=10 \
    +env.name=appworld \
    +env.max_steps=40 \
    2>&1 | tee logs/eval_$NAME.log

grep -h "^\[Eval\]" logs/eval_$NAME.log
