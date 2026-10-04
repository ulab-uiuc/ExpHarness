#!/bin/bash
# ============================================================================
# Evaluate a trained ExpHarness copilot on ALFWorld Seen (140) / Unseen (134).
#
#   bash run/alfworld/eval.sh <experiment_dir> <step>
#   e.g. bash run/alfworld/eval.sh checkpoints/expharness_alfworld_42 300
#
# Uses the copilot checkpoint  <experiment_dir>/actor/global_step_<step>
# and the evolved graph        <experiment_dir>/graph_step_<step>.json (kept frozen).
# Results: outputs/eval/<name>_<split>/eval_summary.json  (success rate + avg. steps)
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
export ALFWORLD_NUM_WORKERS=${ALFWORLD_NUM_WORKERS:-8}
GRAPH_PORT=${GRAPH_PORT:-8011}

start_graph_server "$GRAPH_PORT" "$GRAPH" 2000
eval_common_args "$ACTOR_CKPT"
mkdir -p logs

for SPLIT in eval_in_distribution eval_out_of_distribution; do
    if [ "$SPLIT" = eval_in_distribution ]; then PARQUET=data/alfworld/val_seen.parquet;   LABEL=seen
    else                                          PARQUET=data/alfworld/val_unseen.parquet; LABEL=unseen; fi
    echo "=== ALFWorld ${LABEL} ==="
    PYTHONUNBUFFERED=1 "$PYTHON" -m verl.trainer.main_expharness \
        "${EVAL_COMMON[@]}" \
        data.train_files=$PARQUET \
        data.val_files=$PARQUET \
        data.train_batch_size=16 \
        data.val_batch_size=16 \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.project_name=ExpHarness-ALFWorld-Eval \
        trainer.experiment_name=${NAME}_${LABEL} \
        +output_context_dir=outputs/eval/${NAME}_${LABEL} \
        retriever.url=http://127.0.0.1:${GRAPH_PORT}/retrieve \
        retriever.topk=10 \
        +env.name=alfworld \
        +env.max_steps=50 \
        +eval.split=$SPLIT \
        2>&1 | tee logs/eval_${NAME}_${LABEL}.log
done

grep -h "^\[Eval\]" logs/eval_${NAME}_seen.log logs/eval_${NAME}_unseen.log
