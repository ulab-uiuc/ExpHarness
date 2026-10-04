#!/bin/bash
# ============================================================================
# ExpHarness on ExpSuite-Static — train the retrieval copilot with PPO.
#
#   bash run/static/train.sh <domain> [seed]
#
#   domain = all        all 10 benchmarks (QA + reasoning + coding)
#            reasoning  GSM8K / GSM-Symbolic / MATH
#            coding     HumanEval+ / MBPP+
#
# Paper executors: Llama-3.2-3B-Instruct (small) / Llama-3.1-8B-Instruct (large), e.g.
#   AGENT_API=nvidia NVIDIA_MODEL=meta/llama-3.2-3b-instruct bash run/static/train.sh all
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults static

DOMAIN=${1:-all}
SEED=${2:-42}
case "$DOMAIN" in
    all)       PORT_DEFAULT=8002 ;;
    reasoning) PORT_DEFAULT=8003 ;;
    coding)    PORT_DEFAULT=8004 ;;
    *) echo "unknown domain: $DOMAIN (all | reasoning | coding)"; exit 1 ;;
esac
DATA=data/static/$DOMAIN

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export QA_NUM_WORKERS=${QA_NUM_WORKERS:-16}   # parallel executor calls

GRAPH_PORT=${GRAPH_PORT:-$PORT_DEFAULT}
GRAPH_INIT=${GRAPH_INIT:-$DATA/cold_start_graph.json}
EXPERIMENT=${EXPERIMENT:-expharness_static_${DOMAIN}_${SEED}}

start_graph_server "$GRAPH_PORT" "$GRAPH_INIT" 2000
ppo_common_args
mkdir -p logs

PYTHONUNBUFFERED=1 "$PYTHON" -m verl.trainer.main_expharness \
    "${PPO_COMMON[@]}" \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/val.parquet \
    data.val_batch_size=16 \
    +data.random_seed=$SEED \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.project_name=ExpHarness-Static \
    trainer.experiment_name=$EXPERIMENT \
    trainer.default_local_dir=checkpoints/$EXPERIMENT \
    +output_context_dir=outputs/$EXPERIMENT \
    retriever.url=http://127.0.0.1:${GRAPH_PORT}/retrieve \
    retriever.topk=10 \
    +graph_update.enable=true \
    +graph_update.graph_url=http://127.0.0.1:${GRAPH_PORT} \
    +graph_update.max_per_batch=10 \
    +env.name=qa \
    2>&1 | tee logs/$EXPERIMENT.log
