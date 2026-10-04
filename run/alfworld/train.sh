#!/bin/bash
# ============================================================================
# ExpHarness on ALFWorld — train the retrieval copilot with PPO.
#
#   bash run/alfworld/train.sh [seed]
#
# Copilot  : $BASE_MODEL (default Qwen2.5-3B-Instruct), trained with PPO
# Executor : frozen LLM selected by AGENT_API (paper: Qwen3-32B / Gemini-3.1-Flash-Lite)
# Graph    : initialized from data/alfworld/cold_start_graph.json and evolved online
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults agentic

SEED=${1:-42}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
export ALFWORLD_NUM_WORKERS=${ALFWORLD_NUM_WORKERS:-16}   # parallel ALFWorld episodes
# export ALFWORLD_DATA=~/.cache/alfworld

GRAPH_PORT=${GRAPH_PORT:-8001}
GRAPH_INIT=${GRAPH_INIT:-data/alfworld/cold_start_graph.json}
EXPERIMENT=${EXPERIMENT:-expharness_alfworld_${SEED}}

start_graph_server "$GRAPH_PORT" "$GRAPH_INIT" 2000
ppo_common_args
mkdir -p logs

PYTHONUNBUFFERED=1 "$PYTHON" -m verl.trainer.main_expharness \
    "${PPO_COMMON[@]}" \
    data.train_files=data/alfworld/train.parquet \
    data.val_files=data/alfworld/val.parquet \
    data.val_batch_size=8 \
    +data.random_seed=$SEED \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.project_name=ExpHarness-ALFWorld \
    trainer.experiment_name=$EXPERIMENT \
    trainer.default_local_dir=checkpoints/$EXPERIMENT \
    +output_context_dir=outputs/$EXPERIMENT \
    retriever.url=http://127.0.0.1:${GRAPH_PORT}/retrieve \
    retriever.topk=10 \
    +graph_update.enable=true \
    +graph_update.graph_url=http://127.0.0.1:${GRAPH_PORT} \
    +graph_update.max_per_batch=10 \
    +env.name=alfworld \
    +env.max_steps=50 \
    +env.history_length=5 \
    2>&1 | tee logs/$EXPERIMENT.log
