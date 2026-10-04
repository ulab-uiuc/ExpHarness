#!/bin/bash
# ============================================================================
# ExpHarness on AppWorld — train the retrieval copilot with PPO.
#
#   bash run/appworld/start_servers.sh     # once, in the AppWorld environment
#   bash run/appworld/train.sh [seed]
# ============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
paper_defaults agentic

SEED=${1:-42}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

# AppWorld episode servers (see run/appworld/start_servers.sh)
export APPWORLD_SERVER_URL=${APPWORLD_SERVER_URL:-http://127.0.0.1:8006,8007,8008,8009}
export APPWORLD_MAX_STEPS=40
export APPWORLD_POOL_WORKERS=${APPWORLD_POOL_WORKERS:-4}
export APPWORLD_REQUEST_TIMEOUT=${APPWORLD_REQUEST_TIMEOUT:-150}
export APPWORLD_BASELINE_REFRESH_EVERY=0   # cache the no-experience baseline per task

GRAPH_PORT=${GRAPH_PORT:-8010}
GRAPH_INIT=${GRAPH_INIT:-data/appworld/cold_start_graph.json}
EXPERIMENT=${EXPERIMENT:-expharness_appworld_${SEED}}

start_graph_server "$GRAPH_PORT" "$GRAPH_INIT" 2000
ppo_common_args
mkdir -p logs

PYTHONUNBUFFERED=1 "$PYTHON" -m verl.trainer.main_expharness \
    "${PPO_COMMON[@]}" \
    data.train_files=data/appworld/train.parquet \
    data.val_files=data/appworld/val.parquet \
    data.val_batch_size=8 \
    +data.random_seed=$SEED \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.project_name=ExpHarness-AppWorld \
    trainer.experiment_name=$EXPERIMENT \
    trainer.default_local_dir=checkpoints/$EXPERIMENT \
    +output_context_dir=outputs/$EXPERIMENT \
    retriever.url=http://127.0.0.1:${GRAPH_PORT}/retrieve \
    retriever.topk=10 \
    +graph_update.enable=true \
    +graph_update.graph_url=http://127.0.0.1:${GRAPH_PORT} \
    +graph_update.max_per_batch=10 \
    +env.name=appworld \
    +env.max_steps=40 \
    2>&1 | tee logs/$EXPERIMENT.log
