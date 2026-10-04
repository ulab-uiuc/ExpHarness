#!/bin/bash
# ============================================================================
# Shared settings for all ExpHarness run scripts (sourced by run/<task>/*.sh).
# Override any variable from the command line, e.g.
#   AGENT_API=nvidia NVIDIA_MODEL=meta/llama-3.1-8b-instruct bash run/static/train.sh all
# ============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
export PYTHONPATH="$ROOT:${PYTHONPATH}"

# ---------------------------------------------------------------- executor --
# Frozen executor LLM backend: gemini | nvidia | openai | claude
# (see expharness/utils/llm_api.py). Multiple keys can be comma-separated.
# The backend defaults to the paper setting of each task (see paper_defaults below);
# export AGENT_API=<backend> to use another one.
# export GEMINI_API_KEYS="your-key-1,your-key-2"
# export GEMINI_MODEL="gemini-3.1-flash-lite-preview"
# export NVIDIA_API_KEYS="nvapi-xxx,nvapi-yyy"
# export NVIDIA_MODEL="meta/llama-3.1-8b-instruct"
# export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"  OPENAI_MODEL="Qwen/Qwen3-32B"

# ---------------------------------------------------------------- copilot ---
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# ---------------------------------------------------------------- runtime ---
export VLLM_ATTENTION_BACKEND=XFORMERS
export TORCH_COMPILE_DISABLE=1
export RAY_memory_monitor_refresh_ms=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
unset RAY_ADDRESS
# export WANDB_API_KEY="your-wandb-key"

PYTHON="${PYTHON:-python3}"
# Interpreter / device for the experience graph server (Contriever runs fine on CPU)
GRAPH_PYTHON="${GRAPH_PYTHON:-$PYTHON}"
GRAPH_DEVICE="${GRAPH_DEVICE:-cpu}"

# Executor / summarizer defaults used in the paper (every value can be overridden
# by exporting it before running a script):
#   agentic  executor   Gemini-3.1-Flash-Lite (large) or Qwen3-32B via NVIDIA NIM (small)
#            summarizer Gemini-3.1-Flash-Lite
#   static   executor   Llama-3.2-3B-Instruct (small) or Llama-3.1-8B-Instruct (large) via NVIDIA NIM
#            summarizer Qwen2.5-3B-Instruct, zero-shot, served through an OpenAI-compatible
#                       endpoint, e.g.  vllm serve Qwen/Qwen2.5-3B-Instruct --port 8100
paper_defaults() {  # paper_defaults agentic|static
    if [ "$1" = static ]; then
        export AGENT_API="${AGENT_API:-nvidia}"
        export NVIDIA_MODEL="${NVIDIA_MODEL:-meta/llama-3.2-3b-instruct}"
        export SUMMARIZER_API="${SUMMARIZER_API:-openai}"
        export SUMMARIZER_MODEL="${SUMMARIZER_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
        export SUMMARIZER_BASE_URL="${SUMMARIZER_BASE_URL:-http://127.0.0.1:8100/v1}"
    else
        export AGENT_API="${AGENT_API:-gemini}"
        export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-flash-lite-preview}"
        export SUMMARIZER_API="${SUMMARIZER_API:-gemini}"
        export SUMMARIZER_MODEL="${SUMMARIZER_MODEL:-gemini-3.1-flash-lite-preview}"
    fi
    case "$AGENT_API" in   # name of the executor model (only used for logging)
        nvidia) EXECUTOR_MODEL="${NVIDIA_MODEL:-meta/llama-3.3-70b-instruct}" ;;
        openai) EXECUTOR_MODEL="${OPENAI_MODEL:-Qwen/Qwen2.5-7B-Instruct}" ;;
        claude) EXECUTOR_MODEL="${CLAUDE_MODEL:-claude-sonnet-4-20250514}" ;;
        *)      EXECUTOR_MODEL="${GEMINI_MODEL:-gemini-2.0-flash}" ;;
    esac
}

# ---------------------------------------------------------------- helpers ---
wait_for_url() {  # wait_for_url <url> <timeout_seconds>
    local url=$1 timeout=${2:-300}
    for _ in $(seq 1 "$timeout"); do
        curl -s -m 2 "$url" >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

GRAPH_PID=""
start_graph_server() {  # start_graph_server <port> <graph_json> [max_nodes]
    local port=$1 graph=$2 max_nodes=${3:-2000}
    if curl -s -m 2 "http://127.0.0.1:${port}/stats" >/dev/null 2>&1; then
        echo "[ExpHarness] Reusing experience graph server on port ${port}"
        return
    fi
    echo "[ExpHarness] Starting experience graph server: port=${port} graph=${graph} device=${GRAPH_DEVICE}"
    local gpu=""
    [ "$GRAPH_DEVICE" != "cpu" ] && gpu="${GRAPH_GPU:-0}"
    CUDA_VISIBLE_DEVICES="$gpu" "$GRAPH_PYTHON" -m expharness.graph.experience_graph_server \
        --port "$port" \
        --graph_path "$graph" \
        --contriever_path facebook/contriever \
        --device "$GRAPH_DEVICE" \
        --max_nodes "$max_nodes" \
        --k_neighbors 5 \
        --sim_threshold 0.3 &
    GRAPH_PID=$!
    if ! wait_for_url "http://127.0.0.1:${port}/stats" 300; then
        echo "[ExpHarness] ERROR: graph server did not come up on port ${port}"
        exit 1
    fi
}

cleanup() {
    [ -n "$GRAPH_PID" ] && kill "$GRAPH_PID" 2>/dev/null
    [ -n "$APPWORLD_PID" ] && kill "$APPWORLD_PID" 2>/dev/null
}
trap cleanup EXIT

# PPO settings shared by every task (paper, Appendix "Implementation Details"):
# copilot = $BASE_MODEL, AdamW lr 5e-6 (actor) / 1e-5 (critic) with cosine decay and no warmup,
# KL 0.01, max prompt 4096 / response 500 tokens, sampling temperature 1.0, vLLM memory 0.3,
# batch 16, up to 500 steps over 2 epochs, checkpoint every 20 steps, validation every 30 steps.
ppo_common_args() {
    PPO_COMMON=(
        data.train_data_num=null
        data.val_data_num=null
        data.max_prompt_length=4096
        data.max_response_length=500
        data.max_start_length=2000
        data.max_obs_length=1400
        data.shuffle_train_dataloader=True
        algorithm.adv_estimator=gae
        algorithm.kl_ctrl.kl_coef=0.01
        algorithm.no_think_rl=false
        actor_rollout_ref.model.path="$BASE_MODEL"
        actor_rollout_ref.model.enable_gradient_checkpointing=true
        actor_rollout_ref.model.use_remove_padding=True
        actor_rollout_ref.actor.optim.lr=5e-6
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0
        actor_rollout_ref.actor.optim.warmup_style=cosine
        actor_rollout_ref.actor.fsdp_config.param_offload=true
        actor_rollout_ref.actor.state_masking=true
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.temperature=1.0
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization=0.3
        actor_rollout_ref.rollout.log_prob_micro_batch_size=16
        +actor_rollout_ref.rollout.max_model_len=4096
        actor_rollout_ref.ref.log_prob_micro_batch_size=16
        actor_rollout_ref.ref.fsdp_config.param_offload=True
        critic.model.path="$BASE_MODEL"
        critic.model.enable_gradient_checkpointing=true
        critic.model.use_remove_padding=True
        critic.optim.lr=1e-5
        critic.optim.lr_warmup_steps_ratio=0
        critic.optim.warmup_style=cosine
        trainer.critic_warmup=0
        trainer.default_hdfs_dir=null
        trainer.nnodes=1
        data.train_batch_size=16
        actor_rollout_ref.actor.ppo_mini_batch_size=16
        actor_rollout_ref.actor.ppo_micro_batch_size=8
        critic.ppo_mini_batch_size=16
        critic.ppo_micro_batch_size=8
        trainer.total_epochs=2
        trainer.total_training_steps=500
        trainer.save_freq=20
        trainer.test_freq=30
        max_turns=1
        +generator_llm="$EXECUTOR_MODEL"
    )
}

# Settings shared by every evaluation run (greedy copilot, frozen graph)
eval_common_args() {  # eval_common_args <actor_checkpoint>
    local ckpt=$1
    EVAL_COMMON=(
        data.max_prompt_length=4096
        data.max_response_length=500
        data.max_start_length=2000
        data.max_obs_length=1400
        algorithm.adv_estimator=gae
        algorithm.kl_ctrl.kl_coef=0.01
        algorithm.no_think_rl=false
        actor_rollout_ref.model.path="$ckpt"
        actor_rollout_ref.actor.fsdp_config.param_offload=true
        actor_rollout_ref.actor.state_masking=true
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4
        +actor_rollout_ref.rollout.max_model_len=4096
        actor_rollout_ref.ref.fsdp_config.param_offload=True
        critic.model.path="$ckpt"
        critic.model.enable_gradient_checkpointing=true
        trainer.critic_warmup=0
        trainer.logger=[console]
        +trainer.val_only=true
        +trainer.val_before_train=true
        trainer.default_hdfs_dir=null
        trainer.nnodes=1
        trainer.total_epochs=1
        trainer.total_training_steps=1
        trainer.test_freq=1
        trainer.save_freq=-1
        +data.random_seed=42
        max_turns=1
        +generator_llm="$EXECUTOR_MODEL"
        +graph_update.enable=false
    )
}
