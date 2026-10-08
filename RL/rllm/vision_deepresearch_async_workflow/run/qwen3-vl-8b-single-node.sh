#!/usr/bin/env bash
set -euo pipefail
# OneSearch-VL Qwen3-VL-8B launcher using the RL-10k recipe.

# cd to rllm/ so the Python workflow and its local dependencies resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
RLLM_DIR="$PWD"
RL_DIR="$(cd .. && pwd)"
RELEASE_ROOT="$(cd ../.. && pwd)"

# Load RL/.env, falling back to rllm/.env and root .env when absent.
# Keep shell tracing off while reading credentials, including under bash -x.
set +x
if [[ -n "${ENV_FILE:-}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
elif [[ -f "${RL_DIR}/.env" ]]; then
    set -a
    source "${RL_DIR}/.env"
    set +a
elif [[ -f .env ]]; then
    set -a
    source .env
    set +a
elif [[ -f "${RELEASE_ROOT}/.env" ]]; then
    set -a
    source "${RELEASE_ROOT}/.env"
    set +a
fi
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}

DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi

# ========= CUDA / Torch / Transformer Engine =========
# Use matching CUDA libraries from the training environment. Set LD_LIBRARY_PATH
# externally if system CUDA conflicts with that environment.
export PYTHONPATH="${RLLM_DIR}:${RLLM_DIR}/verl:${RL_DIR}/Megatron-LM:${RL_DIR}/mbridge${PYTHONPATH:+:${PYTHONPATH}}"

# ========= rollout =========
rollout_mode="async"
rollout_name="sglang"
if [[ "$rollout_mode" == async ]]; then
    return_raw_chat="True"
else
    return_raw_chat="False"
fi
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
dtype="bfloat16"
adv_estimator="rloo"

# ========= DAPO / KL / PPO clip =========
kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001
clip_ratio_high=0.28

# ========= batch / prepared RL-10k data =========
# 10,059 rows, one per task. Shuffle each epoch; stop after 201 updates.
export RL_DATASET_NAME="${RL_DATASET_NAME:-OneSearch-VL-RL-10K}"
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-256}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-64}
n_parallel_tasks=${N_PARALLEL_TASKS:-512}
n_parallel_tools=${N_PARALLEL_TOOLS:-1024}
data_shuffle=${DATA_SHUFFLE:-True}

# ========= length =========
max_prompt_length=40960
max_response_length=16384
use_dynamic_bsz=True
actor_ppo_max_token_len_per_gpu=57344
infer_ppo_max_token_len_per_gpu=57344

# ========= parallel / offload =========
offload=True
gen_tp=4
# ---- Megatron (dense 8B: no EP, no ETP) ----
train_tp=2
train_pp=1
train_cp=1

# ========= sampling =========
temperature=0.7
top_p=1.0
top_k=-1
val_temperature=0.7
val_top_p=0.95
calculate_log_probs=True
loss_agg_mode="seq-mean-token-sum"

# ========= SGLang multimodal runtime =========
gpu_mem_util=0.6
enforce_eager=True
max_num_seqs=256
max_total_tokens=196608
mm_max_concurrent_calls=4
mm_per_request_timeout=20
rllm_mm_cpu_offload=1
rllm_mm_cpu_threads=16
rllm_mm_cpu_reservation=8
rllm_server_max_restarts=2
rllm_server_max_task_retries=2
rllm_video_preprocess_cache_bytes=8589934592
rllm_video_preprocess_cache_entries=64
cuda_graph_max_bs=64
disable_overlap_schedule=true
schedule_policy=lpm

# ========= training =========
lr=1e-6
all_valid_max_attempts=8
workflow_timeout=1800
save_freq=${SAVE_FREQ:-10}
test_freq=-1
total_training_steps=${TOTAL_TRAINING_STEPS:-201}
log_episodes=true
resume_mode=${RESUME_MODE:-disable}
resume_from_path=${RESUME_FROM_PATH:-null}
# Split EVGR: independent execution-evidence trace and pixel-grounding judges.
# Total: format * (0.6 accuracy + 0.2 query + 0.2 mean(trace, ground)).
export FORMAL_EVGR_ENABLED=${FORMAL_EVGR_ENABLED:-1}
export FORMAL_EVGR_MODE=${FORMAL_EVGR_MODE:-both}
export FORMAL_EVGR_JUDGE_BACKEND=${FORMAL_EVGR_JUDGE_BACKEND:-seed}
# Weight defaults and validation live in reward_config.py.
export FORMAL_EVGR_MAX_CONCURRENCY=${FORMAL_EVGR_MAX_CONCURRENCY:-32}
export FORMAL_EVGR_EXPECTED_JUDGE_MODEL=${FORMAL_EVGR_EXPECTED_JUDGE_MODEL:-Doubao-Seed-2.0-pro}
export EVGR_JUDGE_MODEL=${EVGR_JUDGE_MODEL:-Doubao-Seed-2.0-pro}
export EVGR_JUDGE_MAX_TOKENS=${EVGR_JUDGE_MAX_TOKENS:-5000}
export EVGR_JUDGE_TIMEOUT=${EVGR_JUDGE_TIMEOUT:-180}
export EVGR_JUDGE_MAX_ATTEMPTS=${EVGR_JUDGE_MAX_ATTEMPTS:-2}
export FORMAL_EVGR_SCORE_VALIDATION=${FORMAL_EVGR_SCORE_VALIDATION:-0}
export FORMAL_EVGR_DATA_ROOT="${FORMAL_EVGR_DATA_ROOT:-${RELEASE_ROOT}/../OneSearch-VL-RL-10K}"
# Relative annotation roots are resolved against the code repository, not rllm/.
if [[ "$FORMAL_EVGR_DATA_ROOT" != /* ]]; then
    export FORMAL_EVGR_DATA_ROOT="${RELEASE_ROOT}/${FORMAL_EVGR_DATA_ROOT}"
fi
export QUERY_REWARD_ENABLED=${QUERY_REWARD_ENABLED:-1}

# Validate weights before loading CUDA, starting workers or submitting rollouts.
python3 -m vision_deepresearch_async_workflow.reward_config

if [[ "$DRY_RUN" == 0 ]]; then
    if [[ "$FORMAL_EVGR_ENABLED" == 1 || "$FORMAL_EVGR_ENABLED" == true || "$FORMAL_EVGR_ENABLED" == yes ]]; then
        python3 "${SCRIPT_DIR}/../reward_preflight.py" --data-root "$FORMAL_EVGR_DATA_ROOT" --check-env --check-registered
    fi
    python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
    python3 -c "import transformer_engine.pytorch; print('transformer_engine: ok')"
fi

# ========= cluster =========
# Start Ray externally for multi-node runs; invoke the driver once on the head.
NNODES=${NNODES:-1}
n_gpus_per_node=${GPUS_PER_NODE:-8}
project_name="onesearch-vl"
exp_name=${EXP_NAME:-onesearch_8b_10k_single_node}

# ========= paths =========
MODEL_PATH=${MODEL_PATH:-/models/your-sft-checkpoint}
OUTPUT_DIR=${OUTPUT_DIR:-${RLLM_DIR}/checkpoints/${project_name}/${exp_name}}
# Normalize relative output paths without creating directories during --dry-run.
if [[ "$OUTPUT_DIR" != /* ]]; then OUTPUT_DIR="${RLLM_DIR}/${OUTPUT_DIR}"; fi
CKPTS_DIR=${OUTPUT_DIR}/checkpoints
episode_log_dir=${OUTPUT_DIR}/episodes
export TRAJ_DUMP_DIR=${OUTPUT_DIR}/trajectory_dumps
export WANDB_DIR=${OUTPUT_DIR}/wandb
export TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard
export COS_UPLOAD_PATHS=${COS_UPLOAD_PATHS:-${RELEASE_ROOT}/onesearch_vl}

# --dry-run prints the exact Python command without imports, files or GPU work.
# The training command below stays in the same direct, multiline form as upstream.
if [[ "$DRY_RUN" == 1 ]]; then
    python3() { printf '%q ' python3 "$@"; printf '\n'; }
else
    if [[ "$MODEL_PATH" == /models/your-sft-checkpoint ]]; then
        printf '%s\n' 'Set MODEL_PATH to your SFT checkpoint.' >&2
        exit 1
    fi
    mkdir -p "$OUTPUT_DIR"
fi

python3 -m vision_deepresearch_async_workflow.train_deepresearch_workflow_megatron \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=64 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=${return_raw_chat} \
    data.seed=3407 \
    data.shuffle=${data_shuffle} \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.dtype=${dtype} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=${max_total_tokens} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_max_concurrent_calls=${mm_max_concurrent_calls} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mm_per_request_timeout=${mm_per_request_timeout} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.rllm_mm_cpu_offload=${rllm_mm_cpu_offload} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.rllm_mm_cpu_threads=${rllm_mm_cpu_threads} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.rllm_mm_cpu_reservation=${rllm_mm_cpu_reservation} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.rllm_server_max_restarts=${rllm_server_max_restarts} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.rllm_server_max_task_retries=${rllm_server_max_task_retries} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.rllm_video_preprocess_cache_bytes=${rllm_video_preprocess_cache_bytes} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.rllm_video_preprocess_cache_entries=${rllm_video_preprocess_cache_entries} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=${cuda_graph_max_bs} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.disable_overlap_schedule=${disable_overlap_schedule} \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.schedule_policy=${schedule_policy} \
    \
    actor_rollout_ref.ref.megatron.dtype=${dtype} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.megatron.dtype=${dtype} \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=${lr} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${offload} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=${n_parallel_tasks} \
    rllm.workflow.n_parallel_tools=${n_parallel_tools} \
    rllm.workflow.require_all_valid=False \
    rllm.workflow.all_valid_max_attempts=${all_valid_max_attempts} \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_unknown=True \
    rllm.compact_filtering.mask_error=True \
    rllm.compact_filtering.mask_timeout=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=False \
    ++rllm.workflow.workflow_args.timeout=${workflow_timeout} \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    +trainer.val_only=False \
    +trainer.final_val_after_train=False \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=100 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.log_episodes=${log_episodes} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.episode_log_dir="${episode_log_dir}" "$@"
