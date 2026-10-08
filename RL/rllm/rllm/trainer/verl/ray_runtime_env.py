import os

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "VLLM_USE_V1": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
    "worker_process_setup_hook": "rllm.patches.verl_patch_hook.setup",
}

FORWARD_PREFIXES = [
    "VLLM_",
    "SGL_",
    "SGLANG_",
    "HF_",
    "TOKENIZERS_",
    "DATASETS_",
    "TORCH_",
    "PYTORCH_",
    "DEEPSPEED_",
    "MEGATRON_",
    "NCCL_",
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "NV_",
    "NVIDIA_",
    "NVTE_",
    # Ray actors must inherit the runtime linker path so the source-built
    # transformer_engine can find the env-bundled cuDNN/cuBLAS (.so). Without
    # this, actor workers fail to import TE and never register with the raylet.
    "LD_LIBRARY_PATH",
    # Tool + reward credentials/config must reach the rollout/reward actors too,
    # otherwise judge falls back to api.openai.com (unreachable) and search tools
    # get empty keys. These are set in rllm/.env (sourced by the launch script in
    # the driver) but the actors are fresh processes that only see forwarded vars.
    "JUDGE_",            # query-utility + accuracy judge endpoint/key/model
    "ACCURACY_JUDGE_",   # explicit accuracy model independently of EVGR
    "POLARS_MAX_THREADS",
    "FORMAL_EVGR_",      # evidence data and scoring controls
    "REWARD_",           # combined reward weights
    "QUERY_REWARD_",     # query reward enable and configuration
    "EVGR_JUDGE_",       # shared Trace/Ground judge settings
    "EVGR_SPLIT_",       # evidence text/image limits and validation retries
    "SEED_",             # Trace/Ground judge endpoint, model and credentials
    "JINA_", "QWEN_",   # reader and OpenAI-compatible summarizer
    "TEXT_SEARCH_", "IMAGE_SEARCH_", "SEARCH_",
    "LAYOUT_PARSING_", "LOCAL_VISUAL_", "VISIT_",
    "RLLM_",             # tool concurrency + rollout runtime controls
    "SERPER_", "SERP_",
    "API_HOST", "API_USER", "API_KEY", "API_GATEWAY_",
    "ASSET_", "COS_UPLOAD_", "COS_USERID", "IMAGE_CROP_CACHE",
    "no_proxy", "NO_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "WANDB_",
    "RL_DATASET_NAME",
]

# These values describe the host on which a process is running.  Forwarding
# them through Ray's job-level runtime_env copies the driver's NIC/identity to
# actors scheduled on other nodes.  Each raylet already inherits the values
# selected by that node's worker launcher, and Megatron workers additionally
# resolve them from Ray's node IP immediately before process-group creation.
NODE_LOCAL_ENV_VARS = {
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_HOSTID",
}


def _get_forwarded_env_vars():
    """
    Get the forwarded environment variables. The `RLLM_EXCLUDE` environment variable can be used to
    exclude specific environment variables or all variables with a specific prefix.

    Example:
    ```
    RLLM_EXCLUDE=VLLM*,CUDA*,NCCL_IB_DISABLE
    ```
    will exclude all variables with prefix `VLLM_`, `CUDA_`, and `NCCL_IB_DISABLE`.

    By default, all environment variables with prefix in `FORWARD_PREFIXES` are forwarded.
    """
    if os.environ.get("RLLM_EXCLUDE", None) is not None:
        rllm_exclude = str(os.environ.get("RLLM_EXCLUDE")).split(",")
    else:
        rllm_exclude = []

    forward_prefix = FORWARD_PREFIXES.copy()

    # RLLM_EXCLUDE controls construction of this payload; it is not itself a
    # runtime setting for remote actors.
    exclude_vars = {"RLLM_EXCLUDE"}
    for name in rllm_exclude:
        if "*" in name:  # denote a prefix match, e.g. "VLLM*"
            forward_prefix.remove(name.replace("*", "_"))
        else:
            exclude_vars.add(name)

    exclude_vars.update(NODE_LOCAL_ENV_VARS)
    forwarded = {
        key: value
        for key, value in os.environ.items()
        if any(key.startswith(prefix) for prefix in forward_prefix) and key not in exclude_vars
    }
    return forwarded


def get_ppo_ray_runtime_env():
    env = PPO_RAY_RUNTIME_ENV["env_vars"].copy()
    env.update(_get_forwarded_env_vars())
    return {
        "env_vars": env,
        # "worker_process_setup_hook": PPO_RAY_RUNTIME_ENV["worker_process_setup_hook"],
    }
