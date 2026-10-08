# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import dataclasses
import logging
import os
import sys
import time
from typing import Any, Optional

import ray
import sglang.srt.entrypoints.engine
import torch
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    _launch_subprocesses,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.request_stats import (
    summarize_scheduler_load,
)
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter, _set_envs_and_config
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _mm_cpu_settings(config) -> tuple[bool, int, int]:
    """Read MM runtime threads and Ray reservation independently."""
    engine_kwargs = config.get("engine_kwargs", {}).get("sglang", {}) or {}
    configured_enabled = engine_kwargs.get("rllm_mm_cpu_offload")
    configured_threads = engine_kwargs.get("rllm_mm_cpu_threads")
    configured_reservation = engine_kwargs.get("rllm_mm_cpu_reservation")
    enabled = (
        str(configured_enabled).lower() in {"1", "true", "yes", "on"}
        if configured_enabled is not None
        else os.environ.get("SGLANG_MM_CPU_OFFLOAD", "0") == "1"
    )
    threads = int(
        configured_threads
        if configured_threads is not None
        else os.environ.get("SGLANG_MM_CPU_THREADS", "32")
    )
    if threads < 1:
        raise ValueError(f"rllm_mm_cpu_threads must be positive, got {threads}")
    reservation = int(
        configured_reservation
        if configured_reservation is not None
        else os.environ.get("SGLANG_MM_CPU_RESERVATION", str(threads))
    )
    if reservation < 1:
        raise ValueError(
            "rllm_mm_cpu_reservation must be positive, "
            f"got {reservation}"
        )
    return enabled, threads, reservation


def _server_actor_fault_tolerance_settings(config) -> tuple[int, int]:
    """Return bounded Ray recovery limits for transient server-process loss."""
    engine_kwargs = config.get("engine_kwargs", {}).get("sglang", {}) or {}
    max_restarts = int(
        engine_kwargs.get(
            "rllm_server_max_restarts",
            os.environ.get("SGLANG_RLLM_SERVER_MAX_RESTARTS", "2"),
        )
    )
    max_task_retries = int(
        engine_kwargs.get(
            "rllm_server_max_task_retries",
            os.environ.get("SGLANG_RLLM_SERVER_MAX_TASK_RETRIES", "2"),
        )
    )
    if not 0 <= max_restarts <= 8:
        raise ValueError(
            "rllm_server_max_restarts must be in [0, 8], "
            f"got {max_restarts}"
        )
    if not 0 <= max_task_retries <= 8:
        raise ValueError(
            "rllm_server_max_task_retries must be in [0, 8], "
            f"got {max_task_retries}"
        )
    return max_restarts, max_task_retries


def _sglang_native_engine_kwargs(config) -> dict:
    """Return only kwargs owned by SGLang's native ``ServerArgs``.

    The ``rllm_`` namespace belongs to this adapter.  Keeping the boundary
    prefix-based prevents every new adapter setting from needing a second,
    easy-to-forget ``pop`` before ``ServerArgs(**kwargs)``.
    """
    engine_kwargs = dict(
        config.get("engine_kwargs", {}).get("sglang", {}) or {}
    )
    return {
        key: value
        for key, value in engine_kwargs.items()
        if not str(key).startswith("rllm_")
    }


def _prune_finished_request_states(tokenizer_manager) -> list[str]:
    """Remove scheduler-confirmed request tombstones before memory release.

    SGLang 0.5.3 removes normally completed requests in its batch-output
    handler.  Its waiting-queue abort handler is different: it marks the
    ``ReqState`` finished and wakes the HTTP coroutine, but leaves the entry in
    ``rid_to_state``.  If the client coroutine was already cancelled by rollout
    early-stop, nobody consumes that wake-up and the entry is permanent.

    A finished state is safe to remove because the scheduler has already sent
    the abort/completion acknowledgement.  Unfinished states are deliberately
    retained: deleting them would hide live scheduler work and make a forced
    KV-cache release unsafe.
    """

    states = getattr(tokenizer_manager, "rid_to_state", None)
    if not isinstance(states, dict):
        return []
    removed: list[str] = []
    for request_id, state in list(states.items()):
        if not bool(getattr(state, "finished", False)):
            continue
        # The actor event loop does not await between snapshot and pop, but the
        # identity guard makes the ownership condition explicit and robust to
        # a future callback that replaces a state under the same request ID.
        if states.get(request_id) is state:
            states.pop(request_id, None)
            removed.append(request_id)
    return removed


def _video_preprocess_cache_settings(config) -> tuple[int, int, int, float]:
    """Read the bounded per-server video decode/resize cache settings."""
    engine_kwargs = config.get("engine_kwargs", {}).get("sglang", {}) or {}
    max_bytes = int(
        engine_kwargs.get(
            "rllm_video_preprocess_cache_bytes",
            os.environ.get("SGLANG_RLLM_VIDEO_PREPROCESS_CACHE_BYTES", "0"),
        )
    )
    max_entries = int(
        engine_kwargs.get(
            "rllm_video_preprocess_cache_entries",
            os.environ.get("SGLANG_RLLM_VIDEO_PREPROCESS_CACHE_ENTRIES", "64"),
        )
    )
    log_every = int(
        engine_kwargs.get(
            "rllm_video_preprocess_log_every",
            os.environ.get("SGLANG_RLLM_VIDEO_PREPROCESS_LOG_EVERY", "50"),
        )
    )
    slow_log_s = float(
        engine_kwargs.get(
            "rllm_video_preprocess_slow_log_s",
            os.environ.get("SGLANG_RLLM_VIDEO_PREPROCESS_SLOW_LOG_S", "2"),
        )
    )
    if max_bytes < 0:
        raise ValueError(
            "rllm_video_preprocess_cache_bytes must be non-negative, "
            f"got {max_bytes}"
        )
    if max_entries < 1:
        raise ValueError(
            "rllm_video_preprocess_cache_entries must be positive, "
            f"got {max_entries}"
        )
    if log_every < 1:
        raise ValueError(
            "rllm_video_preprocess_log_every must be positive, "
            f"got {log_every}"
        )
    if slow_log_s < 0:
        raise ValueError(
            "rllm_video_preprocess_slow_log_s must be non-negative, "
            f"got {slow_log_s}"
        )
    return max_bytes, max_entries, log_every, slow_log_s


def _sglang_server_runtime_env(
    enabled: bool,
    threads: int,
    cuda_visible_devices: str | None = None,
) -> dict[str, dict[str, str]]:
    """Preserve opt-in SGLang settings when creating the nested server actor.

    Supplying ``runtime_env`` on a child actor replaces, rather than extends,
    the job-level environment inherited by that actor.  Keep the CUDA-visible
    devices workaround while explicitly carrying every SGLang setting into the
    process that owns TokenizerManager.
    """
    env_vars = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
    env_vars.update(
        {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("SGL_", "SGLANG_"))
        }
    )
    env_vars["SGLANG_MM_CPU_OFFLOAD"] = "1" if enabled else "0"
    env_vars["SGLANG_MM_CPU_THREADS"] = str(threads)
    # This must be present when Ray starts the actor process.  Setting it in
    # SGLangHttpServer.__init__ is too late if Ray, torch, or an imported
    # extension has already initialized CUDA: all nested HTTP/MM actors then
    # retain allocations on physical GPU 0 even though their scheduler child
    # processes are correctly spread across the node.
    if cuda_visible_devices is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if "CUDA_LAUNCH_BLOCKING" in os.environ:
        env_vars["CUDA_LAUNCH_BLOCKING"] = os.environ["CUDA_LAUNCH_BLOCKING"]
    return {"env_vars": env_vars}


# NOTE: the Qwen3-VL video rope fix is applied inside SGLangHttpServer.launch_server
# (see below), NOT here at module import time. Two reasons: (1) the code that runs
# get_rope_index lives in the SGLangHttpServer Ray actor process, and Ray rebuilds
# the @ray.remote class by qualname WITHOUT re-executing this module's top level,
# so a module-level apply() never runs in the actor; (2) importing via the rllm
# package (`from rllm.patches...`) drags in rllm/__init__ which does a
# signal.signal() registration that raises "signal only works in main thread" when
# launch_server runs in the actor's asyncio thread. The in-method injection loads
# the patch by absolute file path to avoid both problems.



@ray.remote(num_cpus=1)
class SGLangHttpServer:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        startup_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        cuda_initialized_before_init = torch.cuda.is_initialized()
        if startup_cuda_visible_devices != cuda_visible_devices:
            raise RuntimeError(
                "SGLangHttpServer CUDA visibility was not applied at process startup: "
                f"expected={cuda_visible_devices!r}, actual={startup_cuda_visible_devices!r}, "
                f"cuda_initialized={cuda_initialized_before_init}"
            )
        print(
            f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, "
            f"{cuda_visible_devices=}, startup_cuda_visible_devices={startup_cuda_visible_devices!r}, "
            f"cuda_initialized_before_init={cuda_initialized_before_init}",
            flush=True,
        )
        assert torch.cuda.is_available(), "SGLang http server should run on GPU node"

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes

        request_stats_every = (
            config.get("engine_kwargs", {})
            .get("sglang", {})
            .get("rllm_request_stats_log_every", 0)
        )
        self._request_stats_every = max(0, int(request_stats_every or 0))
        self._request_stats_active = 0
        self._request_stats_max_active = 0
        self._request_stats_completed = 0
        self._request_stats_prompt_tokens = 0
        self._request_stats_output_tokens = 0
        self._request_stats_elapsed_s = 0.0
        self._request_stats_cached_tokens = 0
        self._request_stats_queue_s = 0.0
        self._request_stats_queue_samples = 0
        self._request_stats_prefill_delay_s = 0.0
        self._request_stats_prefill_delay_samples = 0
        self._request_stats_prefill_latency_s = 0.0
        self._request_stats_prefill_latency_samples = 0
        self._request_stats_decode_s = 0.0
        self._request_stats_decode_tokens = 0
        self._request_stats_seen_request_ids: set[str] = set()
        now = time.monotonic()
        self._request_stats_clock_s = now
        self._request_stats_active_area = 0.0
        self._request_stats_busy_s = 0.0
        self._request_stats_report_s = now
        self._request_stats_report_completed = 0
        self._request_stats_report_prompt_tokens = 0
        self._request_stats_report_cached_tokens = 0
        self._request_stats_report_output_tokens = 0
        self._request_stats_report_elapsed_s = 0.0
        self._request_stats_report_active_area = 0.0
        self._request_stats_report_busy_s = 0.0

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for NCCL process group
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            logger.info(
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        engine_kwargs = _sglang_native_engine_kwargs(self.config)
        mm_cpu_enabled, mm_cpu_threads, mm_cpu_reservation = (
            _mm_cpu_settings(self.config)
        )
        (
            video_cache_bytes,
            video_cache_entries,
            video_cache_log_every,
            video_cache_slow_log_s,
        ) = _video_preprocess_cache_settings(self.config)
        os.environ["SGLANG_MM_CPU_OFFLOAD"] = "1" if mm_cpu_enabled else "0"
        os.environ["SGLANG_MM_CPU_THREADS"] = str(mm_cpu_threads)
        os.environ["SGLANG_RLLM_VIDEO_PREPROCESS_CACHE_BYTES"] = str(
            video_cache_bytes
        )
        os.environ["SGLANG_RLLM_VIDEO_PREPROCESS_CACHE_ENTRIES"] = str(
            video_cache_entries
        )
        os.environ["SGLANG_RLLM_VIDEO_PREPROCESS_LOG_EVERY"] = str(
            video_cache_log_every
        )
        os.environ["SGLANG_RLLM_VIDEO_PREPROCESS_SLOW_LOG_S"] = str(
            video_cache_slow_log_s
        )
        print(
            "[mm-cpu-offload] actor config "
            f"enabled={mm_cpu_enabled} threads={mm_cpu_threads} "
            f"ray_cpus={mm_cpu_reservation} source=rollout-config "
            f"cuda_launch_blocking={os.environ.get('CUDA_LAUNCH_BLOCKING', '<unset>')}",
            flush=True,
        )
        print(
            "[mm-video-preprocess-cache] actor config "
            f"enabled={video_cache_bytes > 0} max_bytes={video_cache_bytes} "
            f"max_entries={video_cache_entries} log_every={video_cache_log_every} "
            f"slow_log_s={video_cache_slow_log_s}",
            flush=True,
        )
        attention_backend = engine_kwargs.pop("attention_backend", None)
        dist_init_addr = (
            f"[{self._master_address}]:{self._master_port}"
            if is_valid_ipv6_address(self._master_address)
            else f"{self._master_address}:{self._master_port}"
        )

        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": self.config.tensor_model_parallel_size,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "dist_init_addr": dist_init_addr,
            "nnodes": self.nnodes,
            "nccl_port": 30000 + self.replica_rank * 8 + self.node_rank + 1,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = ServerArgs(**args)
        # Apply the Qwen3-VL video rope fix INSIDE this actor process, right
        # before _launch_subprocesses spins up the TokenizerManager (which runs
        # get_rope_index in this same process). The module-top-level apply()
        # above only runs in whichever process imported this module first (the
        # driver), NOT in this Ray actor worker — Ray rebuilds the @ray.remote
        # class by qualname without re-executing module side effects. This
        # instance-method call is guaranteed to run in the actor process, before
        # any video request hits get_rope_index. Idempotent.
        #
        # Load the patch by ABSOLUTE FILE PATH, not `from rllm.patches...`:
        # importing via the rllm package triggers rllm/__init__ -> ... -> a
        # signal.signal() registration, which raises "signal only works in main
        # thread" because launch_server runs in the actor's asyncio thread. The
        # patch file itself only depends on torch/sglang, so file-load is safe.
        try:
            import importlib.util as _ilu

            _patch_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..",  # verl/verl/workers/rollout/sglang_rollout -> rllm/
                "rllm", "patches", "sglang_qwen3vl_video_rope.py",
            )
            _patch_file = os.path.abspath(_patch_file)
            _spec = _ilu.spec_from_file_location("sglang_qwen3vl_video_rope", _patch_file)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _applied = _mod.apply()
            logger.info("qwen3_vl video rope patch applied=%s (in actor, pre-launch, file=%s)", _applied, _patch_file)
        except Exception as _e:  # noqa: BLE001
            logger.warning("qwen3_vl video rope patch (actor) skipped: %s", _e)
        try:
            import importlib.util as _ilu2

            _fps_patch_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..",  # -> rllm/
                "rllm", "patches", "sglang_qwen3vl_video_fps.py",
            )
            _fps_patch_file = os.path.abspath(_fps_patch_file)
            _fps_spec = _ilu2.spec_from_file_location("sglang_qwen3vl_video_fps", _fps_patch_file)
            _fps_mod = _ilu2.module_from_spec(_fps_spec)
            _fps_spec.loader.exec_module(_fps_mod)
            _fps_applied = _fps_mod.apply()
            logger.info(
                "qwen3_vl video fps patch applied=%s (in actor, pre-launch, file=%s)",
                _fps_applied,
                _fps_patch_file,
            )
        except Exception as _e:  # noqa: BLE001
            logger.warning("qwen3_vl video fps patch (actor) skipped: %s", _e)
        try:
            import importlib.util as _ilu3

            _ds_patch_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..",  # -> rllm/
                "rllm", "patches", "sglang_qwen3vl_deepstack.py",
            )
            _ds_patch_file = os.path.abspath(_ds_patch_file)
            _ds_spec = _ilu3.spec_from_file_location("sglang_qwen3vl_deepstack", _ds_patch_file)
            _ds_mod = _ilu3.module_from_spec(_ds_spec)
            _ds_spec.loader.exec_module(_ds_mod)
            _ds_applied = _ds_mod.apply()
            logger.info(
                "qwen3_vl deepstack patch applied=%s (in actor, pre-launch, file=%s)",
                _ds_applied,
                _ds_patch_file,
            )
        except Exception as _e:  # noqa: BLE001
            logger.warning("qwen3_vl deepstack patch (actor) skipped: %s", _e)
        try:
            import importlib.util as _ilu_sft_preprocess

            _sft_preprocess_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..",  # -> rllm/
                "rllm", "patches", "sglang_qwen3vl_sft_preprocess.py",
            )
            _sft_preprocess_file = os.path.abspath(_sft_preprocess_file)
            _sft_preprocess_spec = _ilu_sft_preprocess.spec_from_file_location(
                "sglang_qwen3vl_sft_preprocess", _sft_preprocess_file
            )
            _sft_preprocess_mod = _ilu_sft_preprocess.module_from_spec(
                _sft_preprocess_spec
            )
            _sft_preprocess_spec.loader.exec_module(_sft_preprocess_mod)
            if not _sft_preprocess_mod.apply() or not _sft_preprocess_mod.is_applied():
                raise RuntimeError("SFT preprocessing patch did not install")
            logger.info(
                "qwen3_vl SFT preprocessing patch verified (file=%s)",
                _sft_preprocess_file,
            )
        except Exception as _e:  # noqa: BLE001
            raise RuntimeError(
                "Qwen3-VL SFT preprocessing alignment is required for RL rollout"
            ) from _e
        # Multi-turn RL sends the same video path on every turn. Cache Qwen's
        # prompt-independent video decode/resize result before wrapping the
        # complete processor in the CPU executor below. This ordering lets the
        # executor call the cached method, while prompt-dependent HF processing
        # and mRoPE are still recomputed on every request.
        try:
            import importlib.util as _ilu4

            _video_cache_patch_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..",  # -> rllm/
                "rllm", "patches", "sglang_qwen_vl_video_preprocess_cache.py",
            )
            _video_cache_patch_file = os.path.abspath(_video_cache_patch_file)
            _video_cache_spec = _ilu4.spec_from_file_location(
                "sglang_qwen_vl_video_preprocess_cache",
                _video_cache_patch_file,
            )
            _video_cache_mod = _ilu4.module_from_spec(_video_cache_spec)
            _video_cache_spec.loader.exec_module(_video_cache_mod)
            _video_cache_applied = _video_cache_mod.apply()
            _video_cache_enabled = video_cache_bytes > 0
            _video_cache_verified = _video_cache_mod.is_applied()
            if _video_cache_enabled and not _video_cache_verified:
                raise RuntimeError(
                    "video preprocessing cache was requested but its Qwen-VL "
                    "wrappers are not installed"
                )
            logger.info(
                "qwen_vl video preprocessing cache enabled=%s applied_now=%s "
                "verified=%s bytes=%s entries=%s (file=%s)",
                _video_cache_enabled,
                _video_cache_applied,
                _video_cache_verified,
                video_cache_bytes,
                video_cache_entries,
                _video_cache_patch_file,
            )
            print(
                "[mm-video-preprocess-cache] actor verification "
                f"enabled={_video_cache_enabled} "
                f"applied_now={_video_cache_applied} "
                f"verified={_video_cache_verified}",
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            logger.warning("qwen_vl video preprocessing cache skipped: %s", _e)
            print(
                "[mm-video-preprocess-cache] actor initialization failed "
                f"enabled={video_cache_bytes > 0} "
                f"error={type(_e).__name__}: {_e}",
                flush=True,
            )
            if video_cache_bytes > 0:
                raise RuntimeError(
                    "Qwen-VL video preprocessing cache was requested but actor "
                    "patch initialization failed"
                ) from _e

        # The scheduler's built-in GPU VLM embedding cache has no hit/eviction
        # counters. SGLang forces multiprocessing ``spawn``, so patching the
        # class in this HTTP/Ray actor is not inherited by the scheduler. Load
        # an importable top-level wrapper and replace SGLang's process target;
        # that wrapper installs the counters inside the actual GPU scheduler.
        try:
            import importlib.util as _ilu_vlm_cache

            os.environ["SGLANG_RLLM_REPLICA_TAG"] = (
                f"r{chr(ord('a') + int(self.replica_rank))}"
            )
            _vlm_cache_patch_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..",  # -> rllm/
                "rllm", "patches", "sglang_vlm_embedding_cache_stats.py",
            )
            _vlm_cache_patch_file = os.path.abspath(_vlm_cache_patch_file)
            _vlm_cache_patch_dir = os.path.dirname(_vlm_cache_patch_file)
            if _vlm_cache_patch_dir not in sys.path:
                # multiprocessing.spawn copies sys.path and imports the target
                # by module name in the child interpreter.
                sys.path.insert(0, _vlm_cache_patch_dir)
            _vlm_cache_spec = _ilu_vlm_cache.spec_from_file_location(
                "sglang_vlm_embedding_cache_stats", _vlm_cache_patch_file
            )
            _vlm_cache_mod = _ilu_vlm_cache.module_from_spec(_vlm_cache_spec)
            sys.modules[_vlm_cache_spec.name] = _vlm_cache_mod
            _vlm_cache_spec.loader.exec_module(_vlm_cache_mod)
            _vlm_cache_applied = _vlm_cache_mod.apply()
            sglang.srt.entrypoints.engine.run_scheduler_process = (
                _vlm_cache_mod.run_scheduler_process_with_vlm_cache_stats
            )
            print(
                "[vlm-embedding-cache] actor verification "
                f"applied_now={_vlm_cache_applied} "
                f"verified={_vlm_cache_mod.is_applied()} "
                "scheduler_target=instrumented "
                "log_every="
                f"{os.environ.get('SGLANG_RLLM_VLM_CACHE_STATS_LOG_EVERY', '0')} "
                f"replica_tag={os.environ['SGLANG_RLLM_REPLICA_TAG']}",
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            logger.warning("SGLang VLM embedding cache stats skipped: %s", _e)
            if int(
                os.environ.get("SGLANG_RLLM_VLM_CACHE_STATS_LOG_EVERY", "0")
            ) > 0:
                raise RuntimeError(
                    "SGLang VLM embedding cache stats were requested but patch "
                    "initialization failed"
                ) from _e

        # SGLang 0.5.3's Qwen-VL "async" processor executes PIL/video/HF CPU
        # work synchronously.  At rollout fan-out this serializes the actor
        # event loop and causes its default 10-second MM deadline to expire.
        # Offload the complete processor call to the bounded CPU pool before
        # TokenizerManager captures the method.
        try:
            import importlib.util as _ilu5

            _mm_cpu_patch_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "..", "..",  # -> rllm/
                "rllm", "patches", "sglang_qwen_vl_mm_cpu_offload.py",
            )
            _mm_cpu_patch_file = os.path.abspath(_mm_cpu_patch_file)
            _mm_cpu_spec = _ilu5.spec_from_file_location(
                "sglang_qwen_vl_mm_cpu_offload", _mm_cpu_patch_file
            )
            _mm_cpu_mod = _ilu5.module_from_spec(_mm_cpu_spec)
            _mm_cpu_spec.loader.exec_module(_mm_cpu_mod)
            _mm_cpu_applied = _mm_cpu_mod.apply()
            self._mm_cpu_stats_snapshot = _mm_cpu_mod.stats_snapshot
            _mm_cpu_enabled = os.environ.get("SGLANG_MM_CPU_OFFLOAD", "0") == "1"
            if _mm_cpu_enabled and not _mm_cpu_mod.is_applied():
                raise RuntimeError(
                    "SGLANG_MM_CPU_OFFLOAD=1 but Qwen-VL CPU offload patch "
                    "is not installed in the SGLang server actor"
                )
            logger.info(
                "qwen_vl MM CPU offload patch enabled=%s applied_now=%s "
                "verified=%s (threads=%s, file=%s)",
                _mm_cpu_enabled,
                _mm_cpu_applied,
                _mm_cpu_mod.is_applied(),
                os.environ.get("SGLANG_MM_CPU_THREADS", "32"),
                _mm_cpu_patch_file,
            )
            print(
                "[mm-cpu-offload] actor verification "
                f"enabled={_mm_cpu_enabled} applied_now={_mm_cpu_applied} "
                f"verified={_mm_cpu_mod.is_applied()} "
                f"threads={os.environ.get('SGLANG_MM_CPU_THREADS', '32')}",
                flush=True,
            )
        except Exception as _e:  # noqa: BLE001
            logger.warning("qwen_vl MM CPU offload patch skipped: %s", _e)
            print(
                "[mm-cpu-offload] actor initialization failed "
                f"enabled={os.environ.get('SGLANG_MM_CPU_OFFLOAD')} "
                f"error={type(_e).__name__}: {_e}",
                flush=True,
            )
            if os.environ.get("SGLANG_MM_CPU_OFFLOAD", "0") == "1":
                raise RuntimeError(
                    "SGLANG_MM_CPU_OFFLOAD was requested but actor patch "
                    "initialization failed"
                ) from _e
        self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
            server_args=server_args
        )

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_args = (server_args, None, None)
        self._server_port, self._server_task = await run_unvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            # FIXME(@wuxibin): sglang seems resume with random weights.
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")
        self._reset_request_stats_window()

    async def sleep(self):
        _DRAIN_TIMEOUT_S = 120.0
        _POLL_INTERVAL_S = 0.05
        try:
            self.tokenizer_manager.abort_request(abort_all=True)
        except Exception as e:
            logger.warning(f"abort_request(abort_all=True) before sleep failed: {e}")
        # Wait for rid_to_state to drain after abort signal reaches the scheduler.
        _drain_elapsed = 0.0
        _pruned_total = 0
        _initial_count = len(
            getattr(self.tokenizer_manager, "rid_to_state", {})
        )
        while _drain_elapsed < _DRAIN_TIMEOUT_S:
            _pruned = _prune_finished_request_states(self.tokenizer_manager)
            _pruned_total += len(_pruned)
            try:
                _n = len(getattr(self.tokenizer_manager, "rid_to_state", {}))
            except Exception:
                _n = 0
            if _n == 0:
                if _initial_count or _pruned_total:
                    logger.info(
                        "[sglang-abort-drain] initial=%s pruned_finished=%s "
                        "unresolved=0 elapsed_s=%.2f",
                        _initial_count,
                        _pruned_total,
                        _drain_elapsed,
                    )
                break
            await asyncio.sleep(_POLL_INTERVAL_S)
            _drain_elapsed += _POLL_INTERVAL_S
        else:
            # Never force a release while unfinished scheduler work remains.
            # SGLang asserts that release_memory_occupation has no ongoing
            # requests; violating that invariant can hang the release RPC or
            # corrupt the engine for every later training step.  A bounded,
            # explicit failure lets Ray fault-tolerance restart the server and
            # is strictly safer than an unbounded global gather stall.
            raise RuntimeError(
                "SGLang abort drain did not quiesce: "
                f"initial={_initial_count}, pruned_finished={_pruned_total}, "
                f"unresolved={_n}, timeout_s={_DRAIN_TIMEOUT_S}; "
                "refusing unsafe memory release"
            )
        if self.rollout_mode == RolloutMode.HYBRID:
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    def _advance_request_stats_clock(self, now: float) -> None:
        """Integrate in-flight request occupancy up to ``now``.

        Request latency alone cannot distinguish an underfed replica from a
        slow, saturated one.  The time integral gives a sampling-independent
        mean active count and the fraction of wall time with any work present.
        This actor's event loop updates the counters without an intervening
        await, so the accounting operation is atomic with respect to generate
        arrivals and completions.
        """
        elapsed = max(0.0, now - self._request_stats_clock_s)
        self._request_stats_active_area += self._request_stats_active * elapsed
        if self._request_stats_active > 0:
            self._request_stats_busy_s += elapsed
        self._request_stats_clock_s = now

    def _reset_request_stats_window(self) -> None:
        """Exclude model wake/training gaps from the next rollout window."""
        now = time.monotonic()
        self._advance_request_stats_clock(now)
        self._request_stats_report_s = now
        self._request_stats_report_completed = self._request_stats_completed
        self._request_stats_report_prompt_tokens = (
            self._request_stats_prompt_tokens
        )
        self._request_stats_report_cached_tokens = (
            self._request_stats_cached_tokens
        )
        self._request_stats_report_output_tokens = (
            self._request_stats_output_tokens
        )
        self._request_stats_report_elapsed_s = self._request_stats_elapsed_s
        self._request_stats_report_active_area = (
            self._request_stats_active_area
        )
        self._request_stats_report_busy_s = self._request_stats_busy_s

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(prompt_ids) - 1)
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = GenerateReqInput(
            rid=request_id,
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            image_data=image_data,
            video_data=video_data,
        )
        first_request_on_replica = request_id not in self._request_stats_seen_request_ids
        if first_request_on_replica:
            self._request_stats_seen_request_ids.add(request_id)
        route_started_epoch_s = time.time()
        route_raw_prompt_tokens = len(prompt_ids)
        started = time.monotonic()
        self._advance_request_stats_clock(started)
        self._request_stats_active += 1
        self._request_stats_max_active = max(
            self._request_stats_max_active, self._request_stats_active
        )
        output = None
        try:
            output = await self.tokenizer_manager.generate_request(request, None).__anext__()
        finally:
            finished = time.monotonic()
            self._advance_request_stats_clock(finished)
            elapsed = finished - started
            self._request_stats_active = max(0, self._request_stats_active - 1)
            self._request_stats_completed += 1
            self._request_stats_prompt_tokens += len(prompt_ids)
            self._request_stats_elapsed_s += elapsed
            if output is not None:
                output_tokens = len(output.get("output_ids", ()))
                self._request_stats_output_tokens += output_tokens
                meta_info = output.get("meta_info", {}) or {}
                # SGLang's prompt_tokens includes expanded image/video tokens;
                # len(prompt_ids) does not. Correct the provisional raw-token
                # count so cached_tokens uses the same denominator.
                meta_prompt_tokens = int(
                    meta_info.get("prompt_tokens", len(prompt_ids)) or len(prompt_ids)
                )
                self._request_stats_prompt_tokens += meta_prompt_tokens - len(prompt_ids)
                self._request_stats_cached_tokens += int(
                    meta_info.get("cached_tokens", 0) or 0
                )
                if first_request_on_replica:
                    print(
                        "[sglang-request-route] "
                        f"start_epoch_s={route_started_epoch_s:.6f} "
                        f"end_epoch_s={time.time():.6f} "
                        f"replica_tag=r{chr(ord('a') + int(self.replica_rank))} "
                        f"replica={self.replica_rank} request_id={request_id} "
                        f"raw_prompt_tokens={route_raw_prompt_tokens} "
                        f"expanded_prompt_tokens={meta_prompt_tokens} "
                        f"cached_tokens={int(meta_info.get('cached_tokens', 0) or 0)} "
                        f"elapsed_s={elapsed:.6f} "
                        f"image_items={len(image_data or ())} "
                        f"video_items={len(video_data or ())}",
                        flush=True,
                    )
                for metric, total_attr, count_attr in (
                    ("queue_time", "_request_stats_queue_s", "_request_stats_queue_samples"),
                    (
                        "prefill_delay",
                        "_request_stats_prefill_delay_s",
                        "_request_stats_prefill_delay_samples",
                    ),
                    (
                        "prefill_latency",
                        "_request_stats_prefill_latency_s",
                        "_request_stats_prefill_latency_samples",
                    ),
                ):
                    value = meta_info.get(metric)
                    if value is not None:
                        setattr(self, total_attr, getattr(self, total_attr) + float(value))
                        setattr(self, count_attr, getattr(self, count_attr) + 1)
                time_per_token = meta_info.get("time_per_token")
                if time_per_token is not None and output_tokens:
                    self._request_stats_decode_s += float(time_per_token) * output_tokens
                    self._request_stats_decode_tokens += output_tokens
                elif output_tokens:
                    prefill_finished = meta_info.get("prefill_finished_ts")
                    decode_finished = meta_info.get("decode_finished_ts")
                    if prefill_finished is not None and decode_finished is not None:
                        decode_s = float(decode_finished) - float(prefill_finished)
                        if decode_s > 0:
                            self._request_stats_decode_s += decode_s
                            self._request_stats_decode_tokens += output_tokens
            if (
                self._request_stats_every
                and self._request_stats_completed % self._request_stats_every == 0
            ):
                report_now = time.monotonic()
                self._advance_request_stats_clock(report_now)
                completed = self._request_stats_completed
                prompt_tokens = self._request_stats_prompt_tokens
                cache_ratio = (
                    self._request_stats_cached_tokens / prompt_tokens
                    if prompt_tokens
                    else 0.0
                )
                queue_s = (
                    self._request_stats_queue_s / self._request_stats_queue_samples
                    if self._request_stats_queue_samples
                    else 0.0
                )
                prefill_delay_s = (
                    self._request_stats_prefill_delay_s
                    / self._request_stats_prefill_delay_samples
                    if self._request_stats_prefill_delay_samples
                    else 0.0
                )
                prefill_latency_s = (
                    self._request_stats_prefill_latency_s
                    / self._request_stats_prefill_latency_samples
                    if self._request_stats_prefill_latency_samples
                    else 0.0
                )
                decode_tok_s = (
                    self._request_stats_decode_tokens / self._request_stats_decode_s
                    if self._request_stats_decode_s > 0
                    else 0.0
                )
                window_s = max(1e-9, report_now - self._request_stats_report_s)
                window_completed = (
                    completed - self._request_stats_report_completed
                )
                window_prompt_tokens = (
                    prompt_tokens - self._request_stats_report_prompt_tokens
                )
                window_cached_tokens = (
                    self._request_stats_cached_tokens
                    - self._request_stats_report_cached_tokens
                )
                window_output_tokens = (
                    self._request_stats_output_tokens
                    - self._request_stats_report_output_tokens
                )
                window_elapsed_s = (
                    self._request_stats_elapsed_s
                    - self._request_stats_report_elapsed_s
                )
                window_active_area = (
                    self._request_stats_active_area
                    - self._request_stats_report_active_area
                )
                window_busy_s = (
                    self._request_stats_busy_s
                    - self._request_stats_report_busy_s
                )
                window_cache_ratio = (
                    window_cached_tokens / window_prompt_tokens
                    if window_prompt_tokens
                    else 0.0
                )
                # The actor-level active counter begins before multimodal
                # preprocessing, whereas SGLang's scheduler load only counts
                # requests that have reached the model scheduler. Their gap
                # distinguishes MM/tokenizer starvation from an already-loaded
                # but compute-inefficient scheduler. Query only at the existing
                # reporting cadence and never fail generation on metrics errors.
                scheduler_load = {
                    "ranks": 0,
                    "num_reqs": -1,
                    "num_running_reqs": -1,
                    "num_waiting_reqs": -1,
                    "num_tokens": -1,
                }
                try:
                    scheduler_load = summarize_scheduler_load(
                        await asyncio.wait_for(
                            self.tokenizer_manager.get_load(), timeout=1.0
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SGLang scheduler load sampling failed: %s", exc)
                preprocess_inflight = (
                    max(0, self._request_stats_active - scheduler_load["num_reqs"])
                    if scheduler_load["num_reqs"] >= 0
                    else -1
                )
                mm_cpu_stats = (
                    self._mm_cpu_stats_snapshot()
                    if hasattr(self, "_mm_cpu_stats_snapshot")
                    else {}
                )
                mm_cpu_completed = int(mm_cpu_stats.get("completed", 0))
                mm_cpu_mean_queue_s = (
                    float(mm_cpu_stats.get("queue_total_s", 0.0))
                    / mm_cpu_completed
                    if mm_cpu_completed
                    else 0.0
                )
                mm_cpu_mean_service_s = (
                    float(mm_cpu_stats.get("service_total_s", 0.0))
                    / mm_cpu_completed
                    if mm_cpu_completed
                    else 0.0
                )
                print(
                    "[sglang-request-stats] "
                    f"epoch_s={time.time():.6f} "
                    f"replica_tag=r{chr(ord('a') + int(self.replica_rank))} "
                    f"replica={self.replica_rank} completed={completed} "
                    f"active={self._request_stats_active} "
                    f"max_active={self._request_stats_max_active} "
                    f"mean_prompt_tokens={self._request_stats_prompt_tokens / completed:.1f} "
                    f"mean_output_tokens={self._request_stats_output_tokens / completed:.1f} "
                    f"mean_server_s={self._request_stats_elapsed_s / completed:.3f} "
                    f"cache_ratio={cache_ratio:.4f} "
                    f"mean_queue_s={queue_s:.3f} "
                    f"mean_prefill_delay_s={prefill_delay_s:.3f} "
                    f"mean_prefill_latency_s={prefill_latency_s:.3f} "
                    f"decode_tok_s={decode_tok_s:.2f} "
                    f"window_s={window_s:.3f} "
                    f"window_completed_s={window_completed / window_s:.3f} "
                    f"window_prompt_tok_s={window_prompt_tokens / window_s:.1f} "
                    f"window_uncached_prompt_tok_s="
                    f"{(window_prompt_tokens - window_cached_tokens) / window_s:.1f} "
                    f"window_output_tok_s={window_output_tokens / window_s:.1f} "
                    f"window_cache_ratio={window_cache_ratio:.4f} "
                    f"window_mean_active={window_active_area / window_s:.2f} "
                    f"window_busy_frac={window_busy_s / window_s:.4f} "
                    f"scheduler_ranks={scheduler_load['ranks']} "
                    f"scheduler_reqs={scheduler_load['num_reqs']} "
                    f"scheduler_running={scheduler_load['num_running_reqs']} "
                    f"scheduler_waiting={scheduler_load['num_waiting_reqs']} "
                    f"scheduler_tokens={scheduler_load['num_tokens']} "
                    f"preprocess_inflight={preprocess_inflight} "
                    f"mm_cpu_calls={int(mm_cpu_stats.get('calls', 0))} "
                    f"mm_cpu_completed={mm_cpu_completed} "
                    f"mm_cpu_active={int(mm_cpu_stats.get('active', 0))} "
                    f"mm_cpu_max_active={int(mm_cpu_stats.get('max_active', 0))} "
                    f"mm_cpu_mean_queue_s={mm_cpu_mean_queue_s:.6f} "
                    f"mm_cpu_max_queue_s={float(mm_cpu_stats.get('queue_max_s', 0.0)):.6f} "
                    f"mm_cpu_mean_service_s={mm_cpu_mean_service_s:.6f} "
                    f"mm_cpu_max_service_s={float(mm_cpu_stats.get('service_max_s', 0.0)):.6f} "
                    f"mm_cpu_cancelled={int(mm_cpu_stats.get('cancelled', 0))} "
                    f"mm_video_calls={int(mm_cpu_stats.get('video_calls', 0))} "
                    f"mm_video_completed={int(mm_cpu_stats.get('video_completed', 0))} "
                    f"mm_video_active={int(mm_cpu_stats.get('video_active', 0))} "
                    f"mm_video_max_active={int(mm_cpu_stats.get('video_max_active', 0))} "
                    f"mm_video_waiters={int(mm_cpu_stats.get('video_waiters', 0))} "
                    f"mm_video_max_waiters={int(mm_cpu_stats.get('video_max_waiters', 0))} "
                    f"mm_video_gate_wait_total_s={float(mm_cpu_stats.get('video_gate_wait_total_s', 0.0)):.6f} "
                    f"mm_video_gate_wait_max_s={float(mm_cpu_stats.get('video_gate_wait_max_s', 0.0)):.6f} "
                    f"window_mean_server_s="
                    f"{window_elapsed_s / max(1, window_completed):.3f}",
                    flush=True,
                )
                self._request_stats_report_s = report_now
                self._request_stats_report_completed = completed
                self._request_stats_report_prompt_tokens = prompt_tokens
                self._request_stats_report_cached_tokens = (
                    self._request_stats_cached_tokens
                )
                self._request_stats_report_output_tokens = (
                    self._request_stats_output_tokens
                )
                self._request_stats_report_elapsed_s = (
                    self._request_stats_elapsed_s
                )
                self._request_stats_report_active_area = (
                    self._request_stats_active_area
                )
                self._request_stats_report_busy_s = self._request_stats_busy_s
        if return_logprob:
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None
        meta_info = output.get("meta_info", {}) or {}
        prompt_tokens = int(
            meta_info.get("prompt_tokens", len(prompt_ids)) or len(prompt_ids)
        )
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            prompt_tokens=prompt_tokens,
        )


_rollout_worker_actor_cls = ray.remote(ServerAdapter)


class SGLangReplica(RolloutReplica):
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ["CUDA_VISIBLE_DEVICES"])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        mm_cpu_enabled, mm_cpu_threads, mm_cpu_reservation = (
            _mm_cpu_settings(self.config)
        )
        server_max_restarts, server_max_task_retries = (
            _server_actor_fault_tolerance_settings(self.config)
        )
        print(
            "[mm-cpu-offload] replica config "
            f"enabled={mm_cpu_enabled} threads={mm_cpu_threads} "
            f"ray_cpus={mm_cpu_reservation}",
            flush=True,
        )
        print(
            "[sglang-actor-fault-tolerance] "
            f"max_restarts={server_max_restarts} "
            f"max_task_retries={server_max_task_retries}",
            flush=True,
        )

        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            )
            node_id = worker_node_ids[node_rank * self.gpus_per_node]
            name = (
                f"sglang_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"sglang_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = SGLangHttpServer.options(
                num_cpus=(
                    mm_cpu_reservation if mm_cpu_enabled else 1
                ),
                max_restarts=server_max_restarts,
                max_task_retries=server_max_task_retries,
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env=_sglang_server_runtime_env(
                    mm_cpu_enabled,
                    mm_cpu_threads,
                    node_cuda_visible_devices,
                ),
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
