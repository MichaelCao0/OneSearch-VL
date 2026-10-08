import asyncio
import os
import uuid
import time

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AsyncLLMServerManager
from verl.workers.rollout.replica import TokenOutput

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser
from rllm.workflows import TerminationEvent, TerminationReason


class VerlEngine(RolloutEngine):
    supports_mm_training_input_deferral = True

    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config

        if config.actor_rollout_ref.rollout.name not in ["vllm", "sglang"]:
            raise ValueError(f"VerlEngine only supports vllm or sglang rollout, but got {config.actor_rollout_ref.rollout.name}")

        self.rollout_manager: AgentLoopManager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, server_handles=rollout_manager.server_handles)
        self.tokenizer = tokenizer
        self.processor = processor
        self.chat_parser = ChatTemplateParser.get_parser(tokenizer, processor=processor, disable_thinking=config.get("rllm", {}).get("disable_thinking", False))

        self.max_prompt_length = config.data.max_prompt_length
        self.max_response_length = config.data.max_response_length
        self.accumulate_reasoning = config.get("rllm", {}).get("accumulate_reasoning", False)

        # With cumulative (non-stepwise) advantages, multi-turn trajectories are
        # re-tokenized from their final chat transcript by AgentWorkflowEngine.
        # The per-turn ``prompt_ids`` and ``multi_modal_inputs`` produced here are
        # consumed only when a trajectory ends after its first model response.
        # Keep the first turn exact, but allow later turns to skip TaskRunner-side
        # video decoding + HF processor work. SGLang still receives the identical
        # raw video URLs and prepared PIL images, so generation and tool semantics
        # do not change.
        defer_requested = os.environ.get(
            "RLLM_DEFER_UNUSED_MM_TRAINING_INPUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        stepwise_advantage = bool(
            config.get("rllm", {}).get("stepwise_advantage", {}).get("enable", False)
        )
        self.defer_unused_mm_training_inputs = (
            defer_requested and not stepwise_advantage
        )
        print(
            "defer_unused_mm_training_inputs: "
            f"{self.defer_unused_mm_training_inputs} "
            f"(requested={defer_requested}, stepwise_advantage={stepwise_advantage})"
        )

        # Video decoding and the HF multimodal processor are synchronous CPU
        # work.  Calling them directly from get_model_response serializes every
        # trajectory on the TaskRunner asyncio event loop, leaving rollout GPUs
        # idle even when n_parallel_tasks is large.  Offload those calls to a
        # bounded thread pool.  The bound is deliberately configurable because
        # decoded video tensors are large and unbounded (e.g. 256-way) decode
        # concurrency can exhaust host memory.
        self.mm_preprocess_concurrency = max(
            1, int(os.environ.get("RLLM_MM_PREPROCESS_CONCURRENCY", "8"))
        )
        self._mm_preprocess_semaphore = asyncio.Semaphore(
            self.mm_preprocess_concurrency
        )
        # These counters are mutated only on the TaskRunner event loop.  They
        # let us distinguish CPU service from admission waiting without
        # changing the concrete media/tool implementation.
        self._mm_preprocess_active = 0
        self._mm_preprocess_waiters = 0
        print(
            "multimodal_preprocess_concurrency: "
            f"{self.mm_preprocess_concurrency} (asyncio.to_thread)"
        )

        self.rollout_group_affinity = os.environ.get(
            "RLLM_ROLLOUT_GROUP_AFFINITY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.rollout_group_affinity_shards = max(
            1,
            int(os.environ.get("RLLM_ROLLOUT_GROUP_AFFINITY_SHARDS", "1")),
        )
        print(
            "rollout_group_affinity: "
            f"{self.rollout_group_affinity} "
            f"shards={self.rollout_group_affinity_shards}"
        )

        self.train_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.do_sample is False else config.actor_rollout_ref.rollout.temperature,
            top_k=config.actor_rollout_ref.rollout.top_k,
            top_p=config.actor_rollout_ref.rollout.top_p,
        )

        self.val_sampling_params = dict(
            temperature=0.0 if config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else config.actor_rollout_ref.rollout.val_kwargs.temperature,
            top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
            top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        )

        print(f"train_sampling_params: {self.train_sampling_params}")
        print(f"val_sampling_params: {self.val_sampling_params}")

        self.validate = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks_verl

    async def _run_mm_preprocess(self, func, /, *args, **kwargs):
        """Run blocking MM preprocessing and return admission/service stats."""
        queued_at = time.perf_counter()
        saturated_at_submit = self._mm_preprocess_semaphore.locked()
        self._mm_preprocess_waiters += 1
        waiter_counted = True
        try:
            async with self._mm_preprocess_semaphore:
                queue_s = time.perf_counter() - queued_at
                self._mm_preprocess_waiters -= 1
                waiter_counted = False
                self._mm_preprocess_active += 1
                active_at_admit = self._mm_preprocess_active
                waiters_at_admit = self._mm_preprocess_waiters
                service_started = time.perf_counter()
                try:
                    value = await asyncio.to_thread(func, *args, **kwargs)
                finally:
                    service_s = time.perf_counter() - service_started
                    self._mm_preprocess_active -= 1
        finally:
            # Cancellation while waiting must not leak the diagnostic counter.
            if waiter_counted:
                self._mm_preprocess_waiters -= 1

        return value, {
            "queue_s": queue_s,
            "service_s": service_s,
            "active_at_admit": active_at_admit,
            "waiters_at_admit": waiters_at_admit,
            "saturated_at_submit": saturated_at_submit,
        }

    def _materialize_mm_training_inputs(self, messages: list[dict]) -> bool:
        """Whether this turn's HF processor output can reach the trainer.

        In cumulative mode, AgentWorkflowEngine discards every per-turn
        ModelOutput once a second assistant turn exists. We cannot know on the
        first call whether the trajectory will finish immediately, so the first
        call remains fully materialized. A prior assistant message proves that
        this is a multi-turn trajectory and its per-turn training payload will be
        discarded.
        """
        if not self.defer_unused_mm_training_inputs:
            return True
        return not any(message.get("role") == "assistant" for message in messages)

    @staticmethod
    def _group_routing_key(application_id: str, num_shards: int = 1) -> str:
        """Route rollout siblings through one or more stable prefix shards.

        One shard preserves the original behaviour: all responses sampled for
        a prompt share one SGLang replica and its prefix cache.  More shards
        bound the amount of heterogeneous long-video work pinned to one GPU
        while retaining prefix reuse within each shard.
        """
        task_id, separator, rollout_index = application_id.rpartition(":")
        if separator and task_id and rollout_index.isdigit():
            if num_shards > 1:
                shard = int(rollout_index) % num_shards
                return f"{task_id}:affinity-shard-{shard}"
            return task_id
        return application_id

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        with self.chat_parser.mm_source_cache.capture_request_stats() as cache_stats:
            return await self._get_model_response(
                messages, mm_cache_stats=cache_stats, **kwargs
            )

    async def _get_model_response(
        self, messages: list[dict], *, mm_cache_stats, **kwargs
    ) -> ModelOutput:
        request_started = time.perf_counter()
        video_prepare_s = 0.0
        image_prepare_s = 0.0
        processor_s = 0.0
        mm_preprocess_queue_s = 0.0
        mm_preprocess_service_s = 0.0
        mm_preprocess_ops = 0
        mm_preprocess_waited_ops = 0
        mm_preprocess_saturated_ops = 0
        mm_preprocess_max_active = 0
        mm_preprocess_max_waiters = 0
        mm_training_inputs_deferred = 0
        mm_training_inputs_materialized = 0
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)
        enforce_max_prompt_length = kwargs.pop("enforce_max_prompt_length", True)
        materialize_override = kwargs.pop(
            "materialize_mm_training_inputs", None
        )
        if materialize_override is not None and not isinstance(
            materialize_override, bool
        ):
            raise TypeError(
                "materialize_mm_training_inputs must be a bool when provided"
            )

        # these go to the parser
        tools = kwargs.pop("tools", [])
        accumulate_reasoning = kwargs.pop("accumulate_reasoning", self.accumulate_reasoning)

        sampling_params = self.val_sampling_params.copy() if self.validate or validate else self.train_sampling_params.copy()
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", sampling_params.pop("max_new_tokens", self.max_response_length))

        prompt_started = time.perf_counter()
        prompt = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=accumulate_reasoning)
        request_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)  # list[int]
        prompt_build_s = time.perf_counter() - prompt_started

        _has_images = any(msg.get("images") is not None and msg["role"] == "user" for msg in messages)
        _has_videos = any(msg.get("videos") is not None and msg["role"] == "user" for msg in messages)

        image_data = None
        video_data = None
        materialize_mm_training_inputs = (
            self._materialize_mm_training_inputs(messages)
            if materialize_override is None
            else materialize_override
        )

        if (_has_videos or _has_images) and self.processor is not None:
            proc_kwargs = {"text": [prompt]}

            if _has_videos:
                # sglang's load_video only accepts str (URL/path) or bytes, so we
                # pass the raw video URL strings as video_data and hand the
                # decoded tensors to the processor for token/grid computation.
                video_urls = []
                for msg in messages:
                    if msg.get("videos") and msg["role"] == "user":
                        for v in msg["videos"]:
                            url = v["video"] if isinstance(v, dict) else v
                            video_urls.append(url)
                # SGLang consumes the raw URLs. Decoded tensors exist only for
                # the TaskRunner-side HF processor/training payload, so later
                # cumulative turns can bypass video decode entirely.
                video_data = video_urls or None
                if materialize_mm_training_inputs:
                    video_prepare_started = time.perf_counter()
                    video_tensors, mm_stats = await self._run_mm_preprocess(
                        self.chat_parser.process_video_data, messages
                    )  # list[Tensor (T,C,H,W)]
                    mm_preprocess_queue_s += mm_stats["queue_s"]
                    mm_preprocess_service_s += mm_stats["service_s"]
                    mm_preprocess_ops += 1
                    mm_preprocess_waited_ops += int(mm_stats["queue_s"] >= 0.001)
                    mm_preprocess_saturated_ops += int(
                        mm_stats["saturated_at_submit"]
                    )
                    mm_preprocess_max_active = max(
                        mm_preprocess_max_active, mm_stats["active_at_admit"]
                    )
                    mm_preprocess_max_waiters = max(
                        mm_preprocess_max_waiters, mm_stats["waiters_at_admit"]
                    )
                    video_prepare_s += time.perf_counter() - video_prepare_started
                    if video_tensors:
                        proc_kwargs["videos"] = video_tensors
                        # fps for TIMESTAMP scale, NOT sampling (do_sample_frames=
                        # False keeps all frames). grid.mp4 is composed at 1 fps, so
                        # correct <t seconds> tags need fps=1.0; using the sampling
                        # fps=2.0 would halve every timestamp. Must match the sglang
                        # server patch. Env-overridable via VIDEO_GRID_FPS.
                        _vfps = float(os.environ.get("VIDEO_GRID_FPS", "1.0"))
                        _vmeta = []
                        _vmeta_ok = True
                        for _vt in video_tensors:
                            try:
                                _n = int(_vt.shape[0])
                            except Exception:
                                _vmeta_ok = False
                                break
                            _vmeta.append({
                                "total_num_frames": _n,
                                "fps": _vfps,
                                "duration": float(_n) / _vfps,
                                "frames_indices": list(range(_n)),
                            })
                        if _vmeta_ok and _vmeta:
                            proc_kwargs["video_metadata"] = _vmeta
                            proc_kwargs["do_sample_frames"] = False

            if _has_images:
                image_prepare_started = time.perf_counter()
                images, mm_stats = await self._run_mm_preprocess(
                    self.chat_parser.process_image_data, messages
                )  # list[PIL.Image.Image]
                mm_preprocess_queue_s += mm_stats["queue_s"]
                mm_preprocess_service_s += mm_stats["service_s"]
                mm_preprocess_ops += 1
                mm_preprocess_waited_ops += int(mm_stats["queue_s"] >= 0.001)
                mm_preprocess_saturated_ops += int(
                    mm_stats["saturated_at_submit"]
                )
                mm_preprocess_max_active = max(
                    mm_preprocess_max_active, mm_stats["active_at_admit"]
                )
                mm_preprocess_max_waiters = max(
                    mm_preprocess_max_waiters, mm_stats["waiters_at_admit"]
                )
                image_prepare_s += time.perf_counter() - image_prepare_started
                if images:
                    image_data = images
                    if materialize_mm_training_inputs:
                        proc_kwargs["images"] = images

            # Only invoke the processor if at least one modality actually yielded
            # data; otherwise fall back to the plain text prompt ids.
            if materialize_mm_training_inputs and (
                "videos" in proc_kwargs or "images" in proc_kwargs
            ):
                processor_started = time.perf_counter()
                model_inputs, mm_stats = await self._run_mm_preprocess(
                    self.processor, **proc_kwargs
                )
                mm_preprocess_queue_s += mm_stats["queue_s"]
                mm_preprocess_service_s += mm_stats["service_s"]
                mm_preprocess_ops += 1
                mm_preprocess_waited_ops += int(mm_stats["queue_s"] >= 0.001)
                mm_preprocess_saturated_ops += int(
                    mm_stats["saturated_at_submit"]
                )
                mm_preprocess_max_active = max(
                    mm_preprocess_max_active, mm_stats["active_at_admit"]
                )
                mm_preprocess_max_waiters = max(
                    mm_preprocess_max_waiters, mm_stats["waiters_at_admit"]
                )
                processor_s += time.perf_counter() - processor_started
                prompt_ids = model_inputs.pop("input_ids")[0]
                model_inputs.pop("attention_mask", None)
                multi_modal_inputs = dict(model_inputs)
                mm_training_inputs_materialized = 1
            else:
                multi_modal_inputs = None
                prompt_ids = request_prompt_ids
                mm_training_inputs_deferred = int(
                    not materialize_mm_training_inputs
                )
        else:
            multi_modal_inputs = None
            prompt_ids = request_prompt_ids

        prompt_length = len(prompt_ids)
        if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
            raise TerminationEvent(TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED)

        preprocess_total_s = time.perf_counter() - request_started
        generate_started = time.perf_counter()
        routing_key = (
            self._group_routing_key(
                application_id,
                getattr(self, "rollout_group_affinity_shards", 1),
            )
            if self.rollout_group_affinity
            else None
        )
        token_output: TokenOutput = await self.server_manager.generate(
            request_id=application_id,
            routing_key=routing_key,
            prompt_ids=request_prompt_ids,
            image_data=image_data,
            video_data=video_data,
            sampling_params=sampling_params,
        )  # type: ignore
        server_generate_s = time.perf_counter() - generate_started
        # When the caller explicitly skips TaskRunner-side HF preprocessing,
        # the local prompt ids contain only one placeholder per image/video.
        # SGLang reports the post-expansion count in ``prompt_tokens``; retain
        # that ground-truth length for context checks and diagnostics without
        # putting the discarded training payload back on the critical path.
        server_prompt_tokens = getattr(token_output, "prompt_tokens", None)
        if (
            not materialize_mm_training_inputs
            and server_prompt_tokens is not None
        ):
            prompt_length = int(server_prompt_tokens)
            if enforce_max_prompt_length and prompt_length > self.max_prompt_length:
                raise TerminationEvent(
                    TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED
                )
        response_parse_started = time.perf_counter()
        completion_ids: list[int] = token_output.token_ids

        finish_reason = "stop"
        if len(completion_ids) >= max_tokens:
            finish_reason = "length"
            completion_ids = completion_ids[:max_tokens]

        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        # TODO: implement parse_completion for the standard parser
        parsed_output = self.chat_parser.parse_completion(completion_ids)
        cache_after = self.chat_parser.mm_source_cache.snapshot()

        return ModelOutput(
            text=completion_text,
            content=parsed_output["content"],
            reasoning=parsed_output["reasoning"],
            tool_calls=parsed_output["tool_calls"],
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            multi_modal_inputs=multi_modal_inputs,
            logprobs=[],
            prompt_length=prompt_length,
            completion_length=len(completion_ids),
            finish_reason=finish_reason,
            timing={
                "prompt_build_s": prompt_build_s,
                "video_prepare_s": video_prepare_s,
                "image_prepare_s": image_prepare_s,
                "processor_s": processor_s,
                "mm_preprocess_queue_s": mm_preprocess_queue_s,
                "mm_preprocess_service_s": mm_preprocess_service_s,
                "mm_preprocess_ops": mm_preprocess_ops,
                "mm_preprocess_waited_ops": mm_preprocess_waited_ops,
                "mm_preprocess_saturated_ops": mm_preprocess_saturated_ops,
                "mm_preprocess_max_active": mm_preprocess_max_active,
                "mm_preprocess_max_waiters": mm_preprocess_max_waiters,
                "mm_training_inputs_deferred": mm_training_inputs_deferred,
                "mm_training_inputs_materialized": (
                    mm_training_inputs_materialized
                ),
                "server_prompt_tokens": float(
                    server_prompt_tokens
                    if server_prompt_tokens is not None
                    else prompt_length
                ),
                **{
                    f"mm_source_cache_{key}": value
                    for key, value in mm_cache_stats.items()
                },
                "mm_source_cache_entries": cache_after["entries"],
                "mm_source_cache_bytes": cache_after["bytes"],
                "preprocess_total_s": preprocess_total_s,
                "server_generate_s": server_generate_s,
                "response_parse_s": time.perf_counter() - response_parse_started,
                "request_total_s": time.perf_counter() - request_started,
            },
        )

    async def wake_up(self):
        """Wake up all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.wake_up() for replica in self.rollout_manager.rollout_replicas])

    async def sleep(self):
        """Sleep all rollout replica instances asynchronously."""
        await asyncio.gather(*[replica.sleep() for replica in self.rollout_manager.rollout_replicas])
