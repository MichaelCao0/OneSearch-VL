from __future__ import annotations

import asyncio
import io
import json
import os
import time
import uuid
from typing import Any, List, Optional

import requests
from PIL import Image
import re
from collections import Counter
import json5
from vision_deepresearch_async_workflow.tool_statistics import CallStatistics, observe_call


_MALFORMED_TOOL_CALL_RE = re.compile(
    r'^\s*\{\s*"name"\s*:\s*(?P<name>"(?:\\.|[^"\\])*")'
    r"\s*,\s*(?P<arguments>\{.*\})\s*\}\s*$",
    re.DOTALL,
)


def _parse_tool_call_json(tool_call_text: str) -> dict[str, Any]:
    """Parse a tool call, with one opt-in repair for a known eval-only typo.

    The checkpoint occasionally emits ``{"name":"tool", {args}}`` instead
    of ``{"name":"tool", "arguments":{args}}``.  The repair accepts only
    that exact outer shape and still parses both captured values as JSON; it
    never evaluates model-generated text.  Training/default behavior remains
    unchanged unless ``RLLM_REPAIR_MALFORMED_TOOL_JSON=1`` is exported.
    """
    try:
        parsed = json5.loads(tool_call_text)
    except Exception:
        if os.environ.get("RLLM_REPAIR_MALFORMED_TOOL_JSON", "0") != "1":
            raise
        match = _MALFORMED_TOOL_CALL_RE.fullmatch(tool_call_text)
        if match is None:
            raise
        parsed = {
            "name": json.loads(match.group("name")),
            "arguments": json.loads(match.group("arguments")),
        }
    if not isinstance(parsed, dict):
        raise ValueError("tool call must be a JSON object")
    return parsed


# rLLM imports
from rllm.engine.rollout import RolloutEngine
from vision_deepresearch_async_workflow.tools.shared import (
    ToolCallTimeout,
    run_blocking_bounded,
)

from vision_deepresearch_async_workflow.prompts import get_system_prompt, tool_names

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = int(os.environ.get("MAX_LLM_CALL_PER_RUN", "25"))
_SFT_MAX_LLM_CALLS = {"single_image": 8, "multi_image": 12, "video": 12}
MAX_PROMPT_LENGTH_PER_RUN = 64000
MAX_RESPONSE_LENGTH_PER_RUN = int(os.environ.get("MAX_RESPONSE_LENGTH_PER_RUN", "4096"))
INTERMEDIATE_ROOT = os.environ.get(
    "RLLM_TOOL_TMP_DIR",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "vdr_tools"),
)

_VIDEO_ONLY_TOOL_NAMES = frozenset({"select_timespan", "select_frame"})


def _tool_is_active(name: str, modality: str) -> bool:
    return name in tool_names(modality)


def _select_system_prompt(tools: dict | None = None, modality: str = "video") -> str:
    """Use the same complete system slot as inference for this modality."""
    if tools is not None:
        missing = tool_names(modality) - tools.keys()
        if missing:
            raise ValueError(f"Missing prompt tools for {modality}: {sorted(missing)}")
    return get_system_prompt(modality)


def analyze_repetition_ngram(text: str, n: int = 30, threshold: float = 0.5):
    """
    Use N-grams to detect repetition in a string.

    Args:
        text (str): Input text to analyze.
        n (int): N-gram window size (default 10).
            - For long repetitive sequences, 10-20 is recommended.
        threshold (float): Distinct-N threshold (0~1).
            - Values below this indicate heavy repetition (default 0.5).

    Returns:
        bool: True if repetition is detected, False otherwise.
    """
    if not text or len(text) < n:
        print("text is too short, cannot analyze.")
        return False

    # 1. Generate N-grams (character-level sliding window).
    # List comprehension: slice from index i to i+n.
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]

    total_count = len(ngrams)
    if total_count == 0:
        return False

    # 2. Count frequencies.
    ngram_counts = Counter(ngrams)
    unique_count = len(ngram_counts)

    # 3. Compute Distinct-N (unique count / total count).
    # Repetitive text is typically < 0.4.
    distinct_ratio = unique_count / total_count

    # 4. Determine repetition.
    is_repetitive = distinct_ratio < threshold

    return is_repetitive


def count_words(text: str) -> int:
    # Match segments that look like English words.
    # Rule: starts and ends with a letter, may contain letters, apostrophes, or hyphens.
    pattern = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
    words = pattern.findall(text)
    return len(words)


# Safety cap on how many freshly produced images we forward back to the model
# per tool call. Visual tools currently emit at most one new image per call, but
# the cap protects context length and vision-encoder memory in case future tools
# emit several at once.
_MAX_ATTACHED_IMAGES_PER_TOOL = 4

# Best-effort timeout for downloading tool-produced images hosted on COS / web.
_IMAGE_DOWNLOAD_TIMEOUT = 30
_IMAGE_ATTACH_SEM: asyncio.Semaphore | None = None


def _get_image_attach_sem() -> asyncio.Semaphore:
    global _IMAGE_ATTACH_SEM
    if _IMAGE_ATTACH_SEM is None:
        workers = max(1, int(os.environ.get("IMAGE_ATTACH_MAX_CONCURRENCY", "16")))
        _IMAGE_ATTACH_SEM = asyncio.Semaphore(workers)
    return _IMAGE_ATTACH_SEM


def _tool_call_timeout(tool_name: str) -> float:
    env_name = f"TOOL_CALL_TIMEOUT_{tool_name.upper()}"
    raw = os.environ.get(env_name, os.environ.get("TOOL_CALL_TIMEOUT", "300"))
    try:
        return max(0.001, float(raw))
    except (TypeError, ValueError):
        return 300.0


def _load_pil_from_payload(payload: Any) -> Optional[Image.Image]:
    """Best-effort conversion of a tool-produced image payload into PIL.Image.

    Visual tools store newly produced images in the shared ``image_paths`` dict
    using one of several representations (see ``visual_tools._resolve_image_ref``
    for the full taxonomy):

    * ``http(s)://`` URL  (the default — uploaded to COS)
    * an absolute local path (fallback when COS upload is disabled)
    * raw ``bytes`` containing encoded image data
    * a ``PIL.Image.Image`` instance
    * a HuggingFace-style ``dict`` with ``bytes`` / ``path`` / ``url`` keys

    Returns ``None`` if the payload cannot be decoded; callers should silently
    skip such entries so a single corrupted image never aborts the rollout.
    """
    if isinstance(payload, Image.Image):
        return payload

    if isinstance(payload, bytes):
        try:
            return Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception:  # noqa: BLE001
            return None

    if isinstance(payload, str):
        if payload.startswith(("http://", "https://")):
            try:
                resp = requests.get(payload, timeout=_IMAGE_DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        if payload.startswith("data:image/"):
            try:
                import base64

                _, encoded = payload.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        if os.path.exists(payload):
            try:
                return Image.open(payload).convert("RGB")
            except Exception:  # noqa: BLE001
                return None

    if isinstance(payload, dict):
        for key in ("bytes", "path", "url", "data"):
            val = payload.get(key)
            if val is None:
                continue
            pil = _load_pil_from_payload(val)
            if pil is not None:
                return pil

    return None


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using rLLM's OpenAI engine for model inference.
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        default_max_tries: int = 3,
        executor=None,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            rollout_engine: rLLM OpenAI engine for model inference
            tools: Dictionary of available tools {tool_name: tool_instance}
            system_prompt: Optional custom system prompt
        """
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt
        self.executor = executor
        # Configuration from original DeepResearch
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.default_max_tries = default_max_tries

        # Smart context management using actual API consumption
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Per-turn token counts of the most recent call_server response, stamped
        # onto each assistant message so dumped Steps carry their own token length.
        self._last_prompt_length = None
        self._last_completion_length = None

        self.max_prompt_tokens = MAX_PROMPT_LENGTH_PER_RUN
        self.max_response_tokens = MAX_RESPONSE_LENGTH_PER_RUN

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<think>" in content and "</think>" in content

    async def call_server(
        self, messages: list[dict], max_tries: Optional[int] = None, **kwargs
    ):
        """Call rollout engine once; assumes XML ReAct format."""
        try:
            # Force per-round limits from DeepResearchAgent without local token estimation.
            if hasattr(self.rollout_engine, "max_prompt_length"):
                self.rollout_engine.max_prompt_length = int(self.max_prompt_tokens)
            if hasattr(self.rollout_engine, "max_response_length"):
                self.rollout_engine.max_response_length = int(self.max_response_tokens)

            kwargs.pop("max_new_tokens", None)
            kwargs["max_tokens"] = int(self.max_response_tokens)
            # DeepResearch converts the final chat transcript back into the
            # training sequence and deliberately drops every per-turn
            # ModelOutput. Defer unused HF pixel tensors and expanded prompt
            # ids. SGLang receives the raw image/video inputs and performs the
            # generation-side multimodal preprocessing.
            if getattr(
                self.rollout_engine,
                "supports_mm_training_input_deferral",
                False,
            ):
                kwargs["materialize_mm_training_inputs"] = False
            response = await self.rollout_engine.get_model_response(
                messages=[{k:v for k,v in message.items() if k != "tool_statistics"} for message in messages], **kwargs
            )

            if hasattr(response, "prompt_length") and hasattr(
                response, "completion_length"
            ):
                self.total_prompt_tokens += response.prompt_length
                self.total_completion_tokens += response.completion_length

            return response
        except Exception as exc:  # noqa: BLE001
            print(f"call_server failed: {exc}")
            raise

    def record_token_usage(self, response) -> None:
        """Record the latest prompt/completion token count from rollout engine."""
        prompt_tokens = getattr(response, "prompt_length", None)
        completion_tokens = getattr(response, "completion_length", None)

        # Per-turn token counts (this LLM call only). prompt_length INCLUDES the
        # expanded video/image tokens that sglang re-tokenised, so this is the
        # ground-truth token count of the actual model input for this turn.
        # Stored on the assistant message below so each dumped Step carries its
        # own prompt/completion token length (auditable, comparable to inference
        # promptTokenCount). total_* keep the "latest turn" semantics they had.
        self._last_prompt_length = None
        self._last_completion_length = None

        if prompt_tokens is not None:
            try:
                self.total_prompt_tokens = int(prompt_tokens)
                self._last_prompt_length = int(prompt_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_prompt_tokens = 0

        if completion_tokens is not None:
            try:
                self.total_completion_tokens = int(completion_tokens)
                self._last_completion_length = int(completion_tokens)
            except (TypeError, ValueError):  # noqa: PERF203
                self.total_completion_tokens = 0

    def get_total_tokens_used(self) -> int:
        """Return the latest prompt + completion token usage reported by the engine."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def _estimate_prompt_tokens(self, messages: list[dict]) -> int:
        """Estimate prompt length for the next call using the rollout engine's tokenizer."""
        tokenizer = getattr(self.rollout_engine, "tokenizer", None)
        chat_parser = getattr(self.rollout_engine, "chat_parser", None)

        if tokenizer is None or chat_parser is None:
            return self.total_prompt_tokens

        try:
            prompt = chat_parser.parse(
                messages,
                add_generation_prompt=True,
                is_first_msg=True,
                tools=[],
                accumulate_reasoning=getattr(
                    self.rollout_engine, "accumulate_reasoning", False
                ),
            )
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            return len(token_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"[TokenEstimator] Failed to estimate prompt tokens: {exc}")
            return self.total_prompt_tokens

    def _build_result(
        self,
        *,
        question: str,
        answer: str | None,
        messages: list[dict],
        prediction: str,
        termination: str,
        rounds: int,
        start_time: float,
        # next_prompt_tokens: int | None = None,
    ) -> dict:
        """Assemble result payload with token usage metadata."""
        token_usage = {
            "prompt": self.total_prompt_tokens,
            "completion": self.total_completion_tokens,
            "max_prompt": self.max_prompt_tokens,
        }

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": time.time() - start_time,
            "token_usage": token_usage,
        }
        return result

    async def _run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        video_grid: dict = None,
        modality: str = "video",
        videos: list = None,
        **kwargs,
    ) -> dict:
        """
        Main reasoning loop adapted from original DeepResearch.

        Supports image-processing tools via an internal ``image_paths``
        dictionary that maps ``img_1``, ``img_2``, … to local paths / URLs.

        For Video Deep Research, a ``video_grid`` (frame-grid handle dict, see
        ``tools.video_io``) is registered as ``video_1`` so the agent can call
        ``select_timespan`` / ``select_frame`` against it. Bootstrap frames for
        the model to see should be passed via ``images``.

        ``modality`` controls which system prompt is selected and which tool
        schemas are injected: ``"video"`` (default), ``"multi_image"``,
        or ``"single_image"``.
        """
        start_time = time.time()

        # The SFT select_timespan observation always contains a real clip. Keep
        # explicit diagnostic overrides possible, but make the aligned path the
        # default without modifying the concrete tool implementation.
        if modality == "video":
            os.environ.setdefault("SELECT_TIMESPAN_COMPOSE_SUBCLIP", "1")

        # ---- strip the dataset handle-hint prefix from the user question ----
        # Prepared RL data prefixes questions with "video_id: 1 \n Question: "
        # (or "image_id: 1 \n Question: ") to hint which grid/image handle is
        # seeded. onesearch_vl inference STRIPS this prefix before showing the
        # question to the model (pipeline.py: r"^\s*(image|video)_id:\s*\S+\s+Question:\s*"),
        # feeding only the bare question. Keeping the prefix in RL rollout is a
        # real train-inference token mismatch, so strip it here to match. The
        # handle (video_1 / img_1) is already registered in image_paths and the
        # grid mechanism is explained in the system prompt, so no hint is lost.
        question = re.sub(
            r"^\s*(?:image|video)_id:\s*\S+\s+Question:\s*",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip()

        # ---- active tool set for this rollout ----
        # Dispatch only tools advertised by the shared modality prompt.
        # This mirrors onesearch_vl's per-pipeline tool filtering so both the
        # <tools> schema in the system prompt AND the callable dispatch are
        # restricted to the same set the model was trained on.
        self._active_tools: dict = {
            name: tool
            for name, tool in self.tools.items()
            if _tool_is_active(name, modality)
        }

        if self.system_prompt is not None:
            # Caller supplied a fully-formed prompt; honour it as-is.
            system_prompt = self.system_prompt
        else:
            system_prompt = _select_system_prompt(self._active_tools, modality)

        # ---- image_paths management for visual tools ----
        self._image_paths: dict[str, str] = {}
        self._intermediate_dir = kwargs.pop(
            "intermediate_dir",
            os.path.join(INTERMEDIATE_ROOT, str(int(time.time() * 1000))),
        )
        if image_path:
            self._image_paths["img_1"] = image_path

        # ---- video grid handle for select_timespan / select_frame ----
        # The grid is a dict value (not a path/URL string), keyed ``video_1``.
        # The video tools detect grid handles by ``isinstance(entry, dict)``.
        if video_grid:
            self._image_paths["video_1"] = video_grid

        # ---- register bootstrap frames as img_1..img_N in image_paths ----
        # The model sees bootstrap frames in the first user message and naturally
        # tries to crop/search img_1, img_2, … directly. Without registration
        # the crop tool reports "img_1 not found" even though the model can see
        # them — the model has to call select_frame first (an extra wasted turn).
        # Registering them up-front lets the model operate on them immediately,
        # matching the onesearch_vl inference behaviour where bootstrapped frames
        # are part of the initial image_paths_dict.
        if images and not image_path:
            for i, pil_img in enumerate(images, start=1):
                self._image_paths[f"img_{i}"] = pil_img

        if images:
            user_message = {
                "role": "user",
                "content": question,
                "images": images,
                "ordered_content": [
                    *({"type": "image"} for _ in images),
                    {"type": "text", "text": question},
                ],
            }
        elif videos:
            # Pass video directly (as a list of video dicts with url/fps/max_pixels),
            # aligning rollout with SFT/infer which use video_url rather than static frames.
            user_message = {
                "role": "user",
                "content": question,
                "videos": videos,
                "ordered_content": [
                    *({"type": "video"} for _ in videos),
                    {"type": "text", "text": question},
                ],
            }
        else:
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]

        if os.environ.get("RLLM_USE_INFER_MAX_LLM_CALLS", "0") == "1":
            num_llm_calls_available = self.max_llm_calls
        else:
            num_llm_calls_available = min(
                self.max_llm_calls, _SFT_MAX_LLM_CALLS.get(modality, 12)
            )
        rollout_trace_id = kwargs.get("application_id") or uuid.uuid4().hex
        round = 0
        termination = None
        prediction = ""
        consecutive_bad_steps = 0

        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1
            # Get model response from rollout engine
            _llm_t0 = time.perf_counter()
            try:
                response = await self.call_server(messages, **kwargs)
            except Exception as exc:  # noqa: BLE001
                prediction = "call_server failed"
                termination = "error"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )
            _llm_duration_s = float(format(time.perf_counter() - _llm_t0, ".4f"))

            self.record_token_usage(response)
            engine_timing = getattr(response, "timing", None) or {}
            assistant_timing = {
                f"engine_{key}": float(value)
                for key, value in engine_timing.items()
                if isinstance(value, (int, float))
            }

            content = (
                response.text if hasattr(response, "text") and response.text else ""
            )

            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            if "<tool_call>" in content and "</tool_call>" in content:
                assistant_message = {
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                    "prompt_length": self._last_prompt_length,
                    "completion_length": self._last_completion_length,
                    "llm_duration_s": _llm_duration_s,
                    **assistant_timing,
                }
                messages.append(assistant_message)
                tool_error = False

                # Snapshot image_paths before tool execution so we can detect
                # any *newly produced* images (e.g. from crop / sharpen /
                # perspective_correct / super_resolution) and feed them back
                # to the multi-modal model in the next turn. This mirrors the
                # behaviour of the inference pipeline in
                # ``onesearch_vl/onesearch_infer/pipeline.py`` (see the
                # ``new_images`` branch that appends ``image_url`` /
                # ``inline_data`` parts to ``gemini_contents``).
                prev_image_keys = set(self._image_paths.keys())

                _tool_t0 = time.perf_counter()
                _tool_phase_timing: dict[str, float] = {}
                _tool_statistics = CallStatistics(rollout_id=rollout_trace_id, step_index=round)
                # Keep malformed tool calls observable without letting the
                # timing instrumentation itself crash the trajectory.  The
                # JSON branch assigns the parsed name below, but parse errors
                # still flow through the normal error observation path.
                _tool_name_for_timing = "unknown"

                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[
                    0
                ]
                try:
                    tool_call = _parse_tool_call_json(tool_call_text)
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    _tool_name_for_timing = tool_name or "unknown"
                    with observe_call(_tool_statistics):
                        result = await self.custom_call_tool(
                            tool_name,
                            tool_args,
                            _tool_timing=_tool_phase_timing,
                        )
                except Exception:
                    result = "[Json Parse Error]: Tool call is not a valid JSON."
                    tool_error = True

                if isinstance(result, str) and result.startswith(
                    (
                        "Error:",
                        "Tool execution error:",
                        "Search error",
                        "image_search error",
                    )
                ):
                    tool_error = True
                if tool_error:
                    assistant_message["step_error"] = True

                # Wrap result in <observation> then <tool_response>, matching
                # onesearch_vl pipeline.py:543 + sft_chat_format.observation_user_message
                # The default tool-response token structure is:
                #   <tool_response>\n<observation>\n{tool_output}\n</observation>\n</tool_response>
                observation_text = f"<observation>\n{result}\n</observation>"
                tool_response = f"<tool_response>\n{observation_text}\n</tool_response>"
                tool_response_msg: dict[str, Any] = {
                    "role": "user",
                    "content": tool_response,
                    "tool_name": _tool_name_for_timing,
                }

                # Identify image_paths entries produced by this tool call and
                # attach them as PIL images on the tool-response message so the
                # vision encoder can actually see them on the next turn.
                if not tool_error:
                    new_keys = [k for k in self._image_paths if k not in prev_image_keys]
                    new_image_keys = [
                        key
                        for key in new_keys
                        if key.startswith("img_")
                        and not isinstance(self._image_paths[key], dict)
                    ]
                    new_clip_keys = [
                        key
                        for key in new_keys
                        if key.startswith("clip_")
                        and isinstance(self._image_paths[key], dict)
                    ]
                    attach_timeout = _tool_call_timeout("image_attach")

                    async def _load_attached_image(key: str) -> Optional[Image.Image]:
                        phase_timing: dict[str, float] = {}
                        try:
                            image = await run_blocking_bounded(
                                lambda: _load_pil_from_payload(self._image_paths[key]),
                                executor=self.executor,
                                semaphore=_get_image_attach_sem(),
                                timeout=attach_timeout,
                                queue_timeout=min(15.0, attach_timeout),
                                label="image attachment",
                                timing=phase_timing,
                            )
                        except ToolCallTimeout:
                            return None
                        for name, seconds in phase_timing.items():
                            metric = f"image_attach_{name}"
                            _tool_phase_timing[metric] = (
                                _tool_phase_timing.get(metric, 0.0) + seconds
                            )
                        return image

                    loaded = await asyncio.gather(
                        *(
                            _load_attached_image(key)
                            for key in new_image_keys[:_MAX_ATTACHED_IMAGES_PER_TOOL]
                        )
                    )
                    new_pil_images = [image for image in loaded if image is not None]
                    if new_pil_images:
                        tool_response_msg["images"] = new_pil_images

                    clip_videos = []
                    for key in new_clip_keys:
                        clip = self._image_paths[key]
                        clip_url = clip.get("video_url")
                        # A composed clip is re-indexed at offset zero. An
                        # inherited parent URL would show the wrong temporal
                        # span and is therefore an explicit step error.
                        if (
                            not clip_url
                            or not clip.get("subclip_path")
                            or int(clip.get("url_offset", -1)) != 0
                        ):
                            tool_error = True
                            assistant_message["step_error"] = True
                            result = (
                                "Tool execution error:\nselect_timespan did not "
                                "produce a model-visible re-indexed clip."
                            )
                            observation_text = f"<observation>\n{result}\n</observation>"
                            tool_response = (
                                f"<tool_response>\n{observation_text}\n</tool_response>"
                            )
                            tool_response_msg["content"] = tool_response
                            self._image_paths.pop(key, None)
                            continue
                        clip_videos.append(
                            {
                                "video": clip_url,
                                "fps": 2.0,
                                "min_pixels": 256,
                                "max_pixels": 100352,
                                "max_frames": 128,
                                "n_frames": int(clip.get("n_frames") or 0),
                            }
                        )
                    if clip_videos:
                        tool_response_msg["videos"] = clip_videos

                media_parts = []
                media_parts.extend(
                    {"type": "image"} for _ in tool_response_msg.get("images", [])
                )
                media_parts.extend(
                    {"type": "video"} for _ in tool_response_msg.get("videos", [])
                )
                tool_response_msg["ordered_content"] = [
                    {"type": "text", "text": "<tool_response>\n"},
                    *media_parts,
                    {
                        "type": "text",
                        "text": f"{observation_text}\n</tool_response>",
                    },
                ]

                _tool_dur = float(format(time.perf_counter() - _tool_t0, ".4f"))
                tool_response_msg["tool_duration_s"] = _tool_dur
                tool_response_msg["tool_statistics"] = _tool_statistics.finish(
                    _tool_name_for_timing, result, assistant_message["step_error"], _tool_phase_timing,
                )
                if _tool_phase_timing:
                    tool_response_msg["tool_phase_timing"] = dict(
                        _tool_phase_timing
                    )
                if os.environ.get("RL_TOOL_TIMING_DEBUG", "").lower() in ("1", "true"):
                    import sys as _sys

                    print(
                        f"[ToolTiming] {_tool_name_for_timing} {_tool_dur:.2f}s "
                        f"{json.dumps(_tool_phase_timing, sort_keys=True)}",
                        flush=True,
                        file=_sys.stderr,
                    )

                messages.append(tool_response_msg)
                if assistant_message["step_error"]:
                    consecutive_bad_steps += 1
                else:
                    consecutive_bad_steps = 0
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

            elif "<response>" in content and "</response>" in content:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": False,
                        "prompt_length": self._last_prompt_length,
                        "completion_length": self._last_completion_length,
                        "llm_duration_s": _llm_duration_s,
                        **assistant_timing,
                    }
                )
                prediction = content.split("<response>")[1].split("</response>")[0].strip()
                termination = "answer"
                consecutive_bad_steps = 0
                break

            else:
                is_repetitive = analyze_repetition_ngram(content)
                is_overlong = count_words(content) > 2500
                if is_repetitive and is_overlong:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.strip(),
                            "step_error": True,
                            "prompt_length": self._last_prompt_length,
                            "completion_length": self._last_completion_length,
                            "llm_duration_s": _llm_duration_s,
                            **assistant_timing,
                        }
                    )
                    prediction = "Repetition response"
                    termination = "repetition_detected"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

                observation = "Error: Invalid content format. Content must contain <tool_call> or <response> tags. Let's try again."
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": True,
                        "prompt_length": self._last_prompt_length,
                        "completion_length": self._last_completion_length,
                        "llm_duration_s": _llm_duration_s,
                        **assistant_timing,
                    }
                )
                messages.append({"role": "user", "content": observation})
                consecutive_bad_steps += 1
                if consecutive_bad_steps >= 3:
                    prediction = "Too many consecutive step errors."
                    termination = "consecutive_step_errors"
                    return self._build_result(
                        question=question,
                        answer=answer,
                        messages=messages,
                        prediction=prediction,
                        termination=termination,
                        rounds=round,
                        start_time=start_time,
                    )

            if num_llm_calls_available <= 0 and "<answer>" not in content and "<response>" not in content:
                prediction = f"No answer found after {round} rounds."
                termination = f"answer not found after {round} rounds"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

        last_message_content = (
            messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        )
        if last_message_content and "<response>" in last_message_content:
            prediction = last_message_content.split("<response>")[1].split("</response>")[0].strip()
            termination = "answer"
        elif last_message_content and "<answer>" in last_message_content:
            prediction = last_message_content.split("<answer>")[1].split("</answer>")[0].strip()
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"

        result = self._build_result(
            question=question,
            answer=answer,
            messages=messages,
            prediction=prediction,
            termination=termination,
            rounds=round,
            start_time=start_time,
        )

        print("\n DeepResearch completed:")
        print(f"   Rounds: {round}")
        print(f"   Time: {result['time_taken']:.1f}s")
        print(f"   Termination: {termination}")
        print(
            "   Token usage: prompt={prompt}, completion={completion}, max_prompt={max_prompt}".format(
                prompt=self.total_prompt_tokens,
                completion=self.total_completion_tokens,
                max_prompt=self.max_prompt_tokens,
            )
        )
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """
        Execute tool calls with the available tools.

        Visual / video tools (select_timespan, select_frame, crop,
        layout_parsing, image_search, perspective_correct, super_resolution,
        sharpen) receive extra context via **ctx so they can manage the shared
        ``image_paths`` dictionary (which also holds video/clip grid handles).
        """
        VISUAL_TOOLS = {
            "select_timespan", "select_frame",
            "crop", "layout_parsing", "text_search", "image_search",
            "perspective_correct", "super_resolution", "sharpen",
        }
        # Keys the agent injects as context; a model-supplied arg with the same
        # name would collide on **tool_args/**ctx unpack and raise an opaque
        # TypeError ("got multiple values for keyword argument 'image_paths'").
        RESERVED_CTX_KEYS = {"image_paths", "intermediate_dir", "_tool_timing"}

        # Use per-run active tools (modality-filtered); fall back to self.tools
        # if _run has not yet set _active_tools (e.g. direct call in tests).
        active = getattr(self, "_active_tools", self.tools)

        if tool_name in active:
            try:
                # The model emits `arguments` as free-form JSON; it may not be a
                # dict (e.g. a list) and may collide with reserved ctx keys.
                # Sanitize before unpacking so a malformed call returns an
                # actionable message instead of a bare TypeError.
                if not isinstance(tool_args, dict):
                    return (
                        f"Tool {tool_name} arguments must be a JSON object, got "
                        f"{type(tool_args).__name__}. Example: "
                        f'{{"name": "{tool_name}", "arguments": {{...}}}}'
                    )
                clean_args = {
                    k: v for k, v in tool_args.items() if k not in RESERVED_CTX_KEYS
                }
                dropped = set(tool_args) & RESERVED_CTX_KEYS
                if dropped:
                    print(
                        f"[agent] dropped reserved arg(s) {sorted(dropped)} from "
                        f"model call to {tool_name}"
                    )

                ctx = {}
                if tool_name in VISUAL_TOOLS:
                    ctx["image_paths"] = getattr(self, "_image_paths", {})
                    ctx["intermediate_dir"] = getattr(
                        self, "_intermediate_dir", INTERMEDIATE_ROOT
                    )
                tool_timing = kwargs.get("_tool_timing")
                if isinstance(tool_timing, dict):
                    ctx["_tool_timing"] = tool_timing

                timeout = _tool_call_timeout(tool_name)
                if hasattr(active[tool_name], "call"):
                    if asyncio.iscoroutinefunction(active[tool_name].call):
                        call = active[tool_name].call(**clean_args, **ctx)
                        try:
                            result = await asyncio.wait_for(
                                call,
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            return (
                                "Tool execution error:\n"
                                f"{tool_name} exceeded its total "
                                f"{timeout:.0f}s deadline."
                            )
                    else:
                        result = await run_blocking_bounded(
                            lambda: active[tool_name].call(
                                **clean_args,
                                **ctx,
                            ),
                            executor=self.executor,
                            timeout=timeout,
                            label=tool_name,
                        )
                elif callable(active[tool_name]):
                    result = await run_blocking_bounded(
                        lambda: active[tool_name](**clean_args, **ctx),
                        executor=self.executor,
                        timeout=timeout,
                        label=tool_name,
                    )
                else:
                    result = f"Tool {tool_name} is not callable"

                return str(result)

            except ToolCallTimeout:
                return (
                    "Tool execution error:\n"
                    f"{tool_name} exceeded its total "
                    f"{_tool_call_timeout(tool_name):.0f}s deadline."
                )
            except Exception as e:
                return f"Error calling tool {tool_name}: {e}"
        else:
            # Mirror onesearch_vl/onesearch_infer/tools.py: return a fixed
            # message without leaking the tool inventory into the context.
            # Use "Tool execution error:" prefix so _has_tool_error_observation
            # marks this step as an error (same as other tool failures).
            if tool_name in _VIDEO_ONLY_TOOL_NAMES:
                return (
                    f"Tool execution error:\n"
                    f"{tool_name} is not available in this modality. "
                    f"Operate directly on the provided img_* handles."
                )
            return f"Tool execution error:\nUnknown tool: {tool_name!r}."

    def reset(self):
        """Reset the agent state (for compatibility with rLLM workflow)."""
        # Reset token counters for each new task
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        video_grid: dict = None,
        modality: str = "video",
        videos: list = None,
        **kwargs,
    ) -> dict:
        """
        Public interface for running the agent.

        Args:
            question: Research question to answer
            answer: Ground truth answer (optional, for evaluation)
            video_grid: Optional frame-grid handle registered as ``video_1``
                for select_timespan / select_frame.
            modality: ``"video"`` (default), ``"multi_image"``, or
                ``"single_image"``. Controls the system prompt and which
                tool schemas are injected into the system slot.

        Returns:
            Result dictionary
        """
        # Reset token counters for each new run
        self.reset()
        return await self._run(
            question, answer, images, image_path,
            video_grid=video_grid, modality=modality, videos=videos, **kwargs,
        )


DeepResearchAgent = MultiTurnReactAgent
