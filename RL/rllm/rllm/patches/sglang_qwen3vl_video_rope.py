"""Runtime patch: make sglang 0.5.3's MRotaryEmbedding.get_rope_index handle
Qwen3-VL video the way HF transformers does.

Qwen3-VL separates a T-frame video into T timestamp-delimited vision blocks
(``<t1><vision_start>frame1<vision_end> ... <tT><vision_start>frameT<vision_end>``),
but ``video_grid_thw`` carries a SINGLE row ``[[T, H, W]]``. sglang 0.5.3 folds
qwen3_vl into the qwen2_vl get_rope_index path, which walks ``video_index`` once
per vision_start block (T times) against a size-1 ``video_grid_thw`` and raises::

    IndexError: index 1 is out of bounds for dimension 0 with size 1

HF's qwen3_vl get_rope_index fixes this up front (modeling_qwen3_vl.py)::

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

This module wraps the sglang staticmethod to apply the same split — but ONLY for
qwen3_vl / qwen3_vl_moe, and on a fresh tensor so the caller's original
``video_grid_thw`` (consumed by the vision encoder's cu_seqlens) is untouched.
"""

_QWEN3_MODEL_TYPES = ("qwen3_vl", "qwen3_vl_moe")

_patched = False


def apply() -> bool:
    """Idempotently wrap sglang's get_rope_index. Returns True if applied."""
    global _patched
    if _patched:
        return True
    try:
        import torch
        from sglang.srt.layers import rotary_embedding as _re
    except Exception:
        return False

    MRotary = getattr(_re, "MRotaryEmbedding", None)
    if MRotary is None or not hasattr(MRotary, "get_rope_index"):
        return False

    # Accessing the attribute yields the plain underlying function (staticmethod
    # already unwrapped on attribute access in py3.10). Guard the __func__ case
    # for older wrapping just in case.
    _orig = MRotary.get_rope_index
    _orig = getattr(_orig, "__func__", _orig)
    if getattr(_orig, "_qwen3vl_video_patched", False):
        _patched = True
        return True

    def _split_qwen3_video_grid(video_grid_thw):
        # Fresh tensor: repeat each row T times, then set the temporal dim to 1.
        # Never mutate the caller's tensor (the vision encoder reads the same one).
        vgt = video_grid_thw.clone()
        vgt = torch.repeat_interleave(vgt, vgt[:, 0], dim=0)
        vgt[:, 0] = 1
        return vgt

    def get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        model_type,
        tokens_per_second=None,
        input_ids=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        if model_type in _QWEN3_MODEL_TYPES and video_grid_thw is not None:
            video_grid_thw = _split_qwen3_video_grid(video_grid_thw)
            # qwen3_vl uses timestamp tokens, not second_per_grid_ts; HF passes
            # None. If some caller ever supplies it, keep indices aligned.
            if second_per_grid_ts is not None:
                try:
                    import torch as _t
                    if hasattr(second_per_grid_ts, "shape") and second_per_grid_ts.shape[0] != video_grid_thw.shape[0]:
                        second_per_grid_ts = None
                except Exception:
                    second_per_grid_ts = None
        return _orig(
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            model_type,
            tokens_per_second=tokens_per_second,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

    get_rope_index._qwen3vl_video_patched = True
    MRotary.get_rope_index = staticmethod(get_rope_index)
    _patched = True
    return True
