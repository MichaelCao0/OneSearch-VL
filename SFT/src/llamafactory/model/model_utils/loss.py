# Copyright 2025 the LlamaFactory team.
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

from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)


def _cross_entropy_sum(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels, ignore_index=ignore_index, reduction="sum")


def chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    chunk_size: int = 1024,
    **kwargs,
) -> torch.Tensor:
    r"""Compute FP32 causal-LM cross entropy without retaining a full FP32 logits tensor.

    Transformers normally upcasts every logit to FP32 before cross entropy. For a
    30k-token Qwen3-VL sample with a 151,936-token vocabulary, that temporary is
    about 17.5 GiB. Checkpointing independent token chunks preserves the FP32 loss
    while only one chunk's upcast is live during forward or backward.
    """
    del kwargs
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = shift_labels.reshape(-1).to(flat_logits.device)
    cross_entropy_sum = partial(_cross_entropy_sum, ignore_index=ignore_index)
    loss_chunks = []
    for start in range(0, flat_logits.size(0), chunk_size):
        chunk_logits = flat_logits[start : start + chunk_size]
        chunk_labels = flat_labels[start : start + chunk_size]
        if torch.is_grad_enabled() and chunk_logits.requires_grad:
            chunk_loss = checkpoint(
                cross_entropy_sum,
                chunk_logits,
                chunk_labels,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_loss = cross_entropy_sum(chunk_logits, chunk_labels)

        loss_chunks.append(chunk_loss)

    loss = torch.stack(loss_chunks).sum()
    if num_items_in_batch is None:
        denominator = flat_labels.ne(ignore_index).sum()
    else:
        denominator = num_items_in_batch.to(loss.device)

    return loss / denominator.to(loss.dtype)


def patch_qwen3_vl_causal_lm_loss(model: "PreTrainedModel") -> None:
    if getattr(model.config, "model_type", None) != "qwen3_vl":
        return

    model.loss_function = chunked_causal_lm_loss
    logger.info_rank0("Using checkpointed FP32 causal-LM loss for Qwen3-VL to bound long-sequence logits memory.")
