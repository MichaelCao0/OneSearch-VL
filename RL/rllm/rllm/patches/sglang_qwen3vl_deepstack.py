"""Keep Qwen3-VL DeepStack embeddings aligned across missing modality embeddings."""

_patched = False


def apply() -> bool:
    """Idempotently replace sglang's embed_mm_inputs. Returns True if applied."""
    global _patched
    if _patched:
        return True
    try:
        import torch
        from sglang.srt.managers import mm_utils as _mm
    except Exception:
        return False

    _orig = getattr(_mm, "embed_mm_inputs", None)
    if _orig is None:
        return False
    if getattr(_orig, "_qwen3vl_deepstack_patched", False):
        _patched = True
        return True

    # Pull the module-level names embed_mm_inputs relies on so the replacement
    # closes over the same symbols (they live in mm_utils' globals).
    Modality = _mm.Modality
    get_embedding_and_mask = _mm.get_embedding_and_mask
    flatten_nested_list = _mm.flatten_nested_list

    def embed_mm_inputs(
        mm_inputs_list,
        extend_prefix_lens,
        extend_seq_lens,
        input_ids,
        input_embedding,
        multimodal_model=None,
        data_embedding_func_mapping=None,
        placeholder_tokens=None,
        use_deepstack=False,
        **kwargs,
    ):
        other_info = {}
        if mm_inputs_list is None:
            return None

        item_flatten_list = []
        for mm_inputs in mm_inputs_list:
            item_flatten_list += [
                item for item in mm_inputs.mm_items if item is not None
            ]

        embeddings, masks, deepstack_embeddings = [], [], []

        for modality in Modality.all():
            items = [
                item
                for item in item_flatten_list
                if item.is_modality(modality=modality)
            ]
            embedder = (
                None
                if data_embedding_func_mapping is None
                else data_embedding_func_mapping.get(modality, None)
            )
            if embedder is None:
                modality_id = modality.name.lower()
                embedder = getattr(
                    multimodal_model, f"get_{modality_id}_feature", None
                )
            if len(items) != 0 and embedder is not None:
                placeholder_tensor = torch.as_tensor(
                    [item.pad_value for item in items],
                    device=input_ids.device,
                )
                items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
                items_offsets = []
                for i, mm_inputs in enumerate(mm_inputs_list):
                    mm_items = [
                        item
                        for item in mm_inputs.mm_items
                        if item.is_modality(modality=modality)
                    ]
                    items_size[i + 1] = len(mm_items)
                    items_offsets.append(
                        flatten_nested_list([item.offsets for item in mm_items])
                    )
                items_size = torch.cumsum(items_size, dim=0).tolist()

                embedding, mask = get_embedding_and_mask(
                    data_embedding_func=embedder,
                    embedding_items=items,
                    placeholder_tensor=placeholder_tensor,
                    input_ids=input_ids,
                    items_size=items_size,
                    prefix_length=extend_prefix_lens,
                    extend_length=extend_seq_lens,
                    items_offset_list=items_offsets,
                )

                if use_deepstack and embedding is not None:
                    embedding, deepstack_embedding = (
                        multimodal_model.separate_deepstack_embeds(embedding)
                    )
                    deepstack_embeddings += [deepstack_embedding]
                elif use_deepstack:
                    # PATCH: keep deepstack_embeddings aligned with embeddings so
                    # the write-back loop can index it with the same i. Without
                    # this the None-embedding (prefix-cache-hit) modality leaves
                    # deepstack_embeddings shorter -> IndexError -> engine hang.
                    deepstack_embeddings += [None]
                embeddings += [embedding]
                masks += [mask]

        vocab_size = input_embedding.num_embeddings
        input_ids.clamp_(min=0, max=vocab_size - 1)
        inputs_embeds = input_embedding(input_ids)

        if use_deepstack:
            num_deepstack_embeddings = (
                len(multimodal_model.deepstack_visual_indexes) if use_deepstack else 0
            )
            deepstack_embedding_shape = inputs_embeds.shape[:-1] + (
                inputs_embeds.shape[-1] * num_deepstack_embeddings,
            )
            input_deepstack_embeds = torch.zeros(
                deepstack_embedding_shape,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            other_info["input_deepstack_embeds"] = input_deepstack_embeds

        for i, embedding, mask in zip(range(len(embeddings)), embeddings, masks):
            if embedding is None or mask is None:
                continue
            indices = torch.where(mask.squeeze(dim=-1))[0]
            inputs_embeds[indices] = embedding.to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            # PATCH: guard against a None placeholder (aligned above). If this
            # modality's deepstack slot is None the embedding above was also
            # None and we already skipped via `continue`; this extra check is
            # belt-and-suspenders in case only the deepstack half is missing.
            if use_deepstack and deepstack_embeddings[i] is not None:
                input_deepstack_embeds[indices] = deepstack_embeddings[i].to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

        return inputs_embeds, other_info

    embed_mm_inputs._qwen3vl_deepstack_patched = True
    _mm.embed_mm_inputs = embed_mm_inputs
    # general_mm_embed_routine calls embed_mm_inputs by module global, so the
    # rebind above is picked up. No need to touch the caller.
    _patched = True
    return True
