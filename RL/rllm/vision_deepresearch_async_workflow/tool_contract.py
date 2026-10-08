"""Validate tool dispatch against the system prompt definitions."""
from __future__ import annotations

import json
import os
import re


def allowed_names(known):
    value = os.getenv('RLLM_ALLOWED_TOOLS', '').strip()
    if not value:
        return None
    names = {name.strip() for name in value.split(',') if name.strip()}
    if not names or not names <= set(known):
        raise ValueError('RLLM_ALLOWED_TOOLS contains empty or unknown tool names')
    return names


def schema_names(prompt):
    names = []
    for block in re.findall(r'<tools>(.*?)</tools>', prompt, re.S):
        for line in block.splitlines():
            if line.strip():
                spec = json.loads(line)
                names.append(spec.get('function', spec)['name'])
    return names


def validate_sft_tools(tools):
    """Compare every modality with the SFT inventory before any rollouts.

    Imports are delayed to avoid a cycle with the registry. A configuration
    conflict is a startup error, not a penalized trajectory or prompt edit.
    """
    from vision_deepresearch_async_workflow.deepresearch_agent import _select_system_prompt, _tool_is_active
    from vision_deepresearch_async_workflow.deepresearch_tools_async_executor import DEEPRESEARCH_TOOLS

    for modality in ('single_image', 'multi_image', 'video'):
        prompt = _select_system_prompt(DEEPRESEARCH_TOOLS, modality)
        expected = schema_names(prompt)
        actual = {name for name in tools if _tool_is_active(name, modality)}
        if len(expected) != len(set(expected)) or set(expected) != actual:
            raise ValueError(
                f'SFT tool inventory mismatch for {modality}: '
                f'missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}. '
                'Restore RLLM_ALLOWED_TOOLS to the prompt tool inventory.'
            )
        for name in actual:
            target = getattr(tools[name], 'call', tools[name])
            if not callable(target):
                raise ValueError(f'SFT tool {name} is not callable')
        if _select_system_prompt(tools, modality) != prompt:
            raise ValueError(f'Runtime tool schema differs from SFT prompt for {modality}')
