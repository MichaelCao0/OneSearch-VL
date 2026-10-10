"""Fail closed when ndarray preprocessing would erase a material stream origin.

The pinned Qwen multimedia helper returns zero-based arrays and does not carry
container PTS offsets to the processor. This gate does not synchronize, pad,
repair or adjudicate the media. A 2 ms allowance covers Matroska timestamp
rounding only; fine AV timing capability is not established by this gate.
"""
from __future__ import annotations
import math

CLOCK_ROUNDING_ALLOWANCE_S = 0.002

class ClockAlignmentUnsupportedError(ValueError):
    """Native array preparation cannot represent the supplied stream origin."""


def check_native_clip_clocks(bound_input):
    streams = bound_input['mapping']['streams']
    if not streams or {s.get('codec_type') for s in streams} != {'audio','video'}:
        raise ClockAlignmentUnsupportedError('joint AV stream clocks required')
    for stream in streams:
        origin = stream.get('output_start_time_s')
        if isinstance(origin,bool) or not isinstance(origin,(int,float)):
            raise ClockAlignmentUnsupportedError('known clip stream origins required')
        try:
            valid = math.isfinite(origin) and abs(origin) <= CLOCK_ROUNDING_ALLOWANCE_S
        except OverflowError:
            valid = False
        if not valid:
            raise ClockAlignmentUnsupportedError('nonzero stream origins unsupported by zero-based native arrays')
