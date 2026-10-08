"""Startup guard: make faulthandler.enable() tolerate OSError [Errno 12]."""

import faulthandler as _faulthandler

_GUARD_MARKER = "_te_enomem_guard"

if not getattr(_faulthandler.enable, _GUARD_MARKER, False):
    _orig_enable = _faulthandler.enable

    def _guarded_enable(*args, **kwargs):
        try:
            return _orig_enable(*args, **kwargs)
        except OSError:
            # sigaltstack alloc failed under memory pressure; faulthandler is a
            # debug aid, not required for correctness — degrade gracefully.
            return None

    setattr(_guarded_enable, _GUARD_MARKER, True)
    _faulthandler.enable = _guarded_enable
