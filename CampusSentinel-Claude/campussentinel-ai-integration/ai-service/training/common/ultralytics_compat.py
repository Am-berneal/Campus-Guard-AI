"""Version-tolerant Ultralytics prediction arguments.

Ultralytics 8.4.x replaced the `half=True` prediction argument with `quantize=16`
(`half` became deprecated); older 8.x releases only know `half`. Passing the wrong one
fails or warns, so pick whichever the INSTALLED version supports.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _supports_quantize() -> bool:
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except Exception:  # very old/unusual installs: fall back to the classic argument
        return False
    return "quantize" in DEFAULT_CFG_DICT


def precision_kwargs(half: bool) -> dict:
    """{'quantize': 16|32} on new Ultralytics, {'half': bool} on older versions."""
    if _supports_quantize():
        return {"quantize": 16 if half else 32}
    return {"half": bool(half)}
