from __future__ import annotations

try:
    from prophet import Prophet
except ImportError:  # pragma: no cover
    Prophet = None


def prophet_available() -> bool:
    return Prophet is not None

