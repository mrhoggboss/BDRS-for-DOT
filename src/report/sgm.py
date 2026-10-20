"""Summary statistics of the report layer.

Solve times are summarised by SGM10, the shifted geometric mean
``exp(mean(log(t_i + 10))) - 10`` in seconds: unlike the plain geometric mean it
is defined at t = 0, and the shift keeps sub-second differences from
dominating it. Every function skips None and returns NaN when nothing is left.
"""

import math
from typing import Iterable, Optional


def sgm(values: Iterable[Optional[float]], shift: float = 10.0) -> float:
    """``exp(mean(log(x_i + shift))) - shift``; a NaN in the input propagates."""
    xs = [v + shift for v in values if v is not None]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) - shift


def geomean(values: Iterable[Optional[float]]) -> float:
    """Geometric mean of the positive finite values (zeros are dropped)."""
    xs = [v for v in values if v is not None and v > 0.0 and math.isfinite(v)]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def median(values: Iterable[Optional[float]]) -> float:
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def iqr(values: Iterable[Optional[float]]) -> float:
    """p75 − p25, the quartiles taken at sorted indices without interpolation."""
    xs = sorted(v for v in values if v is not None and math.isfinite(v))
    n = len(xs)
    if n == 0:
        return float("nan")
    p25 = xs[max(0, (n - 1) // 4)]
    p75 = xs[min(n - 1, (3 * (n - 1)) // 4)]
    return p75 - p25
