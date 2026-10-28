"""In-process Prometheus text metrics for request count, latency, and 5xx errors."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

_lock = threading.Lock()
_requests_total: dict[tuple[str, str], int] = defaultdict(int)
_errors_5xx = 0
# Cumulative observed latency in milliseconds, for a simple average and buckets.
_latency_sum_ms = 0.0
_latency_count = 0
_LATENCY_BUCKETS_MS = (50, 100, 250, 500, 1000, 2500, 5000, 10000)
_latency_buckets: dict[int, int] = {b: 0 for b in _LATENCY_BUCKETS_MS}
_latency_bucket_inf = 0


def observe_request(method: str, path: str, status: int, duration_ms: float) -> None:
    # Collapse IDs so cardinality stays bounded.
    route = _normalise_path(path)
    with _lock:
        global _errors_5xx, _latency_sum_ms, _latency_count, _latency_bucket_inf
        _requests_total[(method.upper(), route)] += 1
        if status >= 500:
            _errors_5xx += 1
        _latency_sum_ms += duration_ms
        _latency_count += 1
        placed = False
        for bound in _LATENCY_BUCKETS_MS:
            if duration_ms <= bound:
                _latency_buckets[bound] += 1
                placed = True
                break
        if not placed:
            _latency_bucket_inf += 1


def _normalise_path(path: str) -> str:
    parts = path.split("/")
    cleaned: list[str] = []
    for part in parts:
        if not part:
            cleaned.append(part)
            continue
        # UUID-like or long opaque ids become placeholders.
        if len(part) >= 32 and all(c in "0123456789abcdef-" for c in part.lower()):
            cleaned.append(":id")
        elif part.isdigit():
            cleaned.append(":id")
        else:
            cleaned.append(part)
    return "/".join(cleaned) or "/"


def render_prometheus() -> str:
    lines = [
        "# HELP dataforge_http_requests_total Total HTTP requests.",
        "# TYPE dataforge_http_requests_total counter",
    ]
    with _lock:
        for (method, route), count in sorted(_requests_total.items()):
            lines.append(
                f'dataforge_http_requests_total{{method="{method}",path="{route}"}} {count}'
            )
        lines.extend(
            [
                "# HELP dataforge_http_5xx_total HTTP responses with status >= 500.",
                "# TYPE dataforge_http_5xx_total counter",
                f"dataforge_http_5xx_total {_errors_5xx}",
                "# HELP dataforge_http_request_duration_ms_sum Cumulative request duration in milliseconds.",
                "# TYPE dataforge_http_request_duration_ms_sum counter",
                f"dataforge_http_request_duration_ms_sum {_latency_sum_ms:.3f}",
                "# HELP dataforge_http_request_duration_ms_count Request duration observations.",
                "# TYPE dataforge_http_request_duration_ms_count counter",
                f"dataforge_http_request_duration_ms_count {_latency_count}",
                "# HELP dataforge_http_request_duration_ms Request duration histogram (milliseconds).",
                "# TYPE dataforge_http_request_duration_ms histogram",
            ]
        )
        cumulative = 0
        for bound in _LATENCY_BUCKETS_MS:
            cumulative += _latency_buckets[bound]
            lines.append(f'dataforge_http_request_duration_ms_bucket{{le="{bound}"}} {cumulative}')
        cumulative += _latency_bucket_inf
        lines.append(f'dataforge_http_request_duration_ms_bucket{{le="+Inf"}} {cumulative}')
        lines.append(f"dataforge_http_request_duration_ms_sum {_latency_sum_ms:.3f}")
        lines.append(f"dataforge_http_request_duration_ms_count {_latency_count}")
    lines.append(f"# scraped_at {int(time.time())}")
    return "\n".join(lines) + "\n"
