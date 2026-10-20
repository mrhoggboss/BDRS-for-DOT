"""Read benchmark output back: the run rows the orchestrator appends to
``runs/*.jsonl``."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, Union

from src.protocol.regimes import canonical_algorithm

PathLike = Union[str, Path]


def iter_rows(paths: Iterable[PathLike]) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    """Yield ``(file, row)`` for every row, with the algorithm name canonicalised
    and the status read by ``read_status`` (the written label kept as
    ``status_as_written`` when it differs). A line that is not a JSON object,
    such as one truncated by a run killed while appending, is skipped with a
    warning."""
    for p in map(Path, paths):
        with p.open() as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = None
                if not isinstance(row, dict):
                    print(f"warning: {p}:{lineno} is not a JSON row; skipped", file=sys.stderr)
                    continue
                row["algorithm"] = canonical_algorithm(row.get("algorithm") or "")
                status = read_status(row)
                if status != row.get("status"):
                    row["status_as_written"] = row.get("status")
                    row["status"] = status
                yield p, row


def read_status(row: Dict[str, Any]) -> Optional[str]:
    """The row's status, with an ``ok`` row whose ``timeout_flag`` is set read as
    ``timeout``: cuPDLPx writes such a row when it stops at its own time limit."""
    status = row.get("status")
    flag = row.get("timeout_flag") or (row.get("algo_extras") or {}).get("timeout_flag")
    return "timeout" if status == "ok" and flag else status
