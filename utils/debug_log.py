from __future__ import annotations

import os
from typing import Iterable


DEFAULT_PREFIXES = ("[ASSOC]", "[TRK]", "[TRK-SMALL]")


def emit_debug_line(
    line: str,
    *,
    flush: bool = True,
    prefixes: Iterable[str] = DEFAULT_PREFIXES,
) -> None:
    print(line, flush=flush)
    log_path = os.environ.get("DEBUG_KEY_LOG", "").strip()
    if not log_path:
        return
    if prefixes and not any(line.startswith(prefix) for prefix in prefixes):
        return
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")
