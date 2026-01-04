# model_selector.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

# Support both package-relative and top-level imports.
try:
    from .logger import log  # type: ignore
except Exception:
    from logger import log  # type: ignore

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SelectorConfig:
    force_model: str = os.getenv("FORCE_MODEL", "").strip()
    fallback: str = os.getenv("FALLBACK_MODEL", "baseline")
    allow_untrained_baseline: bool = True


def _list_candidates() -> List[str]:
    names: List[str] = []
    try:
        for p in MODELS_DIR.glob("*.joblib"):
            if p.is_file():
                names.append(p.stem)
    except Exception:
        pass
    # Ensure baseline is present as a logical candidate
    if "baseline" not in names:
        names.append("baseline")
    # sort by mtime (newest first) if files exist
    try:
        names = sorted(names, key=lambda n: (
            -(MODELS_DIR / f"{n}.joblib").stat().st_mtime if (MODELS_DIR / f"{n}.joblib").exists() else 0))
    except Exception:
        names = list(dict.fromkeys(names))  # keep order
    return names


def select_best_model() -> Dict[str, str]:
    """Return {'name': <model_name>} with simple, reliable logic:
       1) FORCE_MODEL env overrides
       2) newest *.joblib in MODELS_DIR
       3) fallback -> 'baseline'
    """
    cfg = SelectorConfig()
    if cfg.force_model:
        log(f"[selector] FORCE_MODEL={cfg.force_model}")
        return {"name": cfg.force_model}

    cands = _list_candidates()
    for name in cands:
        # Prefer a saved artifact if exists; else accept baseline (train will occur on demand)
        if (MODELS_DIR / f"{name}.joblib").exists():
            log(f"[selector] selected saved model: {name}")
            return {"name": name}

    log(f"[selector] fallback: {cfg.fallback}")
    return {"name": cfg.fallback}


def evaluate_models() -> Dict[str, str]:
    """
    Determine which model to use for trading.

    The selection logic follows these steps:

      1. If the environment variable ``MODEL_TYPES`` or ``MODEL_CANDIDATES`` is set,
         treat it as a comma‑separated list of candidate model names and return the
         first entry. This allows an external configuration to override model
         selection (e.g., ``MODEL_TYPES=lstm,gbm,rf`` will select ``lstm``).

      2. Otherwise, call ``select_best_model`` to select among saved artifacts
         or fall back to the default ``baseline``.

    Returns
    -------
    dict
        A dictionary containing the selected model name under the key ``model``
        and a dummy ``score`` value for future extension.
    """
    # Check for explicit model list via environment variables
    env = os.getenv("MODEL_TYPES") or os.getenv("MODEL_CANDIDATES")
    if env:
        types = [m.strip() for m in env.split(',') if m.strip()]
        if types:
            name = types[0]
            log(f"[selector] MODEL_TYPES override: {name}")
            return {"model": name, "score": 0.0}
    # Fallback: choose the best available artifact or baseline
    selection = select_best_model()
    return {"model": selection["name"], "score": 0.0}