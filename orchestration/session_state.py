"""Session state persistence.

Used for two things from the spec: "check for state (recovery from
previous session)" at startup, and "save state" as part of unhandled-error
handling in the main loop.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

DEFAULT_STATE_PATH = Path("session_state.json")


@dataclass
class SessionState:
    last_updated: datetime
    current_regime_label: Optional[str]
    current_regime_state: Optional[int]
    model_trained_at: Optional[datetime]
    open_positions: Dict[str, float]  # symbol -> shares
    equity: float
    circuit_breaker_size_multiplier: float
    trading_halted: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["last_updated"] = self.last_updated.isoformat()
        d["model_trained_at"] = self.model_trained_at.isoformat() if self.model_trained_at else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        d = dict(d)
        d["last_updated"] = datetime.fromisoformat(d["last_updated"])
        d["model_trained_at"] = datetime.fromisoformat(d["model_trained_at"]) if d.get("model_trained_at") else None
        return cls(**d)


def save_session_state(state: SessionState, path: Path = DEFAULT_STATE_PATH) -> None:
    Path(path).write_text(json.dumps(state.to_dict(), indent=2))


def load_session_state(path: Path = DEFAULT_STATE_PATH) -> Optional[SessionState]:
    """Returns None if no prior state exists (e.g. first-ever run) —
    that's an expected case, not an error.
    """
    path = Path(path)
    if not path.exists():
        return None
    return SessionState.from_dict(json.loads(path.read_text()))
