from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass(slots=True)
class Alert:
    timestamp: str
    category: str
    severity: str
    score: float
    message: str

    @classmethod
    def build(cls, category: str, score: float, message: str) -> "Alert":
        severity = "high" if score >= 0.9 else "medium" if score >= 0.75 else "low"
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            severity=severity,
            score=round(score, 4),
            message=message,
        )

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)
