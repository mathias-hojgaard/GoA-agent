"""Few-shot store for confirmed extractions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rapidfuzz import fuzz


class FewShotStore:
    """Persists human-confirmed extractions and provides fuzzy similarity lookup."""

    def __init__(self, tenant_id: str, store_dir: str = "./data/fewshot"):
        safe_id = tenant_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        self.path = Path(store_dir) / f"{safe_id}.json"
        self.entries: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2))

    def add_confirmed(self, field_name: str, value: str, context: str) -> None:
        """Add a human-confirmed extraction."""
        self.entries.append(
            {
                "field_name": field_name,
                "value": value,
                "context": context,
                "confirmed_at": datetime.now().isoformat(),
            }
        )
        self._save()

    def find_similar(self, value: str, field_name: str, top_k: int = 5) -> float:
        """Return max similarity score (0.0-1.0) against confirmed extractions."""
        relevant = [e for e in self.entries if e["field_name"] == field_name]
        if not relevant:
            return 0.0
        scores = [
            fuzz.ratio(value.lower(), e["value"].lower()) / 100.0 for e in relevant
        ]
        return max(scores) if scores else 0.0

    def get_few_shot_examples(self, field_name: str, top_k: int = 3) -> list[dict]:
        """Return top_k most recent confirmed extractions for this field."""
        relevant = [e for e in self.entries if e["field_name"] == field_name]
        return relevant[-top_k:]
