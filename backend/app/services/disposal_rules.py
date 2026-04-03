"""Service for managing disposal rules with JSON file persistence."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict

from app.config import BASE_DIR


# Default disposal rules (fallback if file doesn't exist)
DEFAULT_RULES = {
    "plastic": {
        "title": "Plastic (Dry Waste)",
        "description": "Rinse plastic containers and place them in the blue dry waste bin. Keep plastic separate from wet waste to improve recycling quality."
    },
    "paper": {
        "title": "Paper/Cardboard",
        "description": "Keep paper and cardboard dry before placing in the blue recycling bin. Avoid mixing paper with oily or food-contaminated waste."
    },
    "organic": {
        "title": "Organic / Wet Waste",
        "description": "Put food scraps and biodegradable waste in the green wet waste bin or compost pit. Remove plastic wrappers before disposal."
    },
    "glass": {
        "title": "Glass (Dry Recyclable)",
        "description": "Place clean glass bottles and jars in the dry recycling stream. Wrap broken glass securely before handing over to collection workers."
    },
    "metal": {
        "title": "Metal (Dry Recyclable)",
        "description": "Rinse cans or metal containers and place them in the blue dry waste bin. Sharp metal items should be wrapped safely before disposal."
    },
    "e-waste": {
        "title": "E-Waste",
        "description": "Do not throw e-waste into regular bins. Submit electronics to authorized e-waste collection centers or municipal drives."
    }
}


class DisposalRulesService:
    """Manages disposal rules with JSON file storage."""

    def __init__(self, rules_file_path: Path | None = None):
        self._lock = Lock()
        if rules_file_path is None:
            # Store in backend/data directory
            data_dir = BASE_DIR / "data"
            data_dir.mkdir(exist_ok=True)
            self._rules_file = data_dir / "disposal_rules.json"
        else:
            self._rules_file = Path(rules_file_path)

    def get_rules(self) -> Dict[str, Dict[str, str]]:
        """Load disposal rules from JSON file or return defaults."""
        with self._lock:
            if not self._rules_file.exists():
                # Initialize with defaults
                self._save_rules_unsafe(DEFAULT_RULES)
                return DEFAULT_RULES.copy()

            try:
                with open(self._rules_file, "r", encoding="utf-8") as f:
                    rules = json.load(f)
                return rules
            except (json.JSONDecodeError, IOError) as exc:
                print(f"Warning: Failed to load disposal rules from {self._rules_file}: {exc}")
                return DEFAULT_RULES.copy()

    def save_rules(self, rules: Dict[str, Dict[str, str]]) -> None:
        """Save disposal rules to JSON file."""
        with self._lock:
            self._save_rules_unsafe(rules)

    def _save_rules_unsafe(self, rules: Dict[str, Dict[str, str]]) -> None:
        """Internal save without lock (assumes caller holds lock)."""
        try:
            with open(self._rules_file, "w", encoding="utf-8") as f:
                json.dump(rules, f, indent=2, ensure_ascii=False)
        except IOError as exc:
            print(f"Error: Failed to save disposal rules to {self._rules_file}: {exc}")
            raise


# Global disposal rules service instance
disposal_rules_service = DisposalRulesService()
