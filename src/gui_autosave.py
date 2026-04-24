"""
Sidecar autosave storage for unsaved Field Selection GUI edits.

The autosave file is intentionally separate from the TinyDB subject database.
It stores only unsaved GUI working state so recovery never silently promotes
experiments into canonical analysis results.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from tempfile import NamedTemporaryFile


AUTOSAVE_VERSION = 1


class GUIAutosave:
    def __init__(self, db_path, analysis_id, analysis_name=None):
        self.db_path = Path(db_path)
        self.analysis_id = str(analysis_id)
        self.analysis_name = analysis_name
        self.path = self._autosave_path()

    def _autosave_path(self):
        short_id = sanitize_filename_part(self.analysis_id)[:5] or "id"
        if self.analysis_name:
            label = f"{sanitize_filename_part(self.analysis_name)}-{short_id}"
        else:
            label = f"analysis-{short_id}"
        return self.db_path.with_name(
            f"{self.db_path.stem}.autosave.{label}")

    def set_analysis_name(self, analysis_name):
        self.analysis_name = analysis_name
        self.path = self._autosave_path()

    def load(self):
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, JSONDecodeError):
            return None
        if payload.get("analysis_id") != self.analysis_id:
            return None
        if payload.get("version") != AUTOSAVE_VERSION:
            return None
        if not has_autosave_changes(payload):
            return None
        return payload

    def write(self, *, overview=None, detail_sites=None):
        overview = overview or {}
        detail_sites = detail_sites or {}
        if not overview and not detail_sites:
            self.delete()
            return None

        payload = {
            "version": AUTOSAVE_VERSION,
            "db_path": str(self.db_path),
            "analysis_id": self.analysis_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "overview": normalize_json_value(overview),
            "detail_sites": normalize_json_value(detail_sites),
        }
        self._atomic_write(payload)
        return payload

    def delete(self):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _atomic_write(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, self.path)


def has_autosave_changes(payload):
    return bool(payload.get("overview") or payload.get("detail_sites"))


def sanitize_filename_part(value):
    text = str(value).strip()
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)
    return safe.strip("_") or "unknown"


def normalize_json_value(value):
    if isinstance(value, dict):
        return {str(k): normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(v) for v in value]
    if hasattr(value, "item"):
        return normalize_json_value(value.item())
    return value
