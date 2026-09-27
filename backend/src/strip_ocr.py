"""Local Tesseract OCR with bounded subprocesses and a content-addressed cache."""

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

import cv2
import numpy as np

from submission import write_json


DEFAULT_OCR_WORKERS = min(8, max(1, os.cpu_count() or 1))


class TesseractOCR:
    def __init__(self, mode="auto", language="eng", cache_path: Path | None = None):
        if mode not in ("auto", "required", "off"):
            raise ValueError("OCR mode must be auto, required, or off.")
        self.language, self.cache_path = language, cache_path
        self.cache, self.lock = {}, threading.Lock()
        self.calls = self.cache_hits = 0
        self.status, self.reason, self.version = "disabled", "OCR was explicitly disabled.", None
        self.executable = shutil.which("tesseract") if mode != "off" else None
        if mode != "off":
            self.status, self.reason = "unavailable", "Install Tesseract and its requested language data to enable OCR."
            if self.executable:
                try:
                    version = subprocess.run([self.executable, "--version"], capture_output=True, text=True, check=True, timeout=10)
                    languages = subprocess.run([self.executable, "--list-langs"], capture_output=True, text=True, check=True, timeout=10)
                    available = set(languages.stdout.splitlines()[1:])
                    if not set(language.split("+")).issubset(available):
                        raise ValueError(f"Tesseract language data is missing for {language!r}.")
                    self.version = version.stdout.splitlines()[0]
                    self.status, self.reason = "enabled", None
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    self.reason = str(error)
            if mode == "required" and self.status != "enabled":
                raise ValueError(self.reason)
        if cache_path and cache_path.is_file():
            try:
                stored = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(stored, dict) and stored.get("schema") == 1 and isinstance(stored.get("entries"), dict):
                    self.cache = stored["entries"]
            except (ValueError, OSError):
                pass  # A disposable cache must not prevent processing.

    @property
    def enabled(self):
        return self.status == "enabled"

    def recognize(self, gray: np.ndarray) -> list[dict]:
        """Return word boxes in input-image pixels, with engine confidence 0–1.

        Dictionaries are disabled so partial words and names remain candidates.
        Unexpected OCR failures abort processing instead of fabricating scores.
        """
        if not self.enabled:
            return []
        key = hashlib.sha256(
            f"v1|{self.version}|{self.language}|psm6|no-dictionaries|{gray.shape}".encode() + gray.tobytes()
        ).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]
        ok, encoded = cv2.imencode(".png", gray)
        if not ok:
            raise ValueError("Could not encode the OCR analysis image.")
        command = [self.executable, "stdin", "stdout", "--psm", "6", "-l", self.language,
                   "-c", "load_system_dawg=0", "-c", "load_freq_dawg=0", "tsv"]
        try:
            process = subprocess.run(
                command, input=encoded.tobytes(), capture_output=True, check=True, timeout=30,
                env={**os.environ, "OMP_THREAD_LIMIT": "1"},
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Tesseract failed: {error.stderr.decode(errors='replace').strip()}") from error
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Tesseract failed: {error}") from error
        words = []
        for row in csv.DictReader(io.StringIO(process.stdout.decode("utf-8")), delimiter="\t"):
            if row["level"] != "5" or not row["text"].strip():
                continue
            words.append({
                "text": row["text"], "confidence": float(np.clip(float(row["conf"]) / 100, 0, 1)),
                "box": [int(row[name]) for name in ("left", "top", "width", "height")],
            })
        with self.lock:
            self.calls += 1
            self.cache[key] = words
        return words

    def flush(self):
        if self.cache_path and self.enabled:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.cache_path, {"schema": 1, "entries": self.cache})

    def metadata(self):
        return {"engine": "tesseract", "status": self.status, "reason": self.reason,
                "version": self.version, "language": self.language, "dictionaries_enabled": False,
                "calls": self.calls, "cache_hits": self.cache_hits}
