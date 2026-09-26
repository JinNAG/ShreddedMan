"""Storage and source-file bookkeeping for one document submission."""

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import uuid


DEFAULT_IMG_ROOT = Path(__file__).resolve().parents[1] / "img"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class Submission:
    directory: Path

    @property
    def source_images(self) -> Path:
        return self.directory / "source_images"

    @property
    def cropped_strips(self) -> Path:
        return self.directory / "cropped_strips"

    @property
    def normalized_strips(self) -> Path:
        return self.directory / "normalized_strips"

    @property
    def final_document(self) -> Path:
        return self.directory / "final_document"

    @property
    def manifest_path(self) -> Path:
        return self.directory / "submission.json"

    @property
    def order_path(self) -> Path:
        return self.directory / "order.json"

    def ensure_layout(self) -> None:
        for folder in (self.source_images, self.cropped_strips, self.normalized_strips, self.final_document):
            folder.mkdir(parents=True, exist_ok=True)

    def read_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"submission_id": self.directory.name, "status": "created", "sources": []}

    def save_manifest(self, manifest: dict) -> None:
        write_json(self.manifest_path, manifest)

    def invalidate_result(self) -> None:
        """Remove generated output and its verification together."""
        self.order_path.unlink(missing_ok=True)
        for name in ("document.png", "join_report.html", "join_report.json"):
            (self.final_document / name).unlink(missing_ok=True)

    def sources(self) -> list[Path]:
        if not self.source_images.is_dir():
            raise ValueError(f"Submission has no source_images directory: {self.directory}")
        # Include every supported image, including uploads added after creation.
        paths = sorted(
            (path for path in self.source_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: (path.name.casefold(), path.name),
        )
        if not paths:
            raise ValueError(f"No source images found in {self.source_images}")
        return paths


def create_submission(source_images: list[Path], img_root: Path = DEFAULT_IMG_ROOT) -> Submission:
    """Copy uploads into a new, uniquely reserved submission folder.

    Files with the same upload name get distinct stored names. Source files
    supplied by the caller are never moved or modified.
    """
    paths = [Path(path) for path in source_images]
    if not paths:
        raise ValueError("At least one source photo is required.")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Source photo does not exist: {path}")
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {path.suffix}")
    img_root = Path(img_root)
    img_root.mkdir(parents=True, exist_ok=True)
    while True:
        directory = img_root / uuid.uuid4().hex
        try:
            directory.mkdir()  # Exclusive reservation, also safe across workers.
            break
        except FileExistsError:
            continue
    submission = Submission(directory)
    try:
        submission.ensure_layout()
        used_names, sources = set(), []
        for path in paths:
            name, suffix = path.name, 2
            while name.casefold() in used_names:
                name = f"{path.stem}_{suffix}{path.suffix}"
                suffix += 1
            used_names.add(name.casefold())
            shutil.copyfile(path, submission.source_images / name)
            sources.append({"filename": name, "original_name": path.name})
        submission.save_manifest({
            "submission_id": directory.name, "status": "created", "sources": sources, "strips": [],
        })
    except Exception:
        # Only this freshly reserved folder belongs to the failed copy operation.
        shutil.rmtree(directory)
        raise
    return submission
