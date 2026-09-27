"""HTTP interface; from the repo root run uvicorn api:app --app-dir backend/src."""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore
from typing import Annotated, Literal

import cv2
from assemble_document import process_submission
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from submission import DEFAULT_IMG_ROOT, IMAGE_SUFFIXES, Submission, create_submission

logger = logging.getLogger(__name__)
MAX_FILES = 20
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_PENDING_JOBS = 8


class SubmissionStatus(BaseModel):
    submission_id: str
    status: Literal[
        "created",
        "queued",
        "processing",
        "detected",
        "normalized",
        "complete",
        "failed",
    ]
    status_url: str
    document_url: str | None = None
    join_report_url: str | None = None
    review_required: bool = False
    error: str | None = None


def _upload_name(name: str | None) -> str:
    """Accept ordinary filenames on both Windows and POSIX, never paths."""
    if (
        not name
        or len(name) > 200
        or name in (".", "..")
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
        or name != name.rstrip(" .")
        or re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])", name.split(".")[0])
    ):
        raise HTTPException(400, "Each upload must have a plain, valid filename.")
    if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(
            415, "Supported images: JPG, JPEG, PNG, BMP, TIFF, and WebP."
        )
    return name


def _save_uploads(files: list[UploadFile], staging: Path) -> list[Path]:
    """Validate all photos before creating any persistent submission."""
    if not files:
        raise HTTPException(400, "At least one photo is required.")
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"Upload at most {MAX_FILES} photos per submission.")
    paths, total = [], 0
    for index, upload in enumerate(files):
        name = _upload_name(upload.filename)
        # Separate temporary folders preserve duplicate original filenames.
        folder = staging / str(index)
        folder.mkdir()
        path = folder / name
        size = 0
        with path.open("wb") as target:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                total += len(chunk)
                if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                    raise HTTPException(
                        413,
                        "Photos exceed the 25 MiB/file or 100 MiB/submission limit.",
                    )
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, "Uploaded photos must not be empty.")
        try:
            readable = cv2.imread(str(path)) is not None
        except cv2.error:
            readable = False
        if not readable:
            raise HTTPException(
                415, "An uploaded file could not be decoded as an image."
            )
        paths.append(path)
    return paths


def _run_submission(submission: Submission, rotation: str) -> None:
    try:
        process_submission(submission.directory, rotation)
    except Exception as error:
        # Preserve the cause in the local manifest as well as server logs.
        # describe() keeps internal details out of the public response.
        logger.exception(
            "Processing failed for submission %s", submission.directory.name
        )
        manifest = submission.read_manifest()
        manifest.update(
            status="failed",
            error=str(error),
        )
        manifest.pop("result", None)
        submission.save_manifest(manifest)


def create_app(img_root: Path | None = None) -> FastAPI:
    """Create an isolated API instance; existing CLI submissions remain readable."""
    root = Path(
        img_root
        if img_root is not None
        else os.environ.get("SHREDDEDMAN_IMG_ROOT", DEFAULT_IMG_ROOT)
    ).resolve()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        root.mkdir(parents=True, exist_ok=True)
        application.state.job_slots = BoundedSemaphore(MAX_PENDING_JOBS)
        application.state.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="reconstruct"
        )
        try:
            yield
        finally:
            # Finish accepted jobs on a graceful shutdown.
            await run_in_threadpool(application.state.executor.shutdown, wait=True)

    application = FastAPI(title="ShreddedMan API", version="1.0.0", lifespan=lifespan)
    origins = os.environ.get(
        "SHREDDEDMAN_FRONTEND_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000"
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip().rstrip("/") for origin in origins.split(",") if origin.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def find_submission(submission_id: str) -> Submission:
        if not re.fullmatch(r"[0-9a-f]{32}", submission_id):
            raise HTTPException(404, "Submission not found.")
        directory = (root / submission_id).resolve()
        if directory.parent != root or not (directory / "submission.json").is_file():
            raise HTTPException(404, "Submission not found.")
        return Submission(directory)

    def describe(submission: Submission, manifest: dict) -> SubmissionStatus:
        submission_id = submission.directory.name
        status_url = f"/api/submissions/{submission_id}"
        failed = manifest["status"] == "failed"
        return SubmissionStatus(
            submission_id=submission_id,
            status=manifest["status"],
            status_url=status_url,
            document_url=f"{status_url}/document"
            if manifest["status"] == "complete"
            else None,
            join_report_url=f"{status_url}/join-report"
            if manifest["status"] == "complete"
            else None,
            review_required=manifest["status"] == "complete" and bool(manifest.get("review_required", False)),
            error="Document reconstruction failed. Check the photos and try again."
            if failed
            else None,
        )

    @application.post(
        "/api/submissions", status_code=202, response_model=SubmissionStatus
    )
    def upload_submission(
        response: Response,
        files: Annotated[
            list[UploadFile], File(description="Photos of strips from one document.")
        ],
        rotation: Annotated[Literal["auto", "0", "90", "180", "270"], Form()] = "auto",
    ) -> SubmissionStatus:
        slots = application.state.job_slots
        if not slots.acquire(blocking=False):
            raise HTTPException(
                503,
                "The processing queue is full. Try again shortly.",
                headers={"Retry-After": "5"},
            )
        scheduled = False
        try:
            with TemporaryDirectory(prefix="shreddedman-upload-") as staging:
                paths = _save_uploads(files, Path(staging))
                submission = create_submission(paths, root)
            manifest = submission.read_manifest()
            manifest.update(status="queued")
            submission.save_manifest(manifest)
            result = describe(submission, manifest)
            try:
                future = application.state.executor.submit(
                    _run_submission, submission, rotation
                )
            except RuntimeError as error:
                manifest.update(
                    status="failed", error="The server could not schedule processing."
                )
                submission.save_manifest(manifest)
                raise HTTPException(
                    503, "The server is shutting down. Try again shortly."
                ) from error
            future.add_done_callback(lambda completed: slots.release())
            scheduled = True
            response.headers["Location"] = result.status_url
            response.headers["Cache-Control"] = "no-store"
            return result
        finally:
            if not scheduled:
                slots.release()
            for upload in files:
                upload.file.close()

    @application.get(
        "/api/submissions/{submission_id}", response_model=SubmissionStatus
    )
    def submission_status(submission_id: str, response: Response) -> SubmissionStatus:
        submission = find_submission(submission_id)
        response.headers["Cache-Control"] = "no-store"
        return describe(submission, submission.read_manifest())

    @application.get(
        "/api/submissions/{submission_id}/document",
        response_class=FileResponse,
        responses={200: {"content": {"image/png": {}}}},
    )
    @application.get("/api/submissions/{submission_id}/document.png", include_in_schema=False)
    def submission_document(submission_id: str) -> FileResponse:
        submission = find_submission(submission_id)
        if submission.read_manifest()["status"] != "complete":
            raise HTTPException(
                409, "The document is not available. Check the submission status."
            )
        path = (submission.final_document / "document.png").resolve()
        if not path.is_relative_to(submission.directory) or not path.is_file():
            raise HTTPException(404, "Document not found.")
        return FileResponse(
            path,
            media_type="image/png",
            filename="document.png",
            content_disposition_type="inline",
        )

    @application.get("/api/submissions/{submission_id}/join-report", response_class=FileResponse)
    def submission_join_report(submission_id: str) -> FileResponse:
        submission = find_submission(submission_id)
        if submission.read_manifest()["status"] != "complete":
            raise HTTPException(409, "The join report is not available yet.")
        path = (submission.final_document / "join_report.html").resolve()
        if not path.is_relative_to(submission.directory) or not path.is_file():
            raise HTTPException(404, "Join report not found.")
        return FileResponse(path, media_type="text/html", content_disposition_type="inline")

    @application.get("/api/submissions/{submission_id}/join_report.json", include_in_schema=False)
    def submission_join_report_json(submission_id: str) -> FileResponse:
        submission = find_submission(submission_id)
        if submission.read_manifest()["status"] != "complete":
            raise HTTPException(409, "The join report is not available yet.")
        path = (submission.final_document / "join_report.json").resolve()
        if not path.is_relative_to(submission.directory) or not path.is_file():
            raise HTTPException(404, "Join report not found.")
        return FileResponse(path, media_type="application/json", content_disposition_type="inline")

    return application


app = create_app()
