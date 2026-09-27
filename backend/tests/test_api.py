from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import patch
import uuid

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import create_app
from submission import Submission
from test_assemble_document import source_photo


class ApiTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "img"
        self.app = create_app(self.root)
        self.client = self.enterContext(TestClient(self.app))
        ok, encoded = cv2.imencode(".png", source_photo(1))
        self.assertTrue(ok)
        self.photo = encoded.tobytes()

    def upload(self, name="photo.png", content=None, **kwargs):
        return self.client.post(
            "/api/submissions",
            files=[("files", (name, self.photo if content is None else content, "image/png"))],
            **kwargs,
        )

    def wait_for_result(self, status_url):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            response = self.client.get(status_url)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            if body["status"] in ("complete", "failed"):
                return body
            time.sleep(0.01)
        self.fail("Background processing did not finish within 15 seconds.")

    def test_upload_duplicate_names_runs_real_pipeline_and_serves_png(self):
        with redirect_stdout(io.StringIO()):
            response = self.client.post("/api/submissions", files=[
                ("files", ("photo.png", self.photo, "image/png")),
                ("files", ("photo.png", self.photo, "image/png")),
            ])
            self.assertEqual(response.status_code, 202, response.text)
            accepted = response.json()
            self.assertEqual(uuid.UUID(accepted["submission_id"]).version, 4)
            self.assertEqual(accepted["status"], "queued")
            self.assertIsNone(accepted["document_url"])
            self.assertEqual(response.headers["location"], accepted["status_url"])
            result = self.wait_for_result(accepted["status_url"])
        self.assertEqual(result["status"], "complete", result)
        self.assertIsNone(result["error"])
        self.assertIsInstance(result["review_required"], bool)
        self.assertTrue(result["join_report_url"])
        submission = Submission(self.root / accepted["submission_id"])
        sources = submission.read_manifest()["sources"]
        self.assertEqual([entry["filename"] for entry in sources], ["photo.png", "photo_2.png"])
        self.assertEqual([entry["original_name"] for entry in sources], ["photo.png", "photo.png"])
        self.assertEqual(len(json.loads(submission.order_path.read_text())["order"]), 4)
        document = self.client.get(result["document_url"])
        self.assertEqual(document.status_code, 200)
        self.assertEqual(document.headers["content-type"], "image/png")
        report = self.client.get(result["join_report_url"])
        self.assertEqual(report.status_code, 200)
        self.assertIn("Strip join verification", report.text)
        self.assertEqual(self.client.get(result["status_url"] + "/join_report.json").status_code, 200)
        self.assertEqual(self.client.get(result["status_url"] + "/document.png").status_code, 200)
        self.assertTrue(document.headers["content-disposition"].startswith("inline"))
        self.assertEqual(document.content, (submission.final_document / "document.png").read_bytes())
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(document.content, np.uint8), cv2.IMREAD_COLOR))
        self.assertEqual(self.client.get(accepted["status_url"]).headers["cache-control"], "no-store")

    def test_requests_remain_responsive_while_processing_and_jobs_are_queued(self):
        started, release = Event(), Event()

        def process(directory, rotation):
            submission = Submission(directory)
            manifest = submission.read_manifest()
            manifest["status"] = "processing"
            submission.save_manifest(manifest)
            started.set()
            if not release.wait(10):
                raise TimeoutError("Test worker was not released.")

        with patch("api.process_submission", side_effect=process) as worker:
            try:
                first = self.upload(data={"rotation": "90"})
                self.assertEqual(first.status_code, 202)
                self.assertTrue(started.wait(3))
                first_url = first.json()["status_url"]
                self.assertEqual(self.client.get(first_url).json()["status"], "processing")
                self.assertEqual(self.client.get(first_url + "/document").status_code, 409)
                second = self.upload()
                self.assertEqual(second.status_code, 202)
                second_url = second.json()["status_url"]
                self.assertNotEqual(first_url, second_url)
                self.assertEqual(self.client.get(second_url).json()["status"], "queued")
                self.assertEqual(worker.call_count, 1)
                self.assertEqual(worker.call_args.args[1], "90")
            finally:
                release.set()
                self.app.state.executor.submit(lambda: None).result(timeout=10)
        self.assertEqual(worker.call_count, 2)

    def test_real_processing_failure_is_reported_without_exposing_server_paths(self):
        ok, blank = cv2.imencode(".png", np.zeros((100, 100, 3), np.uint8))
        self.assertTrue(ok)
        with self.assertLogs("api", level="ERROR"), redirect_stdout(io.StringIO()):
            response = self.upload(content=blank.tobytes())
            self.assertEqual(response.status_code, 202)
            result = self.wait_for_result(response.json()["status_url"])
            # The pipeline records failure before the API wrapper logs it.
            self.app.state.executor.submit(lambda: None).result(timeout=10)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["document_url"])
        self.assertIn("Check the photos", result["error"])
        self.assertNotIn(str(self.root), json.dumps(result))
        manifest = Submission(self.root / result["submission_id"]).read_manifest()
        self.assertIn("No paper strips detected", manifest["error"])
        self.assertEqual(self.client.get(result["status_url"] + "/document").status_code, 409)

    def test_missing_files_and_invalid_rotation_are_rejected(self):
        self.assertEqual(self.client.post("/api/submissions").status_code, 422)
        self.assertEqual(self.upload(data={"rotation": "45"}).status_code, 422)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_uploads_never_create_submissions(self):
        cases = [
            ("notes.txt", b"text", 415),
            ("empty.png", b"", 400),
            ("fake.png", b"not a PNG", 415),
            ("../outside.png", self.photo, 400),
            ("folder\\outside.png", self.photo, 400),
            ("CON.png", self.photo, 400),
        ]
        for name, content, expected in cases:
            with self.subTest(name=name):
                response = self.upload(name, content)
                self.assertEqual(response.status_code, expected, response.text)
                self.assertEqual(list(self.root.iterdir()), [])
        # A bad later photo must not leave a partial persistent submission.
        response = self.client.post("/api/submissions", files=[
            ("files", ("good.png", self.photo)),
            ("files", ("bad.png", b"broken")),
        ])
        self.assertEqual(response.status_code, 415)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_upload_size_and_count_limits(self):
        for setting, limit in (("MAX_FILE_BYTES", len(self.photo) - 1), ("MAX_TOTAL_BYTES", len(self.photo) - 1), ("MAX_FILES", 0)):
            with self.subTest(setting=setting), patch("api." + setting, limit):
                self.assertEqual(self.upload().status_code, 413)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_full_queue_rejects_upload_without_creating_a_submission(self):
        slots = self.app.state.job_slots
        acquired = 0
        try:
            while slots.acquire(blocking=False):
                acquired += 1
            response = self.upload()
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["retry-after"], "5")
            self.assertEqual(list(self.root.iterdir()), [])
        finally:
            for _ in range(acquired):
                slots.release()

    def test_scheduler_failure_is_recorded_and_capacity_is_released(self):
        with patch.object(self.app.state.executor, "submit", side_effect=RuntimeError("stopped")):
            self.assertEqual(self.upload().status_code, 503)
        [directory] = list(self.root.iterdir())
        self.assertEqual(Submission(directory).read_manifest()["status"], "failed")
        # Repeated invalid requests also release their reserved queue slots.
        for _ in range(10):
            self.assertEqual(self.upload(content=b"bad").status_code, 415)

    def test_missing_or_malformed_submission_ids_are_not_resolved_as_paths(self):
        for submission_id in (uuid.uuid4().hex, "not-an-id", "%2E%2E%5Coutside"):
            for suffix in ("", "/document"):
                with self.subTest(submission_id=submission_id, suffix=suffix):
                    response = self.client.get(f"/api/submissions/{submission_id}{suffix}")
                    self.assertEqual(response.status_code, 404)

    def test_existing_submission_is_readable_and_missing_output_returns_404(self):
        submission = Submission(self.root / uuid.uuid4().hex)
        submission.ensure_layout()
        submission.save_manifest({"status": "complete", "result": "final_document/document.png"})
        url = f"/api/submissions/{submission.directory.name}"
        self.assertEqual(self.client.get(url).json()["status"], "complete")
        self.assertEqual(self.client.get(url + "/document").status_code, 404)
        (submission.final_document / "document.png").write_bytes(self.photo)
        self.assertEqual(self.client.get(url + "/document").content, self.photo)

    def test_api_docs_describe_the_upload_and_result_contract(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        upload = schema["paths"]["/api/submissions"]["post"]
        self.assertIn("multipart/form-data", upload["requestBody"]["content"])
        self.assertIn("202", upload["responses"])
        document = schema["paths"]["/api/submissions/{submission_id}/document"]["get"]
        self.assertIn("image/png", document["responses"]["200"]["content"])

    def test_local_frontend_can_upload_poll_and_download_across_origins(self):
        for origin in ("http://127.0.0.1:3000", "http://localhost:3000"):
            with self.subTest(origin=origin):
                preflight = self.client.options("/api/submissions", headers={
                    "Origin": origin, "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                })
                self.assertEqual(preflight.status_code, 200)
                self.assertEqual(preflight.headers["access-control-allow-origin"], origin)
                # Even validation errors must be readable by the frontend.
                invalid = self.client.post("/api/submissions", headers={"Origin": origin})
                self.assertEqual(invalid.status_code, 422)
                self.assertEqual(invalid.headers["access-control-allow-origin"], origin)
                submission = Submission(self.root / uuid.uuid4().hex)
                submission.ensure_layout()
                submission.save_manifest({"status": "complete"})
                (submission.final_document / "document.png").write_bytes(self.photo)
                url = f"/api/submissions/{submission.directory.name}"
                for path in (url, url + "/document"):
                    response = self.client.get(path, headers={"Origin": origin})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers["access-control-allow-origin"], origin)

    def test_frontend_origin_allowlist_can_be_configured(self):
        with patch.dict("os.environ", {"SHREDDEDMAN_FRONTEND_ORIGINS": " http://localhost:5173/ "}):
            with TestClient(create_app(self.root)) as client:
                for origin, expected in (("http://localhost:5173", 200), ("http://localhost:3000", 400)):
                    response = client.options("/api/submissions", headers={
                        "Origin": origin, "Access-Control-Request-Method": "POST",
                    })
                    self.assertEqual(response.status_code, expected)
                    if expected == 200:
                        self.assertEqual(response.headers["access-control-allow-origin"], origin)
                    else:
                        self.assertNotIn("access-control-allow-origin", response.headers)


if __name__ == "__main__":
    unittest.main()
