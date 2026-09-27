# ShreddedMan

The Python backend lives in `backend/`: code in `backend/src`, tests in
`backend/tests`, and submissions in `backend/img`. The commands below run from
the repository root.

Create or activate the backend virtual environment and install dependencies:

```sh
python3 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install -r backend/requirements.txt
```

For OCR-assisted sorting, also install the **Tesseract executable** and language
data (English by default). On macOS:

```sh
brew install tesseract
```

On Debian/Ubuntu, use `sudo apt install tesseract-ocr tesseract-ocr-eng`.
No extra Python OCR package or external API is required; images stay local.

Create a submission from one or more photos of **one document**:

```sh
python backend/src/main.py /path/to/photo1.jpg /path/to/photo2.jpg
```

The command prints the new submission path and result path. Each call creates a
new UUID folder under `backend/img/`, regardless of the working directory;
`--img-root /another/location` changes that parent. Input paths and explicit
relative output paths are resolved from your current working directory.
Uploaded files are copied without modifying the originals. Duplicate upload
names are given distinct stored names.

```text
backend/img/
  <random-submission-id>/
    source_images/
      photo1.jpg
      photo2.jpg
    cropped_strips/
      strip1.png
      strip2.png
      ...
    normalized_strips/
      strip1.png
      strip2.png
      ...
    final_document/
      document.png
      join_report.html
      join_report.json
    submission.json
    order.json
```

Detection includes every JPG/JPEG, PNG, BMP, TIFF, or WebP file directly inside
`source_images`, with case-insensitive extensions. Crops from all photos share
one consecutive strip-number sequence. Normalization preserves these filenames.
Sorting reads the entire pooled directory and writes the assembled image and
join reports to `final_document`. It never renames or overwrites the normalized
strip images.

`submission.json` records upload names, each strip's source photo, rotation,
and processing status. `order.json` contains estimated positions, source
mappings, all pair scores, alternative orders, and ambiguous joins. Stored
paths are relative to the submission so it can be moved as a unit. Blank/low-ink
strips stay in the crop and normalized folders, with unresolved positions in the
report; they are excluded from the final image for now. On text-rich pages, OCR
also leaves near-blank strips unresolved unless one neighboring join has
clearly stronger evidence than its alternatives.

Rerun a submission after adding or removing source photos:

```sh
python backend/src/main.py --submission backend/img/<submission-id>
```

Run individual stages for an existing submission:

```sh
python backend/src/detect_paper.py backend/img/<submission-id>
python backend/src/normalize_strips.py backend/img/<submission-id>
python backend/src/sort_strips.py backend/img/<submission-id>
```

From inside `backend/`, omit the `backend/` prefix, for example
`python src/main.py --submission img/<submission-id>`.

Open `final_document/join_report.html` in a browser to review every join. It
shows positions **counted from the left, starting at 1**, original strip names,
low/medium/high confidence with a 0–100 support score, visual and OCR evidence,
competing neighbors, and image snippets of weak text regions. The matching
`join_report.json` includes all recognized fragments and per-line checks for
application use. Confidence is a heuristic support score, **not a probability
of correctness**; broken text may remain even at a high score.
The full-page OCR check can move groups of strips across weak joins before
choosing the most readable reconstruction it found. Small remaining row shifts
are refined from ink crossing the selected joins.
The API also returns `review_required` and `join_report_url` for a completed
reconstruction. The website links to the join report when review is suggested.

OCR defaults to `--ocr auto`: use Tesseract when available, otherwise keep
visual sorting and explicitly report that OCR is unavailable. Require it with:

```sh
python backend/src/sort_strips.py backend/img/<submission-id> --ocr required
```

Use `--ocr off` for visual-only sorting, `--ocr-language eng` to choose installed
language data (also accepts combinations such as `eng+deu`), and
`--ocr-workers 4` to control concurrent OCR jobs (1–8). These flags also work
with `backend/src/main.py`. Python callers can pass `ocr_mode`, `ocr_language`, and
`ocr_workers` to `assemble_document`, `process_submission`, or `sort_submission`.
The first run evaluates every directed text-strip pair and can take several
minutes. A disposable `.ocr_cache.json` in the submission folder speeds up
reruns; changed analysis pixels, engine versions, or languages use new entries.

Detection invalidates the old normalized strips and assembled result.
Normalization invalidates the old assembled result, including join reports.
Reprocessing removes stale numbered strips when the new detection contains
fewer pieces. Auto rotation
turns horizontal strips 90 degrees counterclockwise; use `--rotation 0`, `90`,
`180`, or `270` on the full pipeline or detection command when needed.

For application integration, add `backend/src` to the Python import path and call:

```python
from pathlib import Path
from assemble_document import assemble_document, process_submission
from submission import create_submission

# Copy uploaded files and run the full pipeline synchronously.
result = assemble_document([Path("/uploads/part1.jpg"), Path("/uploads/part2.jpg")])
result_path = Path(result["submission_dir"]) / result["result"]
join_report_path = Path(result["submission_dir"]) / result["verification"]["html_report"]

# Or create storage now and run processing later in a worker.
submission = create_submission([Path("/uploads/part1.jpg")])
result = process_submission(submission.directory)
```

Separate submissions can be processed independently. A single submission
should have one processing job at a time. The Python pipeline is synchronous;
the HTTP API below schedules it in a background worker. The frontend sends all
selected photos as one submission and displays the result in the Edited Image box.

## Backend API

From the repository root, with your virtual environment activated:

```sh
python -m uvicorn api:app --app-dir backend/src --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try uploads and view the full API reference.

Run `npm start` in another terminal at the repository root and open
http://localhost:3000. The form accepts up to 20 PNG/JPEG photos of one document,
up to 20 MiB per photo and 100 MiB total. It polls reconstruction progress and
loads the completed PNG into the existing Edited Image box. The Node server
uses `http://127.0.0.1:8000` by default; set `BACKEND_URL` for a different Python
API origin. Both servers must be running. Restart Node after editing `server.js`.

| Method | Endpoint | Result |
| --- | --- | --- |
| POST | `/api/submissions` | Upload photos; receive a submission ID and status URL (HTTP 202). |
| GET | `/api/submissions/{id}` | Check progress and get the result URL or error. |
| GET | `/api/submissions/{id}/document` | Retrieve the completed PNG. |
| GET | `/api/submissions/{id}/join-report` | Review weak joins in HTML. |

Send photos of one document in the multipart field `files`, with optional
`rotation` (`auto`, `0`, `90`, `180`, or `270`). Poll the returned `status_url`
until `complete` or `failed`, then use `document_url` or display `error`.

Files are stored under `img/`; set `SHREDDEDMAN_IMG_ROOT` to change the location.
This local-development API uses one background worker and an in-memory queue;
unfinished jobs must be resubmitted after a forced stop.

## Reconstruction and tests

Sorting compares all ordered pairs using full-height edge ink and text-row
alignment, with vertical scale/offset search and small smooth local corrections.
When text-bearing crops differ in height by at least 8% of their median, the
analysis centers short strips in transparent padding. This preserves their
vertical pixel spacing; saved strip PNGs are unchanged.
The final page also checks nearby strips against each other to correct a
vertical offset when one adjacent match locks onto the wrong text line.
With OCR enabled, each pair is also reconstructed and analyzed with dictionary
correction disabled, allowing incomplete words and names. OCR rewards readable
ink near the cut and penalizes unrecognized ink, rather than trusting only the
characters that happened to be recognized. The pair score combines visual
matching (50%) and OCR ink evidence (50%). Stroke continuity is shown in the
report as supporting evidence.

The first ordering maximizes the sum of adjacent-pair scores. Exhaustive search
handles up to eight text strips; larger samples use an integer optimizer. The
strongest orders (up to five) then undergo OCR checks across three neighboring strips.
Up to four refinement rounds examine the 24 best untested swaps/moves ranked by
pair score, including non-adjacent changes. A change must improve the complete
objective (25% mean pair score, 75% mean three-strip OCR evidence), including
joins it might damage elsewhere. This bounded context search does not guarantee
the globally best order or continue indefinitely until text looks correct.
The full-page check also measures how many complete English OCR words appear in
the bundled public-domain Webster word list. It uses that evidence to test
block moves across weak joins. These checks rank candidates; they never alter
the photographed text. Non-English OCR keeps the ink-only page check.

Strips must be upright and belong to one page. Similar letter fragments, damaged
cuts, language mismatches, and missing strips can still produce incorrect
neighbors. The reports flag weak evidence and competing choices for review.

Run the regression checks:

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s backend/tests -v
```

API tests use temporary storage and leave sample submissions unchanged.

Run `node --test tests/frontend-api.test.js` for upload forwarding, progress,
completed-image display, and error-handling checks against an isolated HTTP backend.

The server writes each processed submission into a unique folder under
`backend/img/`. These generated folders and temporary files under `uploads/`
are ignored by Git. Curated photos in `backend/img/error_images/` and
`backend/img/good_images/` can still be committed as regression fixtures.
