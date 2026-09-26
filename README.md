# ShreddedMan

Install dependencies in your virtual environment:

```sh
python -m pip install -r requirements.txt
```

Create a submission from one or more photos of **one document**:

```sh
python src/main.py /path/to/photo1.jpg /path/to/photo2.jpg
```

The command prints the new submission path and result path. Each call creates a
new UUID folder under `img/`; `--img-root /another/location` changes that parent.
Uploaded files are copied without modifying the originals. Duplicate upload
names are given distinct stored names.

```text
img/
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
    submission.json
    order.json
```

Detection includes every JPG/JPEG, PNG, BMP, TIFF, or WebP file directly inside
`source_images`, with case-insensitive extensions. Crops from all photos share
one consecutive strip-number sequence. Normalization preserves these filenames;
sorting reads the entire pooled directory and writes only the assembled image
to `final_document`. It never renames or overwrites the normalized strip images.

`submission.json` records upload names, each strip's source photo, rotation,
and processing status. `order.json` contains estimated positions, source
mappings, all pair scores, alternative orders, and ambiguous joins. Stored
paths are relative to the submission so it can be moved as a unit. Blank/low-ink
strips stay in the crop and normalized folders, with unresolved positions in the
report; they are excluded from the final image for now.

Rerun a submission after adding or removing source photos:

```sh
python src/main.py --submission img/<submission-id>
```

Run individual stages for an existing submission:

```sh
python src/detect_paper.py img/<submission-id>
python src/normalize_strips.py img/<submission-id>
python src/sort_strips.py img/<submission-id>
```

Detection invalidates the old normalized strips and assembled result.
Normalization invalidates the old assembled result. Reprocessing removes stale
numbered strips when the new detection contains fewer pieces. Auto rotation
turns horizontal strips 90 degrees counterclockwise; use `--rotation 0`, `90`,
`180`, or `270` on the full pipeline or detection command when needed.

For application integration, add `src` to the Python import path and call:

```python
from pathlib import Path
from assemble_document import assemble_document, process_submission
from submission import create_submission

# Copy uploaded files and run the full pipeline synchronously.
result = assemble_document([Path("/uploads/part1.jpg"), Path("/uploads/part2.jpg")])
result_path = Path(result["submission_dir"]) / result["result"]

# Or create storage now and run processing later in a worker.
submission = create_submission([Path("/uploads/part1.jpg")])
result = process_submission(submission.directory)
```

Separate submissions can be processed independently. A single submission
should have one processing job at a time. The Python pipeline is synchronous;
the HTTP API below schedules it in a background worker. The existing frontend
is unchanged and is not connected to the API yet.

## Backend API

From the repository root, with your virtual environment activated:

```sh
python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/docs to try uploads and view the full API reference.

| Method | Endpoint | Result |
| --- | --- | --- |
| POST | `/api/submissions` | Upload photos; receive a submission ID and status URL (HTTP 202). |
| GET | `/api/submissions/{id}` | Check progress and get the result URL or error. |
| GET | `/api/submissions/{id}/document` | Retrieve the completed PNG. |

Send photos of one document in the multipart field `files`, with optional
`rotation` (`auto`, `0`, `90`, `180`, or `270`). Poll the returned `status_url`
until `complete` or `failed`, then use `document_url` or display `error`.

Files are stored under `img/`; set `SHREDDEDMAN_IMG_ROOT` to change the location.
This local-development API uses one background worker and an in-memory queue;
unfinished jobs must be resubmitted after a forced stop. The frontend is not connected yet.

## Reconstruction and tests

Sorting compares all ordered pairs using full-height edge ink and text-row
alignment, with vertical scale/offset search and small smooth local corrections.
The complete order maximizes the sum of adjacent-pair scores. Exhaustive search
handles up to eight text strips; larger samples use an integer optimizer.
Strips must be upright and belong to one page. Similar letter fragments, damaged
cuts, and missing strips can still produce incorrect neighbors. Scores and
ambiguity flags are not calibrated confidence estimates.

Run the regression checks:

```sh
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

API tests use temporary storage and leave sample submissions unchanged.

The existing sample images have been migrated into these submissions:

| Sample | Submission folder |
| --- | --- |
| photo1 | [0ebcf760ad274996badf236a20836806](img/0ebcf760ad274996badf236a20836806) |
| photo2 | [5f79ef7265914cc4bac385a2a5c68b60](img/5f79ef7265914cc4bac385a2a5c68b60) |
| photo3_1 + photo3_2 | [b5e8deb5e6d64ab4b25f279e96d401e7](img/b5e8deb5e6d64ab4b25f279e96d401e7) |

Each migrated sample also has `previous_results.zip` at its root, preserving
earlier comparison images and reports that are not part of the current output.
