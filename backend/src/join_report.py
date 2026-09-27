"""Explain each selected join in a standalone HTML report and matching JSON."""

import base64
from collections import Counter
import html

import cv2
import numpy as np

from submission import write_json


CONFIDENCE_NOTE = (
    "Confidence is a heuristic support score, not the probability that a join is correct. "
    "It combines visual evidence, OCR evidence, separation from competing neighbors, "
    "and agreement among the tested orders. Low scores identify joins worth reviewing."
)


def confidence_for_join(visual, evidence, margin, stability, ocr_enabled):
    quality = (0.5 * visual + 0.5 * evidence["ocr_score"]
               if ocr_enabled else visual)
    separation = float(np.clip(0.5 + (margin or 0) / 0.2, 0, 1))
    amount = min(1.0, evidence["text_lines"] / 12)
    if ocr_enabled:
        amount *= evidence["ink_coverage"]
    value = 100 * (0.40 * quality + 0.25 * separation + 0.20 * stability + 0.15 * amount)
    reasons = []
    if evidence["text_lines"] < 3:
        value = min(value, 39)
        reasons.append("Too few text lines touch this join to judge it reliably.")
    if not ocr_enabled:
        value = min(value, 59)
        reasons.append("OCR was not available; this join has visual evidence only.")
    elif evidence["recognized_lines"] < 3 or evidence["ink_coverage"] < 0.35:
        value = min(value, 49)
        reasons.append("OCR recognized too little of the visible ink around this join.")
    if margin is not None and margin < 0:
        value = min(value, 59)
        reasons.append("Another neighbor has a higher individual pair score; the complete order chose this join.")
    elif margin is not None and margin < 0.02:
        value = min(value, 74)
        reasons.append("A competing neighbor has a similar pair score.")
    if stability < 0.8:
        value = min(value, 74)
        reasons.append("This join changes among the strongest tested orders.")
    if ocr_enabled and evidence["ocr_score"] < 0.4:
        reasons.append("Some ink crossing this join is unreadable or recognized with low OCR confidence.")
    if not reasons:
        reasons.append("Visual and OCR evidence support this join across the tested alternatives.")
    score = round(float(np.clip(value, 0, 100)), 1)
    return {"score": score, "level": "high" if score >= 80 else "medium" if score >= 60 else "low",
            "reasons": reasons}


def build_join_report(order, ranked, scores, verifier, labels, refinement, document_check=None):
    edges = [set(zip(candidate, candidate[1:])) for _, candidate in ranked]
    items = []
    for position, (a, b) in enumerate(zip(order, order[1:]), 1):
        analysis = verifier.evaluate((a, b))
        evidence = analysis["seams"][0]
        successors = sorted((j for j in range(len(labels)) if j not in (a, b) and np.isfinite(scores[a, j])), key=lambda j: -scores[a, j])[:3]
        predecessors = sorted((j for j in range(len(labels)) if j not in (a, b) and np.isfinite(scores[j, b])), key=lambda j: -scores[j, b])[:3]
        competing = [float(scores[a, j]) for j in successors] + [float(scores[j, b]) for j in predecessors]
        margin = float(scores[a, b]) - max(competing) if competing else None
        support = sum((a, b) in candidate for candidate in edges)
        confidence = confidence_for_join(float(verifier.matches.scores[a, b]), evidence, margin,
                                         support / max(1, len(edges)), verifier.ocr.enabled)
        items.append({
            "join": position, "left_position": position, "right_position": position + 1,
            "left_strip": labels[a], "right_strip": labels[b],
            "confidence": confidence,
            "scores": {"combined_pair": float(scores[a, b]), "visual": float(verifier.matches.scores[a, b]),
                       "stroke_continuity": evidence["stroke_score"],
                       "ocr_ink_evidence": evidence["ocr_score"] if verifier.ocr.enabled else None,
                       "english_plausibility": evidence.get("lexical_score"),
                       "mean_ocr_recognition": evidence["ocr_confidence"] if verifier.ocr.enabled else None},
            "text_lines": evidence["text_lines"], "recognized_lines": evidence["recognized_lines"],
            "sampled_text_lines": analysis["sampled_text_lines"],
            "available_text_lines": analysis["available_text_lines"],
            "ink_coverage": evidence["ink_coverage"] if verifier.ocr.enabled else None,
            "margin_over_competing_neighbor": margin, "supporting_orders": support, "tested_top_orders": len(edges),
            "alternative_right_neighbors": [{"strip": labels[j], "score": float(scores[a, j])} for j in successors],
            "alternative_left_neighbors": [{"strip": labels[j], "score": float(scores[j, b])} for j in predecessors],
            "ocr_fragments": evidence["fragments"] if verifier.ocr.enabled else [],
            "line_checks": evidence["lines"] if verifier.ocr.enabled else [],
        })
    document_check = document_check or {"enabled": False}
    if document_check["enabled"]:
        full_page = verifier.evaluate_document(order)
        for item, evidence in zip(items, full_page["seams"]):
            item["full_page_evidence"] = evidence
            item["flagged_by_document_check"] = item["join"] in document_check["suspect_joins"]
    return {
        "schema_version": 3, "confidence_note": CONFIDENCE_NOTE,
        "position_numbering": "Positions count from the left starting at 1; source filenames are unchanged.",
        "evidence_coordinates": "Pair OCR boxes refer to the locally aligned pair analysis; full-page evidence refers to the complete analysis rendering. Neither uses document.png coordinates.",
        "analysis_height": verifier.height,
        "ocr_sampling": "Pair and context OCR sample up to 24 text lines spread over the full strip height; unread ink inside those lines counts against the score. Whole-page OCR and visual matching include every row. OCR analysis height is capped at 2400 pixels.",
        "candidate_selection": "Every directed pair is compared visually and with OCR. All neighbors remain available; context search can move single strips or intact groups. Whole-page OCR compares the finalists with the pairwise baseline.",
        "confidence_formula": "100 * (0.40*quality + 0.25*separation + 0.20*order_support + 0.15*evidence_amount), with evidence-based caps",
        "formula_details": {
            "quality": "0.5*visual + 0.5*OCR ink evidence; visual only when OCR is off",
            "separation": "clamp(0.5 + neighbor_margin/0.2, 0, 1); 0.5 if no competing neighbor",
            "order_support": "fraction of the reported top orders containing the join",
            "evidence_amount": "min(text_lines/12, 1) * ink_coverage; omit coverage if OCR is off",
            "caps": "39 for <3 text lines; 59 without OCR or with a better neighbor; 49 for poor OCR coverage; 74 for close alternatives or unstable order",
        },
        "confidence_levels": {"high": "80–100", "medium": "60–79.9", "low": "0–59.9"},
        "ocr": verifier.ocr.metadata(), "refinement": refinement, "document_check": document_check,
        "summary": dict(Counter(item["confidence"]["level"] for item in items)), "joins": items,
    }


def _evidence_image(verifier, pair, evidence):
    ink, seams = verifier.render(pair)
    lines = evidence["lines"]
    chosen = sorted(lines, key=lambda line: line["ocr_score"])[:3]
    if not chosen:
        return ""
    panels = []
    for line in chosen:
        top, bottom = max(0, line["y_start"] - 5), min(len(ink), line["y_end"] + 5)
        panel = np.rint(255 * (1 - ink[top:bottom])).astype(np.uint8)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.line(panel, (int(seams[0]), 0), (int(seams[0]), panel.shape[0] - 1), (0, 130, 220), 1)
        panels.append(cv2.copyMakeBorder(panel, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    ok, encoded = cv2.imencode(".png", np.vstack(panels))
    return base64.b64encode(encoded).decode() if ok else ""


def write_join_reports(output_dir, report, verifier, order):
    write_json(output_dir / "join_report.json", report)
    esc = lambda value: html.escape(str(value), quote=True)
    fmt = lambda value: "unavailable" if value is None else f"{value:.3f}"
    rows, details = [], []
    for item, pair in zip(report["joins"], zip(order, order[1:])):
        number, confidence = item["join"], item["confidence"]
        level = confidence["level"]
        rows.append(f'<tr><td><a href="#join-{number}">{number} → {number + 1}</a></td>'
                    f'<td>{esc(item["left_strip"])} → {esc(item["right_strip"])}</td>'
                    f'<td class="{level}">{level.title()} · {confidence["score"]}/100</td>'
                    f'<td>{fmt(item["scores"]["visual"])}</td><td>{fmt(item["scores"]["ocr_ink_evidence"])}</td>'
                    f'<td>{fmt(item["margin_over_competing_neighbor"])}</td></tr>')
        evidence = verifier.evaluate(pair)["seams"][0]
        thumbnail = _evidence_image(verifier, pair, evidence)
        fragments = sorted(item["ocr_fragments"], key=lambda word: word["confidence"])[:8]
        fragment_text = ", ".join(f'{esc(word["text"])} ({word["confidence"]:.2f})' for word in fragments) or "No recognized fragments crossing this join."
        alternatives = "; ".join(f'{esc(candidate["strip"])} ({candidate["score"]:.3f})' for candidate in item["alternative_right_neighbors"]) or "No other right neighbor."
        reasons = "".join(f'<li>{esc(reason)}</li>' for reason in confidence["reasons"])
        image = f'<img class="evidence" src="data:image/png;base64,{thumbnail}" alt="Three weakest text regions; orange marks the cut">' if thumbnail else ""
        page_evidence = item.get("full_page_evidence")
        page_note = (f'<p><strong>Full-page OCR evidence:</strong> {page_evidence["ocr_score"]:.3f}; '
                     f'ink coverage {page_evidence["ink_coverage"]:.3f}. '
                     f'{"Flagged for review." if item["flagged_by_document_check"] else "No full-page flag."}</p>'
                     if page_evidence is not None else "")
        details.append(f'<section id="join-{number}"><h2>Join {number}: positions {number} → {number + 1}</h2>'
                       f'<p>{esc(item["left_strip"])} → {esc(item["right_strip"])} · '
                       f'<strong class="{level}">{level.title()} support, {confidence["score"]}/100</strong></p>'
                       f'<ul>{reasons}</ul>{image}{page_note}'
                       f'<p>Stroke continuity: {fmt(item["scores"]["stroke_continuity"])}. '
                       f'OCR recognition: {fmt(item["scores"]["mean_ocr_recognition"])}. Ink coverage: {fmt(item["ink_coverage"])}. '
                       f'Recognized lines: {item["recognized_lines"]}/{item["text_lines"]}. '
                       f'Text regions sampled: {item["sampled_text_lines"]}/{item["available_text_lines"]}. '
                       f'Present in {item["supporting_orders"]}/{item["tested_top_orders"]} top tested orders.</p>'
                       f'<p><strong>Weak OCR fragments:</strong> {fragment_text}</p>'
                       f'<p><strong>Other right-neighbor candidates:</strong> {alternatives}</p></section>')
    counts = ", ".join(f"{count} {level}" for level, count in sorted(report["summary"].items())) or "No joins (one text strip)."
    check = report["document_check"]
    document_section = ""
    if check["enabled"]:
        suspects = ", ".join(f'<a href="#join-{i}">{i} → {i + 1}</a>' for i in check["suspect_joins"]) or "None flagged."
        document_section = (f'<section><h2>Complete-page OCR check</h2>'
                            f'<p>Readability evidence: {check["baseline_score"]:.3f} before context correction → '
                            f'{check["final_score"]:.3f} in the selected document. Compared {check["evaluated_orders"]} complete orders.</p>'
                            f'<p>{esc(check["score_note"])} Analysis height: {check["final_reading"]["analysis_height"]} pixels; the saved document keeps its original resolution.</p>'
                            f'<p>Joins to review: {suspects}</p><details><summary>Text recognized from the final page</summary>'
                            f'<p>{esc(check["final_reading"]["text"])}</p></details></section>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strip join verification</title><style>
body{{font:16px/1.5 system-ui,sans-serif;color:#182230;background:#f5f7fa;margin:0;padding:32px;max-width:1180px;margin:auto}}
h1,h2{{line-height:1.2}}a{{color:#174ea6}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dde2e8}}
th{{background:#e9edf3}}.high{{color:#17613b}}.medium{{color:#855900}}.low{{color:#a52b24}}section{{background:white;padding:20px;margin:22px 0;border:1px solid #dde2e8;border-radius:8px}}
.evidence{{float:right;max-width:42%;width:260px;margin:0 0 12px 20px;image-rendering:auto}}section::after{{content:"";display:block;clear:both}}
.note{{background:#fff5d7;padding:16px;border-left:4px solid #ba8700}}.table{{overflow-x:auto}}@media(max-width:650px){{body{{padding:16px}}.evidence{{float:none;max-width:100%;margin:0}}}}
</style><h1>Strip join verification</h1><p>{esc(counts)}. <a href="document.png">Open assembled document</a> · <a href="join_report.json">Full JSON evidence</a></p>
<p class="note">{esc(CONFIDENCE_NOTE)}</p><p>{esc(report["position_numbering"])}</p>
<p>OCR: {esc(report["ocr"]["status"])} ({esc(report["ocr"].get("version") or report["ocr"].get("reason"))}).
Dictionary correction is disabled. OCR evidence includes unread ink; raw recognition confidence alone is not a join confidence.</p>
<p>{esc(report["ocr_sampling"])} {esc(report["candidate_selection"])}</p>
{document_section}
<p>Orange lines mark the cut in analysis crops. Crops show the weakest tested text regions; they suggest possible problems, not confirmed errors.</p>
<div class="table"><table><thead><tr><th>Positions</th><th>Source strips</th><th>Confidence</th><th>Visual</th><th>OCR evidence</th><th>Neighbor margin</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
{"".join(details)}<section><h2>How the search stopped</h2><p>{esc(report["refinement"]["stop_reason"])}</p>
<p>The context search is bounded and does not prove that the document is correct. Detailed scoring rules and all recognized fragments are in the JSON report.</p></section></html>'''
    (output_dir / "join_report.html").write_text(page, encoding="utf-8")
