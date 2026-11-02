from result_layout import relocate_path, write_result_text, result_glob
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Validate final artifacts, original source preservation and nine-sample continuity."""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

sys.path.insert(0, str(relocate_path(ROOT / "python_deps")))
import numpy as np
from PIL import Image
from pypdf import PdfReader


def main():
    results = QUERY
    samples = json.loads((relocate_path(results / "samples.json")).read_text())
    assert len(samples) == 300
    for sample in samples:
        folder = relocate_path(results / sample["dataset"] / sample["sample_key"])
        for name in ["attention.npz", "temporal_attention.csv", "metadata.json", "validation.json", "expanded_input.txt", "PLOTTED"]:
            assert (relocate_path(folder / name)).stat().st_size > 0
        for name in ["attention.png", "attention_mtla_time.png"]:
            with Image.open(relocate_path(folder / name)) as image:
                image.verify()
        assert len(PdfReader(relocate_path(folder / "attention.pdf")).pages) == 1
    counts = {}
    for dataset in dict.fromkeys(s["dataset"] for s in samples):
        for suffix, n in [("detail", 100), ("overview", 12)]:
            path = relocate_path(results / f"{dataset}_{suffix}.pdf")
            counts[path.name] = len(PdfReader(path).pages)
            assert counts[path.name] == n
    counts["attention_report.pdf"] = len(PdfReader(relocate_path(results / "attention_report.pdf")).pages)
    assert counts["attention_report.pdf"] == 300
    for name in ["summary", "relative_time_summary", "representative_examples"]:
        with Image.open(relocate_path(results / f"{name}.png")) as image:
            image.verify()
        assert len(PdfReader(relocate_path(results / f"{name}.pdf")).pages) == 1
    for path in result_glob((relocate_path(results / "overview_pages")), "*.png"):
        with Image.open(path) as image:
            image.verify()

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"href", "src"} and value and not value.startswith(("#", "http")):
                    assert (relocate_path(results / value)).is_file(), value
    page = (relocate_path(results / "report.html")).read_text()
    Links().feed(page)
    assert page.count("<article ") == 300

    continuity = json.loads((relocate_path(ROOT/'provenance/migration.json')).read_text())['pilot_absorbed']
    assert len(continuity)==9 and all(x['all_arrays_bitwise_identical'] for x in continuity)
    original = relocate_path(Path("/home/irteam/vtg_works/MTLA"))
    source_hashes = json.loads((relocate_path(ROOT / "source_sha256.json")).read_text())
    current = {str(p.relative_to(original)): hashlib.sha256(p.read_bytes()).hexdigest() for p in original.rglob("*") if p.is_file()}
    assert source_hashes == current
    verified = dict(complete_samples=300, html_cards=300, all_pngs_valid=True,
                    pdf_page_counts=counts, previous_nine_sample_continuity=continuity,
                    pilot_arrays_verified_at_consolidation=True, original_source_unchanged=True)
    write_result_text((relocate_path(results / "artifact_validation.json")), json.dumps(verified, indent=2) + "\n")
    paths = list(result_glob(ROOT, "*.py")) + list(result_glob(ROOT, "*.md"))
    paths += [p for p in results.rglob("*") if p.is_file() and p.name != "artifact_sha256.json"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_result_text((relocate_path(QUERY / 'artifact_sha256.json')), json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(verified, indent=2), flush=True)


if __name__ == "__main__":
    main()
