"""Shared resources and the single result tree for the attention experiment."""
from result_layout import relocate_path, write_result_text
from pathlib import Path

ROOT = relocate_path(Path(__file__)).resolve().parent.parent
RESULTS = relocate_path(ROOT / 'results')
QUERY = relocate_path(RESULTS / 'query')
GENERATION = relocate_path(RESULTS / 'generation')
HIDDEN = relocate_path(RESULTS / 'hidden')
DYNAMICS = relocate_path(RESULTS / 'dynamics')
SPATIAL = relocate_path(RESULTS / 'spatial')
