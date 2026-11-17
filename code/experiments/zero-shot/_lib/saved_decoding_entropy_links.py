"""Link entropy appendices to their canonical experiment and query scope."""
import os
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ('deterministic', '900', '다른 decoding 경로 · deterministic pilot 900'),
    ('oracle_switch_controls', '60', '고정 prefix 영상 perturbation · pilot 60'),
)


def links_from(directory):
    links = []
    for family, count, label in SOURCES:
        base = ROOT / 'results/rollout' / family
        entry = f'pilot/{count}/APPENDIX_ENTROPY.md'
        if (base / entry).exists():
            href = os.path.relpath(base / 'viewer.html', directory) + '?view=' + quote(entry, safe='')
            links.append(f'[{label}]({href})')
    return links


def markdown_from(directory):
    links = links_from(directory)
    if not links:
        return []
    return ['## 다른 저장된 decoding 자료에서의 확인', '',
            '별도 실험에 저장된 분기 경로와 영상 perturbation probe도 재집계했다. '
            '아래 appendix는 원래 실험 계열과 실제 query 수 아래 보존하며 이 보고서의 표본과 합산하지 않는다.', '',
            *['- ' + link for link in links], '',
            'deterministic 분기에서는 강제로 선택한 token과 이후 자유 생성 step을 구분한다. '
            '영상 probe는 동일 prefix의 비교지만 GT 후보를 겨냥한 입력 perturbation이므로 자연스러운 증거 추가·제거와 구분한다. '
            'probe의 entropy는 원본에서 고정한 30개 후보를 사용하므로 각 조건의 native top-30 entropy와 같은 값으로 섞지 않는다.', '']
