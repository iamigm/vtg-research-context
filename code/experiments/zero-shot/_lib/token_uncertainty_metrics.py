"""Report metrics for the conditional distribution on saved top-30 tokens."""
import math

ENTROPY_BASIS = 'top30_renormalized'
DEFINITION = ('이 보고서의 entropy와 후보 확률은 top-30 재정규화 기준이다. '
    '각 step에서 M = Σ(top-30) pᵢ, qᵢ = pᵢ/M, H₃₀ = −Σ qᵢ ln qᵢ로 계산한다. '
    '단위는 nat, 범위는 0–ln(30) ≈ 3.401197이며 높을수록 top-30 내부 분포가 고르다. '
    'raw와 processed는 각각 자신의 top-30을 사용한다. top-30 밖의 질량은 조건부 분포에서 제외한다. '
    '정답 가능성은 기존 문법·반올림·IoU 판정을 뜻하며 모델의 최종 정답 확률과 다르다. '
    '원본 응답·판정·전체 vocabulary 분포 계측값은 보존하고 보고서의 파생 지표만 재계산했다.')


def top30_distribution(dist):
    entries = dist['top30']
    mass = math.fsum(t['probability'] for t in entries)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('Invalid top-30 mass')
    if any(not math.isfinite(t['probability']) or t['probability'] < 0 for t in entries):
        raise ValueError('Invalid token probability')
    return mass, [dict(t, probability=t['probability']/mass) for t in entries]


def entropy_top30(dist):
    _, entries = top30_distribution(dist)
    return -math.fsum(t['probability'] * math.log(t['probability'])
                      for t in entries if t['probability'] > 0)


def selected_top30(step, mode='raw'):
    _, entries = top30_distribution(step[mode])
    return next((t['probability'] for t in entries if t['id'] == step['token_id']), 0.)


def report_definition(lines):
    """Keep formula and interpretation in every independently regenerated report."""
    return [lines[0], '', DEFINITION, ''] + lines[1:]
