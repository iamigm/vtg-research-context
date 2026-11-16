"""Reader-facing explanations derived from the saved within-path statistics."""


def explain(lines, appendix, scope, paths, paired, regression, variances):
    groups = [r for r in paired if r['endpoint'] != 'all']
    comparable = [r for r in groups if r['delta_previous']['mean'] is not None]
    adjusted = [r for r in regression if r['kind'] == 'endpoint_role_position' and r['beta'] is not None]
    endpoints = [r for r in regression if r['kind'] == 'endpoint' and r['beta'] is not None]
    positive = sum(r['delta_previous']['ci95'][0] > 0 for r in comparable)
    adjusted_positive = sum(r['ci95'][0] is not None and r['ci95'][0] > 0 for r in adjusted)
    counts = {key: sum(r[key] for r in groups) for key in (
        'endpoint_paths', 'paired_paths', 'no_error_paths', 'format_error_paths',
        'numeric_error_without_predecessor')}
    first = comparable[0]
    mu = first['delta_previous']['mean']
    lo, hi = first['delta_previous']['ci95']
    mode, count = scope.split('/')
    responses = len({(r['model'], r['sample_key']) for r in paths})
    definition = lines[2].replace('기존 문법·반올림·IoU 판정', '이 분석의 허용 문법·반올림 endpoint 일치 판정')

    # Keep all existing detailed regression values and their method in the appendix.
    fe_start = lines.index('## 경로별 기준 수준과 생성 위치를 통제한 비교')
    fe_end = lines.index('## 해석과 한계')
    fe_detail = lines[fe_start:fe_end]
    appendix += ['', *fe_detail]
    low = min(r['delta_previous']['mean'] for r in comparable)
    high = max(r['delta_previous']['mean'] for r in comparable)
    main = [lines[0], '', '## 먼저 읽을 결론', '',
        f'**같은 endpoint 경로 안에서도 최초 오류 step의 entropy가 직전 정답 가능 숫자보다 평균적으로 높았다.** '
        f'{mode} {int(count):,} queries에서 model × dataset × endpoint의 {positive}/{len(comparable)}개 그룹은 '
        f'평균 차이의 95% CI가 모두 0보다 컸다. 그룹별 평균 차이는 {low:.3f}–{high:.3f} nat이다. '
        '이는 그룹 평균의 결과이며 모든 개별 경로에서 entropy가 상승했다는 뜻은 아니다.', '',
        '**자릿수와 생성 순서도 함께 고려해야 한다.** '
        f'endpoint별 기준 수준만 통제한 차이 β는 {min(r["beta"] for r in endpoints):.3f}–{max(r["beta"] for r in endpoints):.3f} nat이고, '
        f'role과 numeric 순서를 더 통제한 β는 {min(r["beta"] for r in adjusted):.3f}–{max(r["beta"] for r in adjusted):.3f} nat이다. '
        f'후자의 CI가 0보다 큰 그룹은 {adjusted_positive}/{len(adjusted)}개다. '
        '처음 보이는 큰 차이에는 생성 위치와 관련된 차이도 상당히 섞여 있음을 시사한다. '
        '두 β의 차이를 위치가 원인인 비율로 해석할 수는 없다.', '',
        (f'{mode}에서는 위치를 통제한 뒤에도 {len(adjusted)}개 그룹 모두 양의 차이가 남는다.' if adjusted_positive == len(adjusted) else
         '이 pilot에서는 위치 통제 후 일부 그룹의 CI가 0을 포함한다. 해당 그룹에서 차이가 없다고 확정하거나, 음의 추정값만 보고 방향이 반대라고 단정하지 않는다.'), '',
        '## 왜 한 경로 안에서 비교하나', '',
        '어려운 query는 처음부터 여러 step의 entropy가 높을 수 있다. 서로 다른 query의 정답 step과 오류 step을 바로 비교하면, '
        '원래 어려운 sample의 영향과 오류가 나는 순간의 변화가 섞인다. 이번에는 **같은 모델이 같은 query에 답하는 같은 endpoint 경로 안에서** '
        '이전 숫자와 최초 오류 숫자를 비교한다. start는 시작 시각, end는 종료 시각이며 각각 별도로 계산한다.', '',
        '예를 들어 쉬운 sample의 두 step이 0.2 → 0.7 nat이고 어려운 sample은 1.2 → 1.7 nat이라면, '
        '절대값은 다르지만 둘 다 경로 내부 변화는 +0.5 nat이다. 이처럼 차이를 구하면 경로 전체에 일정하게 더해진 기준 수준은 상쇄된다. '
        '이 숫자는 원리 설명을 위한 가상 예시다. 실제로는 prefix와 후보가 계속 바뀌므로 sample에서 비롯된 모든 불확실성이 상쇄되는 것은 아니다.', '',
        '## entropy와 정답 가능성은 각각 무엇인가', '',
        'entropy는 **그 step에서 다음 token의 top-30 후보 사이에 확률이 얼마나 퍼져 있는가**를 나타낸다. '
        '후보 하나에 거의 몰리면 0에 가깝고, 30개가 똑같으면 ln(30) ≈ 3.401 nat이다. '
        '값이 크면 후보 사이의 불확실성이 크다는 뜻이며, 그 자체가 오답 확률이나 모델의 최종 정답 확률은 아니다.', '',
        '먼저 top-30 원래 확률의 합 M을 구하고, 각 확률을 M으로 나누어 합이 1인 q로 만든다. '
        '그 q로 H₃₀를 계산한다. 따라서 top-30 밖에 남은 probability mass는 H₃₀에 반영되지 않는다. '
        '본문의 Δ와 β는 raw 분포 기준이며 processed 결과는 appendix의 보조 확인이다.', '',
        definition, '',
        '이 분석의 **정답 가능성은 반올림 endpoint 일치 기준**이다. 지금까지 쓴 prefix 뒤에 허용된 숫자·서식을 붙여 '
        '반올림한 시각이 GT와 같은 답을 아직 만들 수 있는지를 판정한다. 허용 표기는 음이 아닌 초, 정수 또는 소수 한 자리, '
        '불필요한 선행 0 없음이다. `round(float(value))`를 사용하며 정확히 .5는 가까운 짝수로 반올림한다. '
        'start와 end를 각각 판정하고, 전체 구간의 IoU 성공 여부와는 구분한다.', '',
        'GT가 24초라면 `23.9`는 반올림해서 정답이다. `23.4`의 prefix `23.`까지는 아직 9를 붙여 정답으로 만들 수 있지만, '
        '숫자 4를 선택하면 그 가능성이 처음 사라진다. 여기서 4가 first error다. '
        '따라서 파란색은 GT 문자열과 같은 숫자라는 뜻이 아니라 **정답으로 완성할 가능성을 유지했다**는 뜻이다.', '',
        '## 한 경로의 계산을 예시로 따라가기', '',
        '아래는 GT 24초에 대해 `23.4`를 출력한 경로에 가상의 entropy를 붙인 설명용 예시다. 실제 관측 사례나 평균값이 아니다.', '',
        '| numeric 순서 | 선택 token | 선택 후 상태 | 선택할 때의 H₃₀ (nat) |',
        '|---|---|---|---:|',
        '| 1 | `2` | 정답 가능성 유지 | 0.20 |',
        '| 2 | `3` | 정답 가능성 유지 | 0.40 |',
        '| 3 | `4` | 최초 오류 | 1.10 |', '',
        '중간의 소수점은 numeric step 비교에서 건너뛴다. Δprevious = 1.10 − 0.40 = **+0.70 nat**이고, '
        'Δprior = 1.10 − (0.20 + 0.40)/2 = **+0.80 nat**이다. '
        'Δprevious는 가장 가까운 이전 숫자와, Δprior는 이전 정답 가능 숫자 전체의 평균과 비교한다. '
        '양수면 최초 오류 때 더 높았고, 음수면 더 낮았고, 0이면 같았다는 뜻이다. +0.70 nat을 오류 확률 70% 증가로 읽으면 안 된다.', '',
        '오류 이후에는 이미 잘못된 prefix를 조건으로 다음 token이 생성된다. 그 이후 숫자까지 주 비교에 섞으면 '
        '최초 가능성 상실과 이후 전개의 효과를 구분하기 어려워, 주 비교는 최초 오류까지로 제한한다.', '',
        '## 어떤 경로가 비교에 들어갔나', '',
        f'{mode}의 {int(count):,}개 query × 두 모델 중 지원되는 표기의 {responses:,}개 응답, '
        f'{counts["endpoint_paths"]:,}개 endpoint 경로를 분석했다. '
        f'지원 표기 분석 대상에서 빠진 응답은 {2 * int(count) - responses:,}개다. '
        'query 수, 모델별 응답 수, start/end 경로 수는 서로 다른 단위다.', '',
        '| endpoint 경로의 상태 | 경로 수 | paired 비교에 포함하는가 |',
        '|---|---:|---|',
        f'| numeric 최초 오류와 이전 정답 가능 숫자가 모두 있음 | {counts["paired_paths"]:,} | 포함 |',
        f'| 오류 없음 | {counts["no_error_paths"]:,} | 오류 step이 없어 제외 |',
        f'| 첫 numeric step부터 오류 | {counts["numeric_error_without_predecessor"]:,} | 이전 비교 숫자가 없어 제외 |',
        f'| 소수점·종료 등 서식에서 최초 오류 | {counts["format_error_paths"]:,} | numeric 오류 비교가 아니므로 제외 |', '',
        f'paired 비교는 전체 endpoint 경로 중 {100 * counts["paired_paths"] / counts["endpoint_paths"]:.1f}%에 해당한다. '
        '이 합계는 두 모델과 start/end를 합한 경로 수이며 고유 query 수가 아니다. '
        '따라서 paired 결과를 오류가 없는 경로나 첫 숫자부터 틀린 경로까지 그대로 일반화하지 않는다. '
        '회귀 분석은 아래에서 설명하는 risk set을 사용하므로 paired 표의 분모와 다르다.', '',
        '## paired 결과 표와 그래프 읽기', '',
        '각 행은 모델 하나 × dataset 하나 × endpoint 하나다. 각 경로에서 Δ를 먼저 구한 다음, '
        '해당 행에 포함된 query의 Δ를 같은 가중치로 평균한다. N은 그 행에서 실제 paired 비교가 가능한 query 수다. '
        'start와 end의 N을 합해 고유 query 수로 해석하지 않는다.', '',
        f'예를 들어 {first["model"]} / {first["dataset"]} / {first["endpoint"]}의 '
        f'Δprevious {mu:.4f} [{lo:.4f}, {hi:.4f}]는 '
        f'{first["delta_previous"]["queries"]:,}개 query의 경로 내부 차이를 평균했을 때 '
        f'최초 오류 step이 직전 숫자보다 {mu:.4f} nat 높았다는 뜻이다. '
        '괄호는 개별 경로 95%가 놓이는 범위가 아니라 **평균 차이 추정의 불확실성**을 나타내는 95% CI다.', '',
        'CI는 query를 복원추출하여 평균을 다시 계산하는 bootstrap 2,000회(seed 42)로 구했다. '
        'CI 전체가 0 위에 있으면 이 방식으로 추정한 평균 차이는 양의 방향이다. '
        '0을 포함하면 이 분석만으로 방향을 뚜렷하게 구분하기 어렵다. '
        '그래프의 점은 평균, 세로 막대는 CI, 가로 0선은 차이가 없는 기준이다.', '']
    paired_start = lines.index('## 같은 endpoint 경로의 paired 결과')
    main += lines[paired_start:fe_start]
    main += ['## 자릿수와 생성 순서를 고려하면 무엇이 달라지나', '',
        '오류는 경로의 뒤쪽에서 발생하고, 뒤쪽 숫자는 소수 첫째 자리 등 앞쪽과 다른 역할을 맡을 수 있다. '
        '같은 경로 안에서 비교해도 자릿수나 생성 위치가 원래 가지는 entropy 차이가 남는다. '
        '이를 살펴보기 위해 다음 네 모형을 같은 risk set에서 비교했다. risk set은 **선택 직전까지 정답 가능했던 numeric step**으로, '
        '가능성 유지 step과 최초 numeric 오류 step을 포함하고 오류 이후는 제외한다.', '',
        '| 모형 | 쉽게 풀어 쓴 비교 조건 |',
        '|---|---|',
        '| pooled | 서로 다른 query도 함께 비교하며 start/end 차이만 통제 |',
        '| response FE | 같은 model × query 응답의 기준 수준을 빼고 start/end 차이도 통제 |',
        '| endpoint FE | 같은 model × query × start 또는 end 경로의 기준 수준을 빼고 비교 |',
        '| endpoint + role + position | endpoint FE에 출력 자릿수 role과 numeric 순서 차이를 추가 통제 |', '',
        'FE는 fixed effects다. 예를 들어 H에서 같은 경로의 risk step 평균을 빼면, 경로가 전체적으로 높거나 낮은 수준 차이를 제거할 수 있다. '
        '회귀에서는 오류 indicator와 통제 변수에도 같은 경로 내부 변환을 적용한다. '
        'β는 이런 비교 조건에서 최초 오류 step과 가능성 유지 step 사이에 추정된 entropy 차이다. '
        'β > 0이면 오류 step 쪽이 더 높고, 단위는 nat이다.', '',
        '| 모델 | dataset | endpoint FE β (nat) | role·position 추가 β (nat) [95% CI] |',
        '|---|---|---:|---:|']
    for r in adjusted:
        base = next(x for x in endpoints if (x['model'], x['dataset']) == (r['model'], r['dataset']))
        main.append(f'| {r["model"]} | {r["dataset"]} | {base["beta"]:.4f} | {r["beta"]:.4f} [{r["ci95"][0]:.4f}, {r["ci95"][1]:.4f}] |')
    main += ['', '추가 통제 후 값이 작아졌다는 것은 큰 경로 내부 차이의 일부가 자릿수·생성 순서와 함께 움직였음을 보여준다. '
        '같은 query에서 동일 위치의 정답·오답 두 경로를 관측한 실험은 아니다. '
        'role과 순서의 효과는 다른 경로의 자료도 함께 사용해 추정하며, 통제 후 β도 인과 효과로 해석하지 않는다. '
        'pooled → FE 비교는 가중치도 달라지므로 계수 차이를 sample 불확실성의 제거량이라고 읽으면 안 된다. '
        '모든 모형의 수치, 계산식, 가중치와 CI 계산법은 [appendix](APPENDIX.md)에 보존했다.', '',
        '## sample 자체의 불확실성에 대해 어디까지 말할 수 있나', '',
        f'보조 분산 분해에서 응답 간 평균 차이가 전체 entropy 분산에서 차지하는 비중은 '
        f'모델·dataset별 {100 * min(r["between_fraction"] for r in variances):.1f}–{100 * max(r["between_fraction"] for r in variances):.1f}%다. '
        '응답마다 기준 수준이 다르다는 점을 보여주지만, 이 비중에는 sample 난도 외에 출력 숫자·길이·prefix 구성 차이도 섞여 있다. '
        '따라서 이를 sample 고유 불확실성의 비율이라고 부르지는 않는다.', '',
        '이번 결과는 **sample 사이의 일정한 수준 차이만으로 평균적인 오류 step의 상승을 전부 설명하기는 어렵다**는 근거다. '
        '다만 경로 내부 비교는 sample 불확실성을 완전히 분리하는 실험이 아니다. '
        '각 query에는 모델별 저장된 greedy 경로가 한 개뿐이므로 같은 입력의 여러 stochastic rollout 사이 변동이나 '
        '본질적인 sample 불확실성을 따로 추정한 결과는 아니다.', '',
        '또한 이미 관측된 최초 오류를 기준으로 비교한 사후 분석이다. entropy가 높아지면 앞으로 틀릴 것이라는 '
        '온라인 예측 성능이나 임계값을 검증한 결과는 아니다. 여러 query가 같은 영상을 공유할 수 있으며 '
        'CI는 video 단위가 아닌 query 단위로 계산했다. 여러 비교에 대한 유의성 보정도 하지 않은 탐색적 결과다. '
        'pilot과 full은 별도로 읽고 독립 실험처럼 합산하지 않는다.', '',
        '## viewer에서 직접 확인하는 방법', '',
        '대표 viewer에서 full 또는 pilot을 고르고 `결과` 탭을 연다. 좌측에서 모델·dataset·endpoint·sample을 선택한다. '
        '기본 필터는 오류 전후 비교가 가능한 경로다. 빠른 필터와 정량 범위를 함께 적용할 수 있으며, '
        '`전체`를 고르면 오류가 없거나 이전 숫자가 없는 경로도 볼 수 있다.', '',
        '메인 곡선은 응답 전체 decoding step의 raw top-30 entropy를 생성 순서대로 표시한다. '
        'start/end 영역과 최초 오류를 표시하고, 표시 범위를 선택한 endpoint로 좁힐 수도 있다. '
        '점이나 token을 누르면 선택 전·후 prefix와 아래 token distribution이 함께 바뀐다. '
        '곡선 위의 네 수치는 선택한 endpoint의 직전 숫자 entropy, 최초 오류 entropy, Δprevious, Δprior다. '
        '값이 없는 경로에는 —를 표시하며 0으로 대체하지 않는다.', '',
        '아래 token distribution은 선택한 step의 보조 자료다. raw와 processed는 각각 자신의 top-30으로 재정규화한다. '
        '정답 가능성 후보 판정은 기존 raw 분석이 있는 step에만 표시하고, 미계산을 정답 불가로 보지 않는다. '
        'GT 고정 IoU와 다른 sample의 같은 자릿수 비교는 보조 분석을 펼쳐 확인한다.', '',
        '[경로별 interactive 비교](report.html) · [제외 상세표·분산 분해·processed 결과·회귀 계산법](APPENDIX.md)', '',
        lines[-1]]
    appendix[0] = f'# Within-path 상세 결과 · {mode} · {int(count):,} queries'
    appendix[2] = definition
    appendix[2:2] = ['[쉬운 해설과 핵심 결과로 돌아가기](REPORT.md)', '', '## paired 비교 포함·제외 상세', '']
    variance_heading = appendix.index('## 경로 간·경로 내부 분산')
    appendix[variance_heading + 2:variance_heading + 2] = [
        '여기서 경로 간 분산의 묶음은 endpoint가 아니라 **model × query 응답**이다. '
        '같은 응답의 start/end risk numeric step을 합쳐 평균과 분산을 구한다. '
        'between은 응답 평균들이 서로 다른 정도, within은 한 응답 안에서 step들이 다른 정도다. '
        '각 응답에 같은 가중치를 주며, 이는 모든 step을 같은 가중치로 둔 분산과 다를 수 있다.', '']
    rank_heading = appendix.index('## Processed 및 경로 내부 순위')
    appendix[rank_heading + 2:rank_heading + 2] = [
        'processed Δprevious는 저장된 processed 분포에서 별도로 top-30을 재정규화한 값이다. '
        'raw와 후보 집합이 같다고 가정하지 않는다. rank는 이전 숫자들과의 비교 비율이며 '
        '정답 확률이나 오류 탐지 정확도가 아니다. 예를 들어 이전 숫자 세 개보다 오류 entropy가 각각 '
        '높음·같음·낮음이면 rank = (1 + 0.5 + 0)/3 = 0.5다. '
        '`all` 행은 start/end에서 얻은 값을 먼저 query 안에서 평균한 뒤 query 사이에서 평균한다.', '']
    # Each visual carries its own explanation immediately below it.
    table_notes = {
        '| numeric 순서': '위 표는 계산 원리를 보여주는 가상 경로다. 파란 상태인 `2`, `3` 다음에 `4`에서 처음 정답 가능성이 사라진다. '
            '직전 숫자 0.40과 오류 숫자 1.10을 빼면 +0.70 nat이다. token이 GT와 같은지보다, prefix를 정답으로 완성할 수 있는지를 본다.',
        '| endpoint 경로의 상태': f'위 표의 네 행은 전체 {counts["endpoint_paths"]:,}개 endpoint 경로를 나눈 것이다. '
            f'실제 전후 비교는 첫 행의 {counts["paired_paths"]:,}개({100 * counts["paired_paths"] / counts["endpoint_paths"]:.1f}%)에서만 가능하다. '
            '나머지는 오류가 없거나, 앞 숫자가 없거나, 오류가 서식에 생겨 동일한 두 숫자 비교를 만들 수 없다. 제외는 낮은 entropy를 골라 버렸다는 뜻이 아니다.',
        '| 모델 | dataset | endpoint | N paired query': f'위 표에서 Δprevious 열부터 보면 된다. 예를 들어 첫 행의 {mu:.4f}는 '
            f'{first["delta_previous"]["queries"]:,}개 query 각각에서 “최초 오류 − 직전 숫자”를 계산한 뒤 평균한 값이다. '
            f'{positive}/{len(comparable)}개 행에서 CI 전체가 0보다 크므로 평균적으로 오류 때 더 높았다. '
            '모든 sample이 상승했다는 뜻은 아니다. Δprior 열은 직전 숫자 하나 대신 이전 정답 가능 숫자 전체의 평균을 뺀 결과다.',
        '| 모형 |': '위 표는 숫자 결과가 아니라 비교 방법의 차이다. 위에서 아래로 내려갈수록 응답의 기준 수준, '
            'start/end의 기준 수준, 자릿수와 생성 순서를 차례로 구분한다. 마지막 행은 “같은 sample이라서 높았는가”와 '
            '“그 자릿수라서 높았는가”를 함께 고려하려는 통계적 비교다. 같은 step에서 답을 다시 생성한 실험은 아니다.',
        '| 모델 | dataset | endpoint FE β': f'위 표는 같은 행의 왼쪽 숫자와 오른쪽 숫자를 비교한다. '
            f'오른쪽은 자릿수·순서까지 고려한 값이며 {adjusted_positive}/{len(adjusted)}개 행에서 CI가 0보다 크다. '
            '오른쪽 값이 작아지면 처음 본 차이의 일부가 생성 위치와 함께 달라졌음을 뜻한다. '
            '예를 들어 0.10 nat은 비교 조건을 적용한 뒤 오류 step의 entropy가 약 0.10 nat 높다는 뜻이지 오류 확률 10%가 아니다.',
        '| 모델 | dataset | endpoint | 경로 수': '위 표는 모델·dataset·start/end별 포함·제외 내역이다. 각 행에서 paired, 오류 없음, '
            '서식 최초 오류, 이전 숫자 없음의 합이 경로 수다. paired만 앞의 오류 전후 평균에 들어간다. '
            'start와 end는 같은 query에서 나온 두 경로일 수 있으므로 행의 수를 독립 query 수처럼 더하지 않는다.',
        '| 모델 | dataset | query 수 | between variance': f'위 표의 between 비중은 “응답 평균이 서로 다른 정도”를 전체 분산으로 나눈 값이다. '
            f'현재 범위는 {100 * min(r["between_fraction"] for r in variances):.1f}–{100 * max(r["between_fraction"] for r in variances):.1f}%다. '
            'within은 같은 응답 안에서 step별 entropy가 달라지는 정도다. between에 출력 길이와 숫자 구성 차이도 섞이므로 sample 고유 불확실성의 비율로 읽지 않는다.',
        '| 모델 | dataset | endpoint | processed': '위 표의 왼쪽은 processed 분포에서도 오류 전후 차이가 나타나는지 확인하는 보조 결과다. '
            '오른쪽 rank는 최초 오류 entropy가 이전 정답 가능 숫자보다 높은 비교의 비율이다(동률은 0.5). '
            '예를 들어 0.8이면 비교의 80%에 해당하는 가중 비율이지, 정답률 80%나 오류 탐지 정확도 80%가 아니다. '
            'all은 같은 query의 start/end 값을 먼저 평균한 집계다.',
        '| 모델 | dataset | 모형 | risk step 수': '위 표는 모델·dataset마다 네 가지 비교 방법을 모두 나열한 상세표다. '
            'risk step 수는 오류 전까지의 숫자와 최초 numeric 오류를 합한 관측 수이고 query 수는 그 출처 질문 수다. '
            'β가 양수면 해당 조건에서 오류 step의 entropy가 높았다는 뜻이다. CI가 0을 가로지르면 방향이 뚜렷하지 않다. '
            '단순 paired 평균과 가중치·통제 조건이 달라 β를 Δprevious와 같은 수치로 기대하면 안 된다.',
    }
    def annotate(document):
        result=[];i=0
        while i<len(document):
            line=document[i]
            if line.startswith('|'):
                header=line
                while i<len(document) and document[i].startswith('|'):
                    result.append(document[i]);i+=1
                note=next((v for k,v in table_notes.items() if header.startswith(k)),None)
                if note is None:raise ValueError('Missing table explanation: '+header)
                result+=['','**표 해설.** '+note,'']
                continue
            result.append(line)
            if line.startswith('!['):
                result+=['','**그래프 해설.** 가로축은 dataset과 start/end, 세로축은 같은 경로에서 계산한 entropy 차이다. '
                    '파란 점은 직전 숫자와의 차이(Δprevious), 주황 점은 이전 숫자 평균과의 차이(Δprior)이고, 막대는 평균의 95% CI다. '
                    f'Δprevious의 평균은 {low:.3f}–{high:.3f} nat이며 {positive}/{len(comparable)}개 그룹의 CI가 0선 위에 있다. '
                    '이 그림의 점 하나는 한 sample이 아니라 여러 query의 평균이다. sample 내부의 개별 step은 결과 탭에서 확인한다.','']
            i+=1
        return result
    return annotate(main), annotate(appendix)
