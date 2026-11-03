"""Render all prespecified metrics and axes from completed evaluation, without inference."""
from result_layout import write_result_text
from result_layout import relocate_path
from pathlib import Path
import csv
from adaptive_temporal_sampling import ROOT
RUN='gt_density_allocation_comparison_pilot900_v1'
RUNS={'A':'gt_extent_boundary_density_pilot900_v1','B':'gt_actual_density_pilot900_v1'}
DS=[('charades-timelens','Charades'),('activitynet-timelens','ActivityNet'),('qvhighlights-timelens','QVHighlights')]
METRICS=[('miou','mIoU'),('r03','IoU≥0.3 성공률'),('r05','IoU≥0.5 성공률'),('r07','IoU≥0.7 성공률')]
TB=['<30','30–60','60–120','120–180','≥180'];QB=['<5%','5–10%','10–20%','20–40%','≥40%']
MARK='<!-- COMPLETE_METRICS_AND_AXES -->'
def name(c):
    if c=='uniform':return 'Uniform'
    p=c.split('_');return ('GT' if p[0]=='gt' else 'Random')+' '+('전체' if p[1]=='whole' else '경계')+' '+p[2][1:]+'배'+(' (seed 평균)' if p[-1]=='mean' else (' (seed '+p[-1][1:]+')' if p[-1].startswith('s') else ''))
def table(headers,rows):return ['','| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]+['']
def render():
    root=relocate_path(ROOT/'results'/RUN);out=relocate_path(ROOT/'artifacts/runs'/RUN)
    metrics=list(csv.DictReader((out/'metrics/nested_metrics.csv').open()))
    cells=list(csv.DictReader((out/'metrics/nested_axes.csv').open()))
    ix={(x['size'],x['dataset'],x['design'],x['condition'],x['metric']):x for x in metrics}
    ax={(x['size'],x['dataset'],x['design'],x['axis'],x['duration_bin'],x['GT_fraction_bin'],x['condition'],x['metric']):x for x in cells}
    def score(n,ds,d,c,m):return float(ix[str(n),ds,d,c,m]['percent'])
    def cell(n,ds,d,axis,t,q,c,m):return ax[str(n),ds,d,axis,t,q,c,m]
    def cross(n,ds,d,c):
        lines=[]
        for m,label in METRICS:
            lines+=['','#### '+label+'의 Uniform 대비 변화 (pp)']
            rows=[]
            for q in QB:
                vals=[]
                for t in TB:
                    x=cell(n,ds,d,'joint',t,q,c,m);count=int(x['N'])
                    vals.append('— (0)' if not count else f"{float(x['delta_pp']):+.2f} ({count})"+(' †' if count<10 else ''))
                rows.append([q]+vals)
            lines+=table(['GT/영상 길이 비율']+[t+'초' for t in TB],rows)
        return lines
    lines=['# 900개 실험: 사건 전체와 경계 중 어디를 더 자세히 볼 것인가?','',
        '이 보고서는 완료된 900개 질문의 결과입니다. Charades·ActivityNet·QVHighlights 각 300개이며, 기존 pilot 300개를 그대로 포함합니다. 추가 추론 없이 저장된 질문별 평가를 다시 집계했습니다.','',
        '## 표를 읽는 방법','',
        '- 시간 구간 겹침률(IoU) = 예측과 정답이 겹친 길이 ÷ 두 구간의 합집합 길이. mIoU는 질문별 IoU의 평균입니다. 표에서는 100을 곱해 %로 표시합니다.\n- IoU≥0.3·0.5·0.7 성공률 = 해당 기준 이상인 질문 수 ÷ 전체 질문 수 × 100. 0.7이 더 엄격한 기준입니다. 파싱 실패와 무효 구간도 분모에 포함하고 실패로 셉니다.\n- 변화(pp) = 해당 조건의 점수 − 비교 조건의 점수. +는 개선, −는 악화입니다. 40%에서 42%로 바뀌면 +2pp입니다.\n- GT 비율 = 정답 구간 길이 ÷ 영상 길이. 영상 길이와 다른 축입니다. 구간의 왼쪽 경계는 포함하고 오른쪽 경계는 제외합니다. 마지막 GT 비율 구간은 100%를 포함합니다.\n- Random은 정답과 길이가 같은 구간의 위치만 무작위로 정한 진단 대조군입니다. seed 42·43 각각에 성공 여부를 계산한 뒤 평균합니다. 두 응답을 독립 질문으로 세지 않습니다.','',
        '## 두 실험이 다른 이유','',
        '**A — 집중 프레임 수 공유:** 사건 전체에 쓸 집중 프레임 수 K를 정한 뒤, 경계에도 같은 K를 씁니다. “같은 집중 프레임을 전체에 쓸까, 경계에 쓸까?”를 비교합니다. K=round(F×rL/(rL+T−L)).','',
        '**B — 안팎 밀도 비율 공유:** 각 집중 영역의 실제 길이 D로 K=round(F×rD/(rD+T−D))를 계산합니다. “각 영역 안을 밖보다 r배 촘촘하게 보면 어떨까?”를 비교합니다. 경계 영역이 짧으면 집중 프레임 수도 줄어듭니다.','',
        'F는 총 프레임 수, T는 영상 길이, L은 GT 전체 길이, r은 2·4·8입니다. 경계는 시작·종료 각각 ±1초의 합집합입니다. 같은 질문의 총 프레임 수·해상도·시각 토큰 수는 조건 간 같습니다. 후보 프레임 부족 시 배정 조정은 기존 실행 기록을 따릅니다. Uniform과 전체 집중 조건은 A·B가 공유하므로 독립 반복 결과가 아닙니다.','',
        '## 1. 900개: 조건별 실제 점수','',
        '아래 표는 성공률 자체입니다. 다음 변화량 표는 같은 질문끼리 뺀 값입니다. 전체 8배는 이전 분석과 이어서 보기 위한 기준 조건이며, 최고 조건을 골라 보고한 것이 아닙니다. 2·4·8배와 경계 조건 전체 표는 뒤의 상세 보고서에 있습니다.']
    for ds,label in DS:
        lines+=['','### '+label+' — 300개 질문']
        conds=['uniform','gt_whole_r8','random_whole_r8_mean']
        lines+=table(['조건']+[label+' (%)' for _,label in METRICS],[[name(c)]+[f'{score(900,ds,"A",c,m):.2f}' for m,_ in METRICS] for c in conds])
        lines+=table(['비교']+[label+' 변화 (pp)' for _,label in METRICS],[[name(a)+' − '+name(b)]+[f'{score(900,ds,"A",a,m)-score(900,ds,"A",b,m):+.2f}' for m,_ in METRICS] for a,b in [('gt_whole_r8','uniform'),('gt_whole_r8','random_whole_r8_mean')]])
    lines+=['','## 2. 영상 길이와 GT 비율을 각각 나눈 비교','',
        'GT 전체 8배 − Uniform의 변화입니다. 영상 길이별 표는 GT 비율을 고정하지 않고, GT 비율별 표는 영상 길이를 고정하지 않습니다. 두 요소를 함께 나눈 결과는 다음 절에서 봅니다. N은 질문 수입니다.']
    for ds,label in DS:
        lines+=['','### '+label]
        for axis,title,bins in [('duration','영상 길이별',TB),('GT_fraction','GT 비율별',QB)]:
            lines+=['','#### '+title];rows=[]
            for b in bins:
                t,q=(b,'all') if axis=='duration' else ('all',b)
                xs=[cell(900,ds,'A',axis,t,q,'gt_whole_r8',m) for m,_ in METRICS]
                rows.append([b,xs[0]['N']]+['—' if not int(x['N']) else f"{float(x['delta_pp']):+.2f}" for x in xs])
            lines+=table(['구간','N']+[l+' 변화 (pp)' for _,l in METRICS],rows)
    lines+=['','## 3. 영상 길이 × GT 비율: 두 축 동시 비교','',
        'GT 전체 8배 − Uniform입니다. 행은 GT 비율, 열은 영상 길이입니다. **각 칸은 변화 pp (질문 수 N)**입니다. †는 N<10, — (0)은 표본이 없다는 뜻입니다. 네 표는 각각 평균 겹침률과 0.3·0.5·0.7 기준의 성공률 변화를 보여 줍니다. 표본이 없는 칸을 0점으로 해석하지 마세요.']
    for ds,label in DS:lines+=['','### '+label]+cross(900,ds,'A','gt_whole_r8')
    lines+=['','## 4. 기존 300개 → 600개 → 900개','',
        '세 집합은 중첩됩니다. 부족한 길이×GT 비율 구간을 보완했으므로 표본 수와 구성 모두 달라집니다. 점수 변화가 표본 수 증가만의 효과는 아닙니다. 아래는 GT 전체 8배의 실제 점수와 Uniform 대비 변화입니다.']
    for ds,label in DS:
        lines+=['','### '+label]
        lines+=table(['전체 코호트 / 해당 데이터셋 N']+[l+' % (변화 pp)' for _,l in METRICS],[[f'{n} / {n//3}']+[f'{score(n,ds,"A","gt_whole_r8",m):.2f} ({score(n,ds,"A","gt_whole_r8",m)-score(n,ds,"A","uniform",m):+.2f})' for m,_ in METRICS] for n in [300,600,900]])
    lines+=['','## 5. 전체 조건 상세표와 해석 범위','',
        '아래 상세 보고서는 Uniform, 전체·경계 2·4·8배, Random 두 seed 및 평균의 실제 점수와 변화량을 모두 포함합니다. 각 GT 조건에 대해 영상 길이별·GT 비율별·두 축 교차표를 네 지표 모두 제공합니다.','',
        '[A 전체 조건·네 지표·각 축·교차표](DETAIL_A.md) · [B 전체 조건·네 지표·각 축·교차표](DETAIL_B.md)','',
        '이 표는 관찰된 점수와 차이를 보여 줍니다. 작은 칸의 차이가 통계적으로 확실하다는 뜻은 아닙니다. 원래 실행별 보고서의 영상 단위 bootstrap 95% 구간은 그 보고서에 적힌 mIoU 비교에만 해당하며, 이 교차표나 threshold 차이의 신뢰구간으로 옮겨 해석하면 안 됩니다. 데이터셋의 영상 길이 분포가 겹치지 않는 칸에서는 데이터셋 효과와 영상 길이 효과를 분리할 수 없습니다. 해상도를 고정한 새 실험도 아닙니다.','',
        f'[A 실행별 신뢰구간·오차·비용](../{RUNS["A"]}/REPORT.md) · [B 실행별 신뢰구간·오차·비용](../{RUNS["B"]}/REPORT.md)','',
        f'[전체 점수 CSV](../../artifacts/runs/{RUN}/metrics/nested_metrics.csv) · [각 축과 교차 칸 CSV](../../artifacts/runs/{RUN}/metrics/nested_axes.csv)']
    write_result_text(root/'REPORT.md', '\n'.join(lines)+'\n')
    for d,run in RUNS.items():
        details=[f'# 실험 {d}: 모든 조건과 구간별 비교','', '[통합 보고서와 지표 정의](REPORT.md)','',
            '각 코호트에서 데이터셋별 질문 수는 전체 코호트의 1/3입니다. 점수 표는 실제 % (Uniform 대비 pp)입니다. Random은 각 seed와 두 seed 평균을 모두 표시합니다. 축별·교차표는 GT 조건−Uniform 변화 pp이며, 교차표 괄호는 질문 수입니다. †는 N<10입니다.']
        for n in [900,600,300]:
            details+=['',f'## 코호트 {n}개']
            for ds,label in DS:
                details+=['',f'### {label}: 실제 점수와 Uniform 대비 변화']
                conds=['uniform']+[f'{who}_{mode}_r{r}{suffix}' for r in [2,4,8] for mode in ['whole','boundary'] for who,suffix in [('gt',''),('random','_s42'),('random','_s43'),('random','_mean')]]
                details+=table(['조건']+[l+' % (변화 pp)' for _,l in METRICS],[[name(c)]+[f'{score(n,ds,d,c,m):.2f} ({score(n,ds,d,c,m)-score(n,ds,d,"uniform",m):+.2f})' for m,_ in METRICS] for c in conds])
                details+=table(['GT − 대응 Random 평균']+[l+' 변화 pp' for _,l in METRICS],[[name(c)]+[f'{score(n,ds,d,c,m)-score(n,ds,d,c.replace("gt_","random_")+"_mean",m):+.2f}' for m,_ in METRICS] for c in conds if c.startswith('gt_')])
                for c in [c for c in conds if c.startswith('gt_')]:
                    details+=['',f'### {label} / {name(c)} / {n}개 코호트']
                    for axis,title,bins in [('duration','영상 길이별',TB),('GT_fraction','GT 비율별',QB)]:
                        details+=['','#### '+title];rows=[]
                        for b in bins:
                            t,q=(b,'all') if axis=='duration' else ('all',b)
                            xs=[cell(n,ds,d,axis,t,q,c,m) for m,_ in METRICS]
                            rows.append([b,xs[0]['N']]+['—' if not int(x['N']) else f"{float(x['delta_pp']):+.2f}" for x in xs])
                        details+=table(['구간','N']+[l+' 변화 pp' for _,l in METRICS],rows)
                    details+=cross(n,ds,d,c)
        (root/f'DETAIL_{d}.md').write_text('\n'.join(details)+'\n')
        rp=relocate_path(ROOT/'results'/run/'REPORT.md');old=rp.read_text().split(MARK)[0].rstrip()
        extra=[MARK,'','## IoU 기준별 전체 성능과 두 축 비교','',f'[실험 {d}의 전체 조건·300/600/900·영상 길이×GT 비율 교차표](../{RUN}/DETAIL_{d}.md)','', '아래는 900개에서의 실제 성공률입니다. 각 축과 교차표는 위 상세 보고서에 네 지표 모두 수록했습니다.']
        for ds,label in DS:
            extra+=['','### '+label]
            conds=['uniform']+[f'{who}_{mode}_r{r}{suffix}' for r in [2,4,8] for mode in ['whole','boundary'] for who,suffix in [('gt',''),('random','_mean')]]
            extra+=table(['조건']+[l+' (%)' for _,l in METRICS],[[name(c)]+[f'{score(900,ds,d,c,m):.2f}' for m,_ in METRICS] for c in conds])
        rp.write_text(old+'\n\n'+'\n'.join(extra)+'\n')
if __name__=='__main__':render()
