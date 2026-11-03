"""Canonical English technical terms; Korean explanatory prose remains Korean."""
import re
TERMS={
 '결정적 분기 rollout':'deterministic rollout','결정적 완성 rollout':'deterministic rollout','결정적 rollout':'deterministic rollout',
 '결정적 분기 탐색':'deterministic branch search','확률적 rollout':'stochastic rollout','조건부 rollout':'conditional rollout',
 '토큰 분포':'token distribution','누적 경로 확률':'cumulative path probability','경로 확률':'path probability',
 '확률 질량':'probability mass','확률 분포':'probability distribution','엔트로피':'entropy','소프트맥스':'softmax',
 '그리디':'greedy','입력 순서':'input order','프레임 샘플링':'frame sampling','공간 해상도':'spatial resolution',
}
def normalize_terms(text):
 for old,new in sorted(TERMS.items(),key=lambda x:len(x[0]),reverse=True):text=text.replace(old,new)
 return text

def normalize_report(text):
 lines=[];fenced=False
 for line in text.splitlines(keepends=True):
  if line.lstrip().startswith('```'):fenced=not fenced
  if not fenced:line=normalize_terms(line)
  lines.append(line)
 return ''.join(lines)

def normalize_markup(text):
 # Inline datasets and original model answers in script/code blocks are immutable.
 pieces=re.split(r'(<(?:script|style|pre|code)\b[^>]*>.*?</(?:script|style|pre|code)\s*>)',text,flags=re.S|re.I)
 return ''.join(piece if re.match(r'<(?:script|style|pre|code)\b',piece,re.I) else normalize_terms(piece) for piece in pieces)
