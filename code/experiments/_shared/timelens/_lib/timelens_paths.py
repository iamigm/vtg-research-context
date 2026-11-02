"""Shared prediction cache; experiments remain independent consumers."""
from result_layout import relocate_path, write_result_text
from pathlib import Path
EXPERIMENTS=relocate_path(Path(__file__)).resolve().parents[3]
ATTENTION_ROOT=relocate_path(EXPERIMENTS/'attention')
ZEROSHOT_ROOT=relocate_path(EXPERIMENTS/'zero-shot')
RESOURCES=relocate_path(EXPERIMENTS/'_shared/timelens')
ZEROSHOT=relocate_path(ZEROSHOT_ROOT/'predictions')
QWEN3_ZERO=relocate_path(ZEROSHOT/'qwen3')
QWEN25_ZERO=relocate_path(ZEROSHOT/'qwen25')
PILOT_COHORT=relocate_path(ZEROSHOT/'cohorts/pilot_300')
PILOT_SAMPLES=relocate_path(PILOT_COHORT/'samples.json')
PILOT_SELECTION=relocate_path(PILOT_COHORT/'selection.json')
def baseline_path(model,sample,order='video_query'):
    if model=='qwen3':return relocate_path(QWEN3_ZERO/sample['dataset']/sample['sample_key']/order/'answer.json')
    if model=='qwen25' and order=='video_query':return relocate_path(QWEN25_ZERO/'predictions'/(sample['sample_key']+'.json'))
    raise ValueError((model,order))
def pilot_export(model,dataset):
    path=relocate_path(ZEROSHOT/model/'pilot_300'/(dataset+'_video_query.jsonl'))
    path.parent.mkdir(parents=True,exist_ok=True)
    return path
