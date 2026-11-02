"""Read reusable generation records; never load attention arrays."""
from result_layout import relocate_path, write_result_text
import json,hashlib
from timelens_paths import baseline_path,PILOT_SAMPLES,QWEN3_ZERO
def read_prediction(model,dataset,sample_key,order='video_query'):
    return json.loads(baseline_path(model,dict(dataset=dataset,sample_key=sample_key),order).read_text())
def iter_predictions(model='qwen3',order='video_query',cohort=None):
    if model not in ('qwen3','qwen25'):raise ValueError(model)
    if order not in ('video_query','query_video'):raise ValueError(order)
    if model=='qwen25' and order!='video_query':raise ValueError('Qwen2.5 has video_query only')
    if cohort not in (None,'full','pilot_300'):raise ValueError(cohort)
    if model=='qwen25' and cohort=='full':raise ValueError('Qwen2.5 has pilot_300 only')
    path=PILOT_SAMPLES if cohort=='pilot_300' or model=='qwen25' else relocate_path(QWEN3_ZERO/'samples.json')
    for sample in json.loads(path.read_text()):
        yield read_prediction(model,sample['dataset'],sample['sample_key'],order)


def save_prediction(model,prediction,order='video_query'):
    """Write a new immutable baseline; reject changed results under an existing key."""
    from timelens_inputs import write
    pure={k:v for k,v in prediction.items() if k not in ('attention','elapsed_seconds','baseline_tokens_identical')}
    assert not {'steps','restricted_answer'}.intersection(pure)
    path=baseline_path(model,pure,order)
    if path.exists():
        assert json.loads(path.read_text())==pure,'Refusing to overwrite a different cached baseline'
    else:
        write(path,pure)
    return path
