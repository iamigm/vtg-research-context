"""Attention sidecars reference the shared, measurement-free prediction cache."""
from result_layout import relocate_path, write_result_text
import hashlib
import json
from pathlib import Path
from timelens_paths import baseline_path
from timelens_inputs import write

MEASUREMENT_FIELDS=('attention','baseline_tokens_identical','elapsed_seconds')


def load_answer(folder):
    sidecar=json.loads((relocate_path(relocate_path(Path(folder))/'measurement.json')).read_text())
    path=relocate_path(Path(sidecar['prediction_path']))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sidecar['prediction_sha256']
    prediction=json.loads(path.read_text())
    return dict(prediction,**{k:sidecar[k] for k in MEASUREMENT_FIELDS})


def save_answer(folder,prediction,expanded_input):
    path=baseline_path('qwen3',prediction,prediction['order'])
    pure={k:v for k,v in prediction.items() if k not in MEASUREMENT_FIELDS}
    if path.exists():
        assert json.loads(path.read_text())==pure,'Refusing to overwrite a different cached prediction'
    else:
        write(path,pure)
    write_result_text((relocate_path(path.parent/'expanded_input.txt')), expanded_input)
    write(relocate_path(relocate_path(Path(folder))/'measurement.json'),dict(prediction_path=str(path),
        prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        **{k:prediction[k] for k in MEASUREMENT_FIELDS}))
