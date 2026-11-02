"""Regenerate plots and report from saved results; no inference."""
from result_layout import relocate_path, write_result_text
import csv,json,textwrap,time,traceback
from pathlib import Path
import timelens_digit_correctness as source
from timelens_digit_correctness import OUT,MODELS,DATASETS,ROLES,np,plt
rng=np.random.default_rng(42)
try:
    print('Starting plot refresh: 0/2 (0%)',flush=True)
    with (relocate_path(OUT/'per_step.csv')).open() as f:allsteps=list(csv.DictReader(f))
    for row in allsteps:
        row['entropy']=float(row['entropy'])
        for key in ['numeric','first_error','previous_correct','prefix_correct']:row[key]=row[key]=='True'
    code=relocate_path(Path(source.__file__)).read_text()
    start=code.index("    for ep in ['start','end']:\n        fig,axes=")
    end=code.index('    # Each sample',start)
    exec(textwrap.dedent(code[start:end]))
    source.write_report(json.loads((relocate_path(OUT/'summary.json')).read_text()))
    write_result_text((relocate_path(OUT/'plot_revision.json')), json.dumps(dict(complete=True,timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),completed=2,total=2,percent=100,axis_explanation_removed=True,statistics_unchanged=True),indent=2)+'\n')
    print('Complete: 2/2 (100%); statistics unchanged.',flush=True)
except Exception:
    write_result_text((relocate_path(OUT/'plot_revision.json')), json.dumps(dict(complete=False,error=traceback.format_exc())))
    raise
