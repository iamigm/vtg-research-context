from result_layout import relocate_path, write_result_text
from paths import ROOT, QUERY, GENERATION, HIDDEN, DYNAMICS, SPATIAL
"""Validate measured curves against live-generation arrays and saved reports."""
import csv,hashlib,json,sys
import numpy as np
from PIL import Image
from dynamics import ROOT,PREV,OUT,DATASETS,SCOPES,merged,fractions,write
sys.path.insert(0,str(relocate_path(ROOT/'python_deps')))
from pypdf import PdfReader

def main():
    samples=json.loads((relocate_path(OUT/'samples.json')).read_text())
    assert samples==json.loads((relocate_path(GENERATION / 'samples.json')).read_text())
    counts={};tokens=0;max_delta=0
    for k,s in enumerate(samples):
        src=relocate_path(GENERATION/ s['dataset'] / s['sample_key']);folder=relocate_path(OUT/ s['dataset'] / s['sample_key']);d=json.loads((relocate_path(folder/'dynamics.json')).read_text());a=json.loads((relocate_path(src/'answer.json')).read_text())
        assert d['tokens']==a['tokens'] and d['token_ids']==a['token_ids'] and d['answer']==a['text']
        n=len(d['tokens']);tokens+=n;counts[d['dataset']]=counts.get(d['dataset'],0)+1
        gt=merged(d['gt'],d['duration']);ov=fractions(np.array(d['edges']),gt)
        with np.load(relocate_path(src/'attention.npz')) as z:
            for scope,key in [('predictive','predictive_temporal'),('consumed','answer_temporal')]:
                raw=np.array(d['scopes'][scope]['raw']);expected=z[key][8:22].mean((0,1)).astype(float)
                assert np.array_equal(raw,expected)
                m=d['scopes'][scope]['metrics'];p=raw/raw.sum(-1,keepdims=True)
                assert raw.shape==(n,len(d['timestamps'])) and np.isfinite(raw).all() and (raw>=0).all()
                assert np.allclose(np.array(m['keys']).sum(-1),1,atol=.0041)
                assert np.allclose(raw.sum(-1),m['video_attention'])
                assert np.allclose(p@ov,m['gt_mass']) and np.allclose(raw@ov,m['raw_gt_attention'])
                assert m['tv_previous'][0] is None and np.allclose(np.abs(np.diff(p,axis=0)).sum(-1)/2,m['tv_previous'][1:])
                assert np.allclose(np.array(d['timestamps'])[raw.argmax(-1)],m['peak_seconds'])
            assert np.array_equal(np.array(d['scopes']['predictive']['raw'])[1:],np.array(d['scopes']['consumed']['raw'])[:-1])
        assert d['scopes']['predictive']['metrics']['keys'][0][2]==0
        assert len(list(csv.DictReader((relocate_path(folder/'token_metrics.csv')).open())))==2*n
        assert len(PdfReader(relocate_path(folder/'token_dynamics.pdf')).pages)==1
        with Image.open(relocate_path(folder/'token_dynamics.png')) as im:im.verify()
        if (k+1)%50==0:print('Verified',k+1,flush=True)
    assert counts==dict.fromkeys(DATASETS,100) and tokens==4139
    assert len(list(csv.DictReader((relocate_path(OUT/'all_token_metrics.csv')).open())))==2*tokens
    assert len(PdfReader(relocate_path(OUT/'all_token_dynamics.pdf')).pages)==300
    for ds in DATASETS:assert len(PdfReader(relocate_path(OUT/f'{ds}_token_dynamics.pdf')).pages)==100
    before=json.loads((relocate_path(ROOT/'source_sha256.json')).read_text())
    source=relocate_path(ROOT.parent.parent/'vtg_works/MTLA')
    now={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    assert now==before
    assert all(hashlib.sha256((relocate_path(ROOT/'MTLA'/p)).read_bytes()).hexdigest()==h for p,h in before.items())
    browser=json.loads((relocate_path(OUT/'browser_validation.json')).read_text());assert browser['passed']
    validation=dict(samples_per_dataset=counts,emitted_tokens_excluding_eos=tokens,csv_rows_both_conventions=2*tokens,
        reduced_curves_exactly_match_source_all_300=True,predictive_consumed_shift_verified=True,
        gt_mass_and_adjacent_tv_recomputed=True,original_mtla_unchanged=True,source_measurements_verified=True,
        source_files=len(before),all_300_figures_valid=True,browser_validation=browser)
    write(relocate_path(OUT/'validation.json'),validation)
    paths=[p for p in ROOT.iterdir() if p.is_file() and p.name!='artifact_sha256.json']+[p for p in OUT.rglob('*') if p.is_file() and p.name != "artifact_sha256.json"]
    write(relocate_path(DYNAMICS / 'artifact_sha256.json'),{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(validation,indent=2),flush=True)

if __name__=='__main__':main()
