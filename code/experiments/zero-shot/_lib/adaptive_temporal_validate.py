"""CPU integrity fixtures independent of model answers/performance."""
from adaptive_temporal_sampling import *

def main():
    from tempfile import TemporaryDirectory
    checks={}
    with TemporaryDirectory(dir=OUT/'validation') as d:
        p=Path(d)/'response.json';tmp=p.with_suffix('.json.tmp');tmp.write_text('{"partial":')
        need(not p.exists(),'Partial JSON exposed as a complete response')
        identity=dict(run_id=RUN,config_hash='config',cohort_hash='cohort',sample_key='fixture',condition='uniform',planner_seed=None,input_hash='input',plan_hash='plan')
        value=dict(identity=identity,generated_ids=[1,2,3],answer='fixture')
        write(p,value);need(read(p)==value,'Atomic JSON readback differs')
        need(read(p)['identity']==identity,'Valid resume rejected')
        for k in identity:
            wrong=dict(identity);wrong[k]='DIFFERENT'
            need(read(p)['identity']!=wrong,'Mismatched resume accepted: '+k)
        q=Path(d)/'tuple.json';immutable(q,{'intervals':[(1.,2.)]});immutable(q,{'intervals':[(1.,2.)]})
        checks.update(atomic_readback=True,partial_json_not_complete=True,all_identity_fields_mismatch_rejected=True,tuple_json_resume=True)
    records=read(OUT/'metadata/baseline_records.json')
    source=read(OUT/'metadata/source_manifest.json');videos={v['path']:v for v in source['videos']}
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(MODEL),padding_side='left',do_resize=False,local_files_only=True)
    import torch
    torch.set_num_threads(2)
    tests=[]
    for r in records[:5]:
        key=r['sample']['sample_key'];cachefile=OUT/'frames'/(key+'.json')
        if not cachefile.exists():continue
        cache=read(cachefile);plan=read(OUT/'plans/uniform'/(key+'.json'));v=videos[r['sample']['video_path']]
        old,detail=make_input(r,v,cache,plan,processor)
        hashes=detail['hashes']
        for action in ['remove','alter']:
            altered=copy.deepcopy(r)
            if action=='remove':altered['sample'].pop('span')
            else:altered['sample']['span']=[[999999,1000000]]
            changed,newdetail=make_input(altered,v,cache,plan,processor)
            need(newdetail['hashes']==hashes and all(torch.equal(old[k],changed[k]) for k in old),'GT affects generation input '+key)
        tests.append(dict(sample_key=key,removed_gt_input_identical=True,altered_gt_input_identical=True))
    checks.update(gt_input_tests=tests,implementation=fileinfo(Path(__file__)),sampler=sampler_checks())
    write(OUT/'validation/integrity_fixtures.json',dict(passed=True,**checks))
    print(json.dumps(dict(passed=True,gt_input_tests=len(tests))),flush=True)

if __name__=='__main__':main()
