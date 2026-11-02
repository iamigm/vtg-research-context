"""Reconstruct frozen plans and explicitly record scout/cache dependencies."""
from adaptive_temporal_sampling import *

def main():
    records=read(OUT/'metadata/baseline_records.json');dependencies=[]
    for record in records:
        s=record['sample'];key=s['sample_key'];cache=read(OUT/'frames'/(key+'.json'));scout_path=OUT/'scouts'/(key+'.json');scout=read(scout_path)
        public=dict(sample_key=key,duration=s['duration'],baseline_ids=record['baseline_metadata']['decoder_metadata']['frames_indices'],candidate_ids=cache['ids'],candidate_pts=cache['pts'])
        for c in CONDITIONS:
            seed=int(c[-2:]) if c.startswith('random') else 42
            pred=scout['parsed'][0] if scout['parsed'] else None
            expected=select_plan(public,c,seed,scout=pred,gt=s['span'][0] if c=='gt' else None)
            planpath=OUT/'plans'/c/(key+'.json');plan=read(planpath)
            need(encoded(plan)==encoded(expected),'Full frozen-plan reconstruction mismatch '+key+' '+c)
            dependencies.append(dict(sample_key=key,condition=c,plan_path=str(planpath),plan_file_sha256=sha(planpath),plan_canonical_hash=objhash(plan),scout_path=str(scout_path),scout_sha256=sha(scout_path),scout_used_for_selection=c=='pred',candidate_cache_path=cache['path'],candidate_cache_sha256=cache['sha256'],source_video_sha256=record['video_sha256'],config_hash=scout['identity']['config_hash'],cohort_hash=scout['identity']['cohort_hash']))
    write(OUT/'metadata/plan_dependencies.json',dict(passed=True,reconstructed_plans=600,producer=fileinfo(Path(__file__)),dependencies=dependencies))
    snap=OUT/'code'/('provenance_'+sha(Path(__file__))[:12]+'.py')
    if not snap.exists():snap.write_bytes(Path(__file__).read_bytes())
    from artifact_store import register
    register(RUN+'.scouts','predictions',OUT/'scouts','100 freshly generated common Uniform scouts',str(Path(__file__).relative_to(ROOT)),'pilot_300','building',[RUN+'.inputs',RUN+'.manifests'])
    print('All 600 frozen plans reconstructed exactly; scout/cache hashes linked.',flush=True)

if __name__=='__main__':main()
