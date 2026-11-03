"""Update native producers and local viewers for the single artifact hierarchy."""
import json,re,os,ast
from pathlib import Path
from result_layout import ROOT,relocate_path,rewrite_document
STATE=ROOT/'artifacts/maintenance'
changed=set(json.loads((STATE/'changed_files.json').read_text()))
# Native similarity producers use the canonical run data path, including new workers.
sim=ROOT/'similarity/_lib'
p=sim/'prepare.py';s=p.read_text().replace("OUT = VIEW/'artifacts'","OUT = ROOT/'artifacts/runs'/RUN");p.write_text(s);changed.add(str(p))
p=sim/'run_pipeline.py';s=p.read_text().replace("ROOT/'results/pilot_300_v1/artifacts'","ROOT/'artifacts/runs/pilot_300_v1'").replace("ROOT/'artifacts/logs'","ROOT/'artifacts/common/logs'");p.write_text(s);changed.add(str(p))
for name in ['report.py','check_viewer.py','prepare_media.py']:
 p=sim/name;s=p.read_text()
 if 'from result_layout import' not in s:
  s='from result_layout import relocate_path, write_result_text\n'+s
 if name=='report.py':
  s=s.replace("(VIEW/'REPORT.md').write_text(","write_result_text(VIEW/'REPORT.md', ").replace("(VIEW/'report.html').write_text(","write_result_text(VIEW/'report.html', ")
 if name=='check_viewer.py':s=s.replace('target.write_text(', 'write_result_text(target, ')
 if name=='prepare_media.py':
  s=s.replace('from prepare import OUT, video_key, write','from prepare import OUT, VIEW, video_key, write')
  s=s.replace("cached=Path('/home/irteam/experiments/zero-shot/results/uncertainty/token_analysis/timelens_top30_full/media')","cached=relocate_path(Path('/home/irteam/experiments/zero-shot/results/uncertainty/token_analysis/timelens_top30_full/media'))")
  s=s.replace("media=f'artifacts/media/{key}.mp4'","media=os.path.relpath(target,VIEW)")
 ast.parse(s);p.write_text(s);changed.add(str(p))
link=sim/'result_layout.py'
if not link.exists():link.symlink_to('../../_lib/result_layout.py')
# Reusable cross-run records are an explicit common store; new kinds follow the same rule.
p=ROOT/'zero-shot/_lib/artifact_store.py';s=p.read_text().replace("STORE=relocate_path(ROOT/'artifacts')","STORE=relocate_path(ROOT/'artifacts/common')");p.write_text(s);changed.add(str(p))
# Every local viewer serves the experiment tree and redirects its port root to its own page.
servers=[ROOT/'zero-shot/_lib'/name for name in ['timelens_top30_serve.py','timelens_rollout_serve.py','timelens_entropy_integrated_serve.py','timelens_digit_correctness_serve.py','timelens_full_digit_serve.py']]
servers.append(ROOT/'analysis/failure_mode_timestamp/failure_cases/_lib/serve_gallery.py')
for p in servers:
 s=p.read_text();m=re.search(r'^ROOT = (.+)$',s,re.M);expr=m.group(1)
 code='RESULT_VIEW = '+expr+"\nROOT = Path('/home/irteam/experiments')\n"
 if p.name=='serve_gallery.py':code+="LANDING = '/' + str(relocate_path(RESULT_VIEW/'cases/report.html')).replace('/home1/irteam/','/home/irteam/').split('/experiments/',1)[1]"
 else:code+="LANDING = '/' + str(RESULT_VIEW).replace('/home1/irteam/','/home/irteam/').split('/experiments/',1)[1] + '/report.html'"
 s=s[:m.start()]+code+s[m.end():]
 if p.name=='serve_gallery.py':s=s.replace("self.send_header('Location', '/artifacts/data/cases/')","self.send_header('Location', LANDING)")
 else:
  marker='        self.byte_count = None\n'
  s=s.replace(marker,marker+"        if self.path.split('?',1)[0] == '/':\n            self.send_response(302)\n            self.send_header('Location', LANDING)\n            self.send_header('Content-Length', '0')\n            self.end_headers()\n            return None\n",1)
 ast.parse(s);p.write_text(s);changed.add(str(p))
p=relocate_path(ROOT/'zero-shot/results/baseline/input_order/consistency_v1/artifacts/code/serve.py');s=p.read_text().replace('server.ROOT=ROOT',"server.ROOT=ROOT\nserver.LANDING='/zero-shot/results/baseline/input_order/consistency_v1/report.html'");p.write_text(s);changed.add(str(p))
# Port 8775 now redirects to its own report; browser checks use the redirected locations.
p=ROOT/'_lib/check_result_viewers.py';s=p.read_text();s=s.replace("ROOT/'analysis/failure_mode_timestamp/failure_cases/results/pilot_100'","ROOT/'analysis/failure_mode_timestamp/failure_cases/results/pilot_100'");p.write_text(s)
# Storage docs and immutable resource identity metadata follow their actual new paths.
p=ROOT/'zero-shot/artifacts/README.md';s=rewrite_document(p.read_text(),p,p);p.write_text(s);changed.add(str(p))
p=ROOT/'zero-shot/artifacts/common/predictions/legacy/README.md';old=ROOT/'zero-shot/artifacts/predictions/legacy/README.md';p.write_text(rewrite_document(p.read_text(),old,p));changed.add(str(p))
for p in (ROOT/'zero-shot/artifacts/common/registry').glob('*.json'):
 obj=json.loads(p.read_text());path=obj.get('path')
 if path:
  target=relocate_path(ROOT/'zero-shot'/path);obj['path']=os.path.relpath(target,ROOT/'zero-shot');p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');changed.add(str(p))
# Extend original numeric baselines to cover JSON path-only changes in this migration.
from update_result_json_paths import digest
upfile=STATE/'json_path_updates.json';updates={x['old']:x for x in json.loads(upfile.read_text())};original=json.loads((STATE/'before_metrics_sha256.json').read_text());hierarchy=json.loads((STATE/'artifact_hierarchy.json').read_text())
by_dest={str(relocate_path(Path(k))):k for k in original}
for edit in hierarchy['json_path_updates']:
 if edit['path'] in by_dest:
  old=by_dest[edit['path']];obj=json.loads(Path(edit['path']).read_text());assert digest(obj)==edit['numeric_tree_sha256'];updates[old]=dict(edit,old=old)
upfile.write_text(json.dumps(list(updates.values()),indent=2)+'\n')
(STATE/'changed_files.json').write_text(json.dumps(sorted(changed),indent=2)+'\n')
print('Native producers, storage catalog and seven viewer entrypoints updated')
