"""Write current-layout reports without interpreting JavaScript identifiers as URLs."""
import os
from pathlib import Path
import re
import shutil
from result_layout import relocate_path, canonical

ROOT=Path('/home/irteam/experiments/zero-shot')


def document_base(path):
    value=canonical(path)
    if '/artifacts/runs/' in value and '/web/' in value:
        return Path(value.replace('/artifacts/runs/','/results/').split('/web/')[0])
    return Path(value).parent


def reference(value, base):
    if not value or value.startswith(('http:','https:','data:','mailto:','javascript:','#','//')):
        return value
    raw,sep,fragment=value.partition('#')
    path=Path(raw) if raw.startswith('/home') else Path(os.path.normpath(str(base/raw)))
    path=Path(canonical(relocate_path(path)))
    return os.path.relpath(path,base)+(sep+fragment if sep else '')


def write_result_text(path,text,*args,**kwargs):
    path=Path(canonical(relocate_path(path)))
    base=document_base(path)
    if path.suffix=='.js' and path.name=='viewer.js':
        # These are the only URL prefixes in the viewer source; JSON contents and
        # token strings must never pass through a generic text/path replacement.
        for prefix in ('traces/','raw/'):
            url=reference(prefix,base).rstrip('/')+'/'
            text=text.replace("'"+prefix+"'","'"+url+"'")
        samples=ROOT/'artifacts/runs/uncertainty/token_analysis/pilot/300/samples'
        text=text.replace("'artifacts/samples/'","'"+os.path.relpath(samples,base)+"/'")
    elif path.suffix=='.html':
        from result_terms import normalize_markup
        from build_index import add_navigation
        text=normalize_markup(text)
        text=re.sub(r'\b(src|href)=(\x22|\x27)([^\x22\x27]+)\2',
                    lambda m:m[1]+'='+m[2]+reference(m[3],base)+m[2],text)
        text=add_navigation(path,text)
    elif path.suffix=='.md':
        from result_terms import normalize_report
        from build_index import scope_label
        text=normalize_report(text)
        text=re.sub(r'\]\(([^)]+)\)',lambda m:']('+reference(m[1],base)+')',text)
        scope=scope_label(path)
        first,sep,rest=text.partition('\n')
        if scope and scope not in first:text=first+' · '+scope+sep+rest
    elif path.suffix=='.css':
        text=re.sub(r'url\(([\x22\x27]?)([^)\x22\x27]+)\1\)',
                    lambda m:'url('+m[1]+reference(m[2],path.parent)+m[1]+')',text)
    path.parent.mkdir(parents=True,exist_ok=True)
    return path.write_text(text,*args,**kwargs)


def copy_result_file(source,destination,*args,**kwargs):
    destination=Path(canonical(relocate_path(destination)))
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.suffix in ('.html','.md','.js','.css'):
        write_result_text(destination,Path(source).read_text())
        return destination
    return shutil.copyfile(source,destination,*args,**kwargs)


def media_reference(base, output, value):
    target=Path(canonical(relocate_path(Path(os.path.normpath(str(base/value))))))
    assert target.exists(),target
    return os.path.relpath(target,Path(canonical(output)))
