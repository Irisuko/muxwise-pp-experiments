#!/usr/bin/env python3
"""Resume storage recovery and recheck frozen launch settings; never replay dev150."""
import argparse, datetime, hashlib, json, subprocess, sys
from pathlib import Path

ROOT=Path('/root/workspace/muxpp_work')
LOG=ROOT/'logs/p10a'

def event(**data):
    data['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (LOG/'resume-state.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data),flush=True)

def run(arguments, name):
    with (LOG/name).open('x') as output:
        return subprocess.run([sys.executable,*arguments],stdout=output,stderr=subprocess.STDOUT).returncode

def qualify(path):
    meta=json.loads((path/'run.json').read_text())
    if meta.get('status')=='PASS':return 'PASS'
    smoke=meta.get('smoke',{})
    if not (smoke.get('completed',0)>=18 and smoke.get('failed')==0 and smoke.get('unfinished',0)>0):
        return 'NEEDS_DIAGNOSIS'
    details=path/'smoke'/f"{meta['label']}_smoke_requests.jsonl"
    rows=[json.loads(line) for line in details.read_text().splitlines()]
    pending=[r for r in rows if r['status']=='unfinished']
    log=(path/'server.log').read_text(errors='replace')
    live=all(r.get('chunks') and r['chunks'][-1]['at']>=smoke['duration_sec']-5 for r in pending)
    clean=not any(s in log for s in ('OutOfMemoryError','Scheduler hit an exception','#new-token: -'))
    # Continued output at the limit is not a stalled protocol or a 20/20 pass.
    if pending and live and clean:return 'BOUNDED_SMOKE_INCOMPLETE'
    return 'NEEDS_DIAGNOSIS'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflights-only',action='store_true',
                        help='Recheck validated RAM while an independent persistent copy finishes')
    args=parser.parse_args()
    selected=ROOT/'configs/p10a_selected.json'
    frozen=json.loads((ROOT/'configs/p10a_frozen.json').read_text())
    assert hashlib.sha256(selected.read_bytes()).hexdigest()==frozen['selected_sha256']
    if args.preflights_only:
        from p10a_model import ensure
        ensure()
    else:
        event(stage='wait_for_model_recovery')
        if run([str(ROOT/'scripts/p10a_model.py'),'ensure'],'resume-ensure-verified.log'):
            raise RuntimeError('Model recovery/validation failed')
        event(stage='persistent_copy')
        if run([str(ROOT/'scripts/p10a_model.py'),'persist'],'resume-persist.log'):
            raise RuntimeError('Persistent copy failed')
    results={}
    for label in frozen['selected']:
        tag='resume-preflight-'+label
        path=LOG/tag
        event(stage='preflight',label=label,tag=tag)
        rc=run([str(ROOT/'scripts/p10a_runner.py'),'preflight',label,'--tag',tag],tag+'-driver.log')
        status=qualify(path) if (path/'run.json').exists() else 'NEEDS_DIAGNOSIS'
        results[label]={'tag':tag,'returncode':rc,'status':status}
        (LOG/'resume-preflights.json').write_text(json.dumps(results,indent=2)+'\n')
        if status=='NEEDS_DIAGNOSIS':
            event(stage='needs_diagnosis',label=label,tag=tag)
            return 2
    event(stage='preflights_done' if args.preflights_only else 'recovery_and_preflights_done',results=results)
    return 0

if __name__=='__main__':sys.exit(main())
