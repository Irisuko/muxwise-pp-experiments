#!/usr/bin/env python3
"""One sequential development queue; unexpected failures stop for diagnosis."""
import argparse, datetime, json, subprocess, sys, time
from pathlib import Path
from p10a_model import RAM, ROOT, MANIFEST

LOG=ROOT/'logs/p10a'

def event(**data):
    data['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (LOG/'queue-state.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data),flush=True)


def run(args, log):
    with (LOG/log).open('x') as f:
        p=subprocess.run([sys.executable]+args,stdout=f,stderr=subprocess.STDOUT)
    return p.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','reference','correctness','sanity']);a=p.parse_args()
    if a.stage=='preflight':
        remote=json.loads((LOG/'modelscope-files.json').read_text())
        files=[r for r in remote if r['Path'].startswith('model-') or r['Path'] in ('config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json')]
        event(stage='waiting_for_download')
        while not all((RAM/r['Path']).is_file() and (RAM/r['Path']).stat().st_size==r['Size'] for r in files):time.sleep(5)
        if not MANIFEST.exists():
            event(stage='model_validation')
            if run([str(ROOT/'scripts/p10a_model.py'),'manifest'],'model-validation.log'):raise RuntimeError('Model validation failed')
        mems={
            'big_chunked_tp4':[.90,.88,.86], 'big_chunked_pp4':[.90,.88,.86], 'big_chunked_pp2tp2':[.90,.88,.86],
            'big_muxwise_tp4':[.84,.82,.80,.78,.76], 'big_muxpp_pp4':[.82,.80,.78,.76,.74,.72],
            'big_muxpp_pp2tp2':[.82,.80,.78,.76,.74,.72],
        }
        selected_path=ROOT/'configs/p10a_selected.json'
        selected=json.loads(selected_path.read_text()) if selected_path.exists() else {}
        for label,candidates in mems.items():
            if label in selected:continue
            for mem in candidates:
                tag=f'preflight-{label}-m{mem:.2f}'
                event(stage='preflight',label=label,mem=mem,tag=tag)
                rc=run([str(ROOT/'scripts/p10a_runner.py'),'preflight',label,'--mem',str(mem),'--tag',tag],tag+'-driver.log')
                meta=json.loads((LOG/tag/'run.json').read_text()) if (LOG/tag/'run.json').exists() else {}
                if rc==0 and meta.get('status')=='PASS':
                    selected[label]={'mem_fraction':mem,'preflight':tag,'server_info':meta['server_info']}
                    (ROOT/'configs/p10a_selected.json').write_text(json.dumps(selected,indent=2)+'\n')
                    break
                server=(LOG/tag/'server.log').read_text(errors='replace') if (LOG/tag/'server.log').exists() else ''
                if 'out of memory' in server.lower() or 'Not enough memory' in server:
                    event(stage='preflight_oom',label=label,mem=mem)
                    continue
                event(stage='needs_diagnosis',label=label,mem=mem,tag=tag,returncode=rc)
                return 2
            else:raise RuntimeError('All memory candidates failed: '+label)
        event(stage='preflight_done',selected=list(selected))
    elif a.stage=='reference':
        event(stage='reference')
        rc=run([str(ROOT/'scripts/p10a_runner.py'),'correctness','big_plain_tp4','--mem','.82','--tag','reference-tp4-serial20'],'reference-tp4-serial20-driver.log')
        event(stage='reference_done',returncode=rc)
        return rc
    elif a.stage=='correctness':
        for label in ('big_muxwise_tp4','big_muxpp_pp4','big_muxpp_pp2tp2'):
            tag='correctness-'+label
            event(stage='correctness',label=label,tag=tag)
            rc=run([str(ROOT/'scripts/p10a_runner.py'),'correctness',label,'--tag',tag],tag+'-driver.log')
            if rc:
                event(stage='needs_diagnosis',label=label,tag=tag,returncode=rc)
                return rc
        event(stage='correctness_done')
    else:
        selected=json.loads((ROOT/'configs/p10a_selected.json').read_text())
        assert len(selected)==6
        for label in selected:
            tag='sanity-'+label
            cmd=[str(ROOT/'scripts/p10a_runner.py'),'sanity',label,'--tag',tag]
            event(stage='sanity',label=label,tag=tag)
            rc=run(cmd,tag+'-driver.log')
            if rc:
                event(stage='needs_diagnosis',label=label,tag=tag,returncode=rc)
                return rc
        event(stage='sanity_done')
    return 0

if __name__=='__main__':sys.exit(main())
