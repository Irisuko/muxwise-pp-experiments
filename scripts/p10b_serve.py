#!/usr/bin/env python3
"""Launch the selected P10B matrix; no trace replay entrypoint."""
import argparse,datetime,json,shlex,signal,subprocess
from pathlib import Path
import p10a_runner as a
from p7a_runner import Cluster
ROOT=a.ROOT
SELECTED=ROOT/'configs/p10b_selected.json'

def commands(label):
    selected=json.loads(SELECTED.read_text())
    if label.startswith('big_muxpp_'):
        cfg=selected[label]
        jobs=a.commands(label,cfg['mem_fraction'])
        for _,cmd,env,_ in jobs:
            # Replace scalar arguments rather than leave duplicate overrides.
            flags=cfg['flags'];i=0
            while i<len(flags):
                key=flags[i]
                if i+1<len(flags) and not flags[i+1].startswith('--'):
                    val=flags[i+1];i+=2
                    if key in cmd:cmd[cmd.index(key)+1]=val
                    else:cmd.extend([key,val])
                else:
                    i+=1
                    if key not in cmd:cmd.append(key)
            env.update(cfg['env'])
        return jobs
    return a.commands(label)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['commands','launch'])
    p.add_argument('label',choices=[k for k in a.MATRIX if k!='big_plain_tp4'])
    p.add_argument('--tag')
    x=p.parse_args();jobs=commands(x.label)
    if x.mode=='commands':
        print('\n'.join(shlex.join(['env']+[k+'='+v for k,v in env.items()]+cmd) for _,cmd,env,_ in jobs));return
    a.ensure();a.base.commands=lambda _:jobs
    tag=x.tag or 'launch-'+x.label+'-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    c=Cluster(x.label,ROOT/'logs/p10b'/tag)
    c.meta['commit']=subprocess.check_output(['git','-C',str(a.SOURCE),'rev-parse','HEAD'],text=True).strip()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:c.start();signal.pause()
    except KeyboardInterrupt:c.meta['status']='STOPPED'
    except BaseException as e:c.meta.update(status='FAIL',error=repr(e));raise
    finally:c.close()
if __name__=='__main__':main()
