#!/usr/bin/env python3
"""One development-only validation of the selected PP2xDP2 launch candidate."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import pynvml
import p11a_runner as r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--config',type=Path,default=r.ROOT/'configs/p11a_selected.json')
    x=p.parse_args()
    selected=json.loads(x.config.read_text())
    assert r.monitoring.git(x.source,'rev-parse','HEAD')==selected['commit']
    assert not r.monitoring.git(x.source,'status','--porcelain')
    pynvml.nvmlInit();r.monitoring.OUT=r.OUT
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (r.ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(r.OUT.glob('STOP-*.json'))
        r.experiment('v5-dp2','llama_pp2dp2',x.source,selected['flags']['llama_pp2dp2'])


if __name__=='__main__':main()
