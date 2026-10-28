#!/usr/bin/env python3
"""After R6 cleanup, validate/commit CPU prefill metadata and freeze R6."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import pynvml
import p11a_runner as r

FILES = ['python/sglang/srt/multiplex/multiplexing_pp.py',
         'python/sglang/srt/multiplex/pp_chunk.py', 'python/sglang/srt/managers/tp_worker.py',
         'python/sglang/srt/layers/attention/flashinfer_backend.py',
         'python/sglang/srt/server_args.py', 'test/srt/test_multiplexing_pp_chunk.py']
TESTS = ['test/srt/test_multiplexing_'+name+'.py' for name in
         ['pp','pp_admission','pp_calibration','pp_chunk','pp_decode','pp_policy','pp_tp','mlp_chunk']]


def patch_hash():
    return hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=r.MAIN)).hexdigest()


def main():
    expected = patch_hash()
    original = r.monitoring.git(r.MAIN,'rev-parse','HEAD')
    print('WAIT R5; frozen patch',expected,flush=True)
    while True:
        assert not list(r.OUT.glob('STOP-*.json'))
        try:meta=json.loads((r.OUT/'r5-big_pp4/run.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):meta={}
        if 'finished_utc' in meta:
            assert meta['status']=='PASS',meta
            break
        time.sleep(5)
    time.sleep(15)
    for label in r.LABELS[:4]:
        m=json.loads((r.OUT/('r5-'+label)/'run.json').read_text())
        assert m['status']=='PASS' and all(k in m for k in ['standard','double','saturated'])
    assert r.monitoring.git(r.MAIN,'rev-parse','HEAD')==original
    assert r.monitoring.git(r.MAIN,'rev-parse','--abbrev-ref','HEAD')=='muxwise-pp'
    assert patch_hash()==expected
    status=subprocess.check_output(['git','status','--porcelain'],cwd=r.MAIN,text=True)
    assert {line[3:] for line in status.splitlines()}==set(FILES),status
    source=r.MAIN.parent/'sglang-p11a-r6';assert not source.exists()
    pynvml.nvmlInit();r.monitoring.OUT=r.OUT
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    env=dict(os.environ,PYTHONPATH=str(r.MAIN/'python'))
    with (r.ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r.monitoring.idle('r6-prepare-before')
        guard=r.Guard('r6-prepare');guard.start()
        try:
            print('CPU TESTS',flush=True)
            with (r.ROOT/'logs/p11a-unit-r6-full.log').open('x') as log:
                subprocess.run([r.b.PYTHON,'-m','pytest',*TESTS,'-q'],cwd=r.MAIN,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            with (r.ROOT/'logs/p11a-format-r6.log').open('x') as log:
                for cmd in [[r.b.PYTHON,'-m','isort','--check-only',*FILES],
                            [r.b.PYTHON,'-m','black','--check',*FILES],['git','diff','--check']]:
                    subprocess.run(cmd,cwd=r.MAIN,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert patch_hash()==expected
            subprocess.run(['git','add','--',*FILES],cwd=r.MAIN,check=True)
            subprocess.run(['git','commit','-m','Avoid CUDA readback for pipeline prefill sequence and prefix metadata'],cwd=r.MAIN,check=True)
            head=r.monitoring.git(r.MAIN,'rev-parse','HEAD')
            assert not r.monitoring.git(r.MAIN,'status','--porcelain')
            subprocess.run(['git','worktree','add','--detach',str(source),head],cwd=r.MAIN,check=True)
            r.b.save(r.OUT/'r6-preparation.json',dict(original=original,commit=head,patch_sha256=expected,
                     source=str(source),tests=TESTS,status='CPU_PASS_COMMITTED'))
        finally:
            guard.close()
        assert not guard.error
        time.sleep(3);r.monitoring.idle('r6-prepare-after')
    print('READY R6',head,str(source),flush=True)


if __name__=='__main__':main()
