#!/usr/bin/env python3
"""One authorized startup-only retry after P10a's preparation lock blocked the client."""
import fcntl
import json
import signal
from pathlib import Path
import shutil
import p11b_runner as r

LOCK=r.ROOT/'configs/p10a_formal.lock.json'
HELD=r.OUT/'p10a_formal.lock.json'
RECOVERY=r.OUT/'launch-recovery'
RETRY='big_muxpp_v5_pp2tp2'


def gate(frozen):
    assert r.git('rev-parse','HEAD')==r.HEAD and not r.git('status','--porcelain')
    assert r.mapping()==frozen['gpu_mapping']
    for rel,digest in frozen['protected_sha256'].items():
        p=r.ROOT/rel
        if p==LOCK and not p.exists():p=HELD
        assert r.sha(p)==digest,rel
    assert not (r.OUT/'STOP-external-gpu.json').exists()


def prepare(frozen):
    assert not RECOVERY.exists()
    assert 'P10a FORMAL LOCK' in (r.OUT/RETRY/'formal-client.log').read_text()
    assert not list((r.ROOT/'results').glob(RETRY+'_*.json*'))
    assert not (r.OUT/'big_muxpp_v5_pp4.launch-started.json').exists()
    RECOVERY.mkdir()
    files=[]
    candidates=[r.OUT/RETRY,*r.OUT.glob(RETRY+'-*'),r.OUT/(RETRY+'.launch-started.json'),
                r.ROOT/'results'/(RETRY+'.formal-started.json'),r.OUT/'big_muxpp_v5_pp4-before-baseline.json']
    for p in candidates:
        if not p.exists():continue
        q=RECOVERY/p.relative_to(r.ROOT);q.parent.mkdir(parents=True,exist_ok=True)
        targets=list(p.rglob('*')) if p.is_dir() else [p]
        hashes={str(x.relative_to(r.ROOT)):r.sha(x) for x in targets if x.is_file()}
        shutil.move(str(p),str(q))
        for rel,digest in hashes.items():
            dest=RECOVERY/rel;assert r.sha(dest)==digest
            files.append(dict(source=rel,destination=str(dest.relative_to(r.ROOT)),sha256=digest))
    r.save(RECOVERY/'manifest.json',dict(utc=r.now(),label=RETRY,formal_requests_sent=0,
          complete_results_produced=False,reason='P10a preparation lock rejected load_trace before warmup/requests',
          correction='Temporarily hold original preparation lock for authorized formal runs; restore byte-for-byte at exit',
          authorized_retry_count=1,files=files))


def main():
    r.nv.nvmlInit()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (r.ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=json.loads((r.OUT/'frozen.json').read_text());gate(frozen)
        for label in r.A[:5]:
            s=json.loads((r.ROOT/'results'/(label+'_summary.json')).read_text())
            assert s['completed']==s['planned_requests'] and s['failed']==0
        with (r.OUT/'resume-started.json').open('x') as f:
            json.dump(dict(utc=r.now(),resume_source_sha256=r.sha(Path(__file__)),
                           completed_preserved=r.A[:5],startup_retry=RETRY),f,indent=2)
        prepare(frozen)
        assert LOCK.exists() and not HELD.exists()
        LOCK.rename(HELD)
        r.gate=gate
        try:
            for label in r.A[5:]:
                r.experiment(label,frozen)
                if label==RETRY:
                    p=r.OUT/label/'run.json';meta=json.loads(p.read_text())
                    meta.update(restart_count=1,formal_client_start_attempts=2,
                                startup_recovery=str(RECOVERY/'manifest.json'))
                    r.save(p,meta)
            chosen={}
            for family in ['tool','big']:
                summaries=[json.loads((r.ROOT/'results'/(l+'_summary.json')).read_text()) for l in r.A if r.family(l)==family]
                winner=min(summaries,key=lambda s:(-s['slo'][r.MAIN]['rate'],s['latencies_sec']['ttft']['p50']
                    if isinstance(s['latencies_sec']['ttft']['p50'],(int,float)) else float('inf')))
                chosen[family]=winner['label']
            r.save(r.OUT/'saturation-selection.json',dict(utc=r.now(),chosen=chosen,rule='highest A main SLO, tie lowest TTFT p50'))
            labels=['sat_muxwise_tp2dp2','sat_chunked_pp2dp2','sat_'+chosen['tool'],
                    'sat_big_muxwise_tp4','sat_big_chunked_pp2tp2','sat_'+chosen['big']]
            for label in labels:r.experiment(label,frozen)
            gate(frozen);r.idle('final',seconds=35)
            r.save(r.OUT/'batch-finished.json',dict(utc=r.now(),labels=r.A+labels,restarts=1,
                   startup_retries={RETRY:1},prior_five_measurements_unchanged=True,
                   external_interruption_archive='archive/aborted_p11b'))
        finally:
            assert not LOCK.exists();HELD.rename(LOCK)
            r.save(r.OUT/'preparation-lock-restored.json',dict(utc=r.now(),path=str(LOCK),sha256=r.sha(LOCK),
                   matches_original=r.sha(LOCK)==frozen['protected_sha256']['configs/p10a_formal.lock.json']))


if __name__=='__main__':main()
