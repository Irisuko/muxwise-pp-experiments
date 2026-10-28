#!/usr/bin/env python3
"""Resume only unlaunched P10c labels after an audited cleanup guard false alarm."""
import csv,fcntl,json,os,signal,time
from pathlib import Path
import psutil,pynvml
import p10c_runner as r

class Guard(r.Guard):
    def run(self):
        hidden_count=0;known={};last=[]
        while not self.done.wait(2):
            root=psutil.Process(os.getpid())
            for p in [root,*root.children(recursive=True)]:
                try:known[p.pid]=p.create_time()
                except psutil.NoSuchProcess:pass
            s=r.snapshot();foreign=[]
            for g in s['gpu']:
                for p in g['processes']:
                    try:owned=p['pid'] in known and psutil.Process(p['pid']).create_time()==known[p['pid']]
                    except psutil.NoSuchProcess:
                        # NVML may briefly retain a PID after its owned process exits.
                        owned=p['pid'] in known
                    if not owned:foreign.append(p)
            hidden=any(g['memory_mib']-sum(p['memory_mib'] for p in g['processes'])>512 for g in s['gpu'])
            hidden_count=hidden_count+1 if hidden else 0
            last.append(s);last=last[-3:]
            if foreign or hidden_count>=2:
                self.error=dict(tag=self.tag,samples=last,foreign=foreign,hidden=hidden,owned_pid_create_times=known,
                                action='Interrupt owned runner; do not continue sequence or touch foreign jobs')
                r.b.save(r.OUT/'STOP-external-gpu.json',self.error)
                os.kill(os.getpid(),signal.SIGINT);return
        r.b.save(r.OUT/(self.tag+'-guard-clean.json'),dict(ended=r.now(),owned_pid_create_times=known,last_samples=last))

def main():
    pynvml.nvmlInit();frozen=json.loads((r.OUT/'frozen.json').read_text());r.gate(frozen)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    with (r.ROOT/'logs/p7a/gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r.idle('resume-before')
        stop=r.OUT/'STOP-external-gpu.json';e=json.loads(stop.read_text())
        label=r.LABELS[0];folder=r.OUT/label
        with (folder/'gpu-rank-processes.csv').open() as f:owned={int(g['pid']) for g in csv.DictReader(f,skipinitialspace=True)}
        assert e['tag']==label and {p['pid'] for p in e['foreign']}==owned and not e['hidden']
        s=json.loads((r.ROOT/'results'/(label+'_summary.json')).read_text())
        from datetime import datetime
        incident=datetime.fromisoformat(e['snapshot']['utc']).timestamp()
        assert incident>s['run_started_unix']+s['duration_sec'] and s['completed']==555
        assert 'SIGTERM received' in (folder/'server.log').read_text()
        r.b.save(r.OUT/'cleanup-guard-false-alarm-review.json',dict(incident_utc=e['snapshot']['utc'],
            formal_finished_unix=s['run_started_unix']+s['duration_sec'],owned_scheduler_pids=sorted(owned),
            conclusion='All four reported PIDs are this run schedulers; parent shutdown/reparenting broke instantaneous descendant matching. No external occupancy.',
            correction='Track PID plus creation time across reparenting; do not rerun completed data.',
            resume_script_sha256=r.sha(Path(__file__)),original_frozen_runner_unchanged=True))
        stop.rename(r.OUT/'cleanup-guard-false-alarm.json')
        # Preserve incomplete shutdown metadata before recording the independently verified cleanup.
        (folder/'run-before-cleanup-recovery.json').write_bytes((folder/'run.json').read_bytes())
        m=json.loads((folder/'run.json').read_text())
        m.update(status='PASS_WITH_CLEANUP_GUARD_FALSE_ALARM',finished_utc=r.now(),cleanup_recovery=str(r.OUT/'cleanup-guard-false-alarm-review.json'))
        r.b.save(folder/'run.json',m);(folder/'gpu-cleanup.txt').write_text(r.b.gpu());r.idle(label+'-after')
        r.Guard=Guard
        assert r.LOCK.exists();r.LOCK.rename(r.OUT/'p10a_formal.lock.json')
        try:
            for label in r.LABELS[1:]:r.experiment(label,frozen)
        finally:(r.OUT/'p10a_formal.lock.json').rename(r.LOCK)
        r.b.save(r.OUT/'batch-finished.json',dict(utc=r.now(),labels=r.LABELS,restarts=0,continued_after_owned_cleanup_false_alarm=True))
        r.idle('formal-final');r.gate(frozen)

if __name__=='__main__':main()
