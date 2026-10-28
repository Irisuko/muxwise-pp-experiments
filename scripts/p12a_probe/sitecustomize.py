"""Opt-in host TP synchronization accounting, diagnostic runs only."""
import os
if os.getenv('P12A_TP_PROFILE'):
    import functools
    import json
    import time
    from collections import defaultdict
    from sglang.srt.multiplex.multiplexing_pp import SchedulerMultiplexPPMixin
    cls=SchedulerMultiplexPPMixin
    original=cls._pp_progress_prefill
    def flush(self):
        now=time.time()
        if now<getattr(self,'_p12_next',0):return
        if not hasattr(self,'_p12_file'):
            self._p12_file=open(os.environ['P12A_TP_PROFILE']+f'.pp{self.pp_rank}.tp{self.tp_rank}.jsonl','x',buffering=1)
        self._p12_file.write(json.dumps(dict(at=now,stage=self.pp_rank,tp=self.tp_rank,counters=self._p12_stats))+'\n')
        self._p12_next=now+5
    def wrap(name):
        f=getattr(cls,name)
        @functools.wraps(f)
        def measured(self,*args,**kwargs):
            if not hasattr(self,'_p12_stats'):self._p12_stats=defaultdict(lambda:[0,0.0,0.0])
            t=time.perf_counter()
            try:return f(self,*args,**kwargs)
            finally:
                elapsed=time.perf_counter()-t;v=self._p12_stats[name]
                v[0]+=1;v[1]+=elapsed;v[2]=max(v[2],elapsed)
                flush(self)
        setattr(cls,name,measured)
    for name in ['_pp_tp_all','_pp_tp_broadcast']:
        wrap(name)
