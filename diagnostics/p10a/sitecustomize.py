"""Opt-in signal stack dumps for P10a stalled CUDA workers."""
import faulthandler, os, signal
from pathlib import Path
_path=Path('/root/workspace/muxpp_work/logs/p10a/stack-dumps')
_path.mkdir(exist_ok=True)
_p10_stack=(_path/f'stacks-{os.getpid()}.log').open('a',buffering=1)
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, file=_p10_stack, all_threads=True)

def _state(signum, frame):
    import json, sys, time
    frames=list(sys._current_frames().values())
    for top in frames:
        f=top
        while f:
            if f.f_code.co_name=='event_loop_pdmux_pp':
                s=f.f_locals['self'];t=s._pp_decode_transport
                data=dict(time=time.time(),rank=s.pp_rank,step=f.f_locals.get('step'),mb=f.f_locals.get('mb_id'),
                    partition=s._pp_stream_idx,flight=f.f_locals.get('flight_steps'),
                    streams=[s._pp_prefill_stream.query(),s._pp_decode_stream.query(),t.rx_stream.query(),t.tx_stream.query()],
                    prefill=[dict(bid=k,state=w.state,layer=w.batch.split_index,event=None if w.event is None else w.event.query(),launched=None if w.launched is None else w.launched.is_set()) for k,w in s._pp_prefills.items()],
                    rx_free=[v.is_set() for v in t.rx_free],
                    graphs=[(idx,bs,[e.query() for e in ev]) for idx,bs,ev in getattr(s.tp_worker.model_runner.graph_runner,'_p10_events',[])],
                    tx_work=[None if v is None else dict(launched=v.launched.is_set(),done=[w.is_completed() for w in v.works]) for v in t.tx_work],
                    pending_sends=list(s.prefill_mailbox.pending_sends),
                    envelopes=list(s._pp_envelopes))
                _p10_stack.write('STATE '+json.dumps(data)+'\n');_p10_stack.flush()
                break
            if f.f_code.co_name=='_receive_controls' and 'transfer' in f.f_locals:
                v=f.f_locals['transfer'];packet=f.f_locals.get('packet')
                _p10_stack.write('RX '+json.dumps(dict(step=getattr(packet,'step',None),done=None if v is None else [w.is_completed() for w in v.works]))+'\n');_p10_stack.flush()
            f=f.f_back
signal.signal(signal.SIGUSR2,_state)
