"""Opt-in, read-only runtime probes for the borderline 128-token test prompt."""
import os
if os.environ.get('PDMUX_PP_LOGITS_PROBE') == '1':
    import json
    from sglang.srt.managers.tp_worker import TpModelWorker
    original = TpModelWorker.forward_batch_generation
    def probed(self, batch, *args, **kwargs):
        result = original(self, batch, *args, **kwargs)
        if self.pp_group.is_last_rank:
            reqs = batch.sampling_info.penalizer_orchestrator.batch.reqs
            for i, req in enumerate(reqs):
                if len(req.origin_input_ids) == 128 and len(req.output_ids) in (33, 34):
                    logits = result[0].next_token_logits[i]
                    values, ids = logits.float().topk(5)
                    row = dict(rid=req.rid, step=len(req.output_ids), bs=len(reqs),
                               seq_len=int(batch.seq_lens_cpu[i]), req_pool_idx=req.req_pool_idx,
                               tensor_pool_idx=int(batch.req_pool_indices[i]),
                               previous=req.output_ids[-1], tensor_input=int(batch.input_ids[i]),
                               dtype=str(logits.dtype), top_ids=ids.tolist(), top_logits=values.tolist(),
                               incomplete=float(logits[33243]), just=float(logits[1120]))
                    assert row['req_pool_idx'] == row['tensor_pool_idx']
                    assert row['previous'] == row['tensor_input']
                    assert row['seq_len'] == 128 + row['step']
                    print('PP_LOGITS_PROBE '+json.dumps(row), flush=True)
        return result
    TpModelWorker.forward_batch_generation = probed
