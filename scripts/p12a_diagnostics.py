#!/usr/bin/env python3
"""Summarize recorded P12a evidence without launching a server or client."""
import json
from pathlib import Path

OUT = Path('/root/workspace/muxpp_work/logs/p12a')


def inspect(path):
    a = json.loads(path.read_text())
    run = path.parent.parent
    meta = json.loads((run / 'run.json').read_text())
    s = a['summary']
    start, end = s['run_started_unix'], s['run_started_unix'] + s['duration_sec']
    flags = meta.get('flags', [])
    def flag(name):
        return int(flags[flags.index(name) + 1]) if name in flags else 0
    chunk = flag('--pdmux-pp-prefill-chunk-size')
    tile = flag('--pdmux-pp-prefill-projection-pad-to')
    parent_tokens = []
    for timeline in run.glob('requests-timeline*.jsonl'):
        for line in timeline.open():
            e = json.loads(line)
            if start <= e['at'] <= end and e['event'] == 'admit' and e['stage'] == 0:
                parent_tokens.append(e['tokens'])
    dummy = 0
    if chunk and tile:
        for n in parent_tokens:
            if n > tile:
                dummy += sum(max(tile - min(chunk, n - offset), 0)
                             for offset in range(0, n, chunk))
    large = 'big_' in run.name
    hidden, layers = (5120, 40) if large else (4096, 32)
    pp = 4 if 'pp4' in run.name else 2
    partition = meta.get('environment', {}).get('SGLANG_PP_LAYER_PARTITION')
    stage_layers = list(map(int, partition.split(','))) if partition else [layers // pp] * pp
    result = dict(run=run.name, rate=a['rate'], matched=a['parent_matched'],
                  decomposition_errors=a['decomposition_errors'],
                  mean_components_seconds={k: v['mean'] for k, v in a['component_distributions'].items()},
                  quantile_decomposition=a['quantile_decomposition'],
                  prefill_cuda_total_seconds={stage: data['cuda_slices'].get('n',0)*data['cuda_slices'].get('mean',0)
                                              for stage,data in a['timeline_diagnostics'].items()},
                  parent_batches=len(parent_tokens), new_prefill_tokens=sum(parent_tokens),
                  chunked_parents=sum(n > chunk for n in parent_tokens) if chunk else 0,
                  padded_o_rows=dummy,
                  stage_layers=stage_layers,
                  padded_o_interface_gib_per_stage_rank=(dummy * hidden * stage_layers[0] * 2 / 2**30)
                      if len(set(stage_layers)) == 1 else None,
                  padded_o_interface_gib_by_stage_rank={i: dummy * hidden * n * 2 / 2**30 for i,n in enumerate(stage_layers)},
                  payload_note='Modeled BF16 input volume per rank, not measured bus bytes; TP1 has no allreduce.',
                  workers={})
    for worker, window in a['worker_stage_windows']['full'].items():
        stages = {}
        for stage, data in window['stages'].items():
            wall = data['wall_seconds']
            stages[stage] = dict(window_seconds=data['seconds'], cuda=data['cuda_seconds'],
                                 prefill_lane=wall['prefill'], decode_lane=wall['decode'])
        result['workers'][worker] = dict(stages=stages, admission=window['admission'])
    return result


def main():
    records = [inspect(p) for p in sorted(OUT.glob('*/*/analysis.json'))
               if p.parent.name in ['partial', 'full']]
    (OUT / 'diagnostics.json').write_text(json.dumps(records, indent=2) + '\n')
    lines = ['| run | size | matched | admission / stage0 queue / stage0 span / downstream tail (s, mean) | new prefill tokens | dummy O rows |',
             '| --- | --- | ---: | --- | ---: | ---: |']
    for r in records:
        d = r['mean_components_seconds']
        means = [d.get(k, 0) for k in ['admission_queue', 'stage0_queue', 'stage0_parent_prefill_span']]
        means.append(sum(v for k, v in d.items() if k.endswith('_exposed_prefill_tail')))
        lines.append(f"| {r['run']} | {r['rate']} | {r['matched']} | {' / '.join(f'{v:.3f}' for v in means)} | {r['new_prefill_tokens']:,} | {r['padded_o_rows']:,} |")
    (OUT / 'diagnostics.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
