#!/usr/bin/env python3
"""Review all unchanged DP2 controls, including per-replica routing/cache mix."""
import json
from pathlib import Path
import statistics

OUT = Path('/root/workspace/muxpp_work/logs/p12a')
TAGS = ['ab-v5-llama_pp2dp2', 'r4-llama_pp2dp2',
        'control-repeat-v5-llama_pp2dp2', 'control-repeat-v51-llama_pp2dp2']


def main():
    result = []
    for tag in TAGS:
        path = OUT / tag / 'partial'
        rows = [json.loads(line) for line in (path / 'ttft-requests.jsonl').open()]
        raw = {v['idx']: v for v in map(json.loads, (path / 'partial_requests.jsonl').open())}
        assert len(rows) == len(raw) == 300
        workers = {}
        for worker in sorted({v['parent'][0] for v in rows}):
            subset = [v for v in rows if v['parent'][0] == worker]
            workers[worker] = dict(
                count=len(subset),
                cached_fraction=sum(v['cached_tokens'] for v in subset) / sum(v['input_len'] for v in subset),
                input_tokens=sum(v['input_len'] for v in subset),
                main_slo_pass=sum(raw[v['idx']]['ttft'] <= 5 and raw[v['idx']]['tpot'] <= .1 for v in subset),
                ttft_mean=statistics.mean(v['ttft'] for v in subset),
                ttft_median=statistics.median(v['ttft'] for v in subset),
                admission_mean=statistics.mean(v['components']['admission_queue'] for v in subset))
        result.append(dict(run=tag, workers=workers))
    (OUT / 'dp2-routing-review.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
