"""Collect the P2 acceptance evidence without modifying workload results."""
import json
import re
from pathlib import Path

ROOT = Path('/root/workspace/muxpp_work')
LOG = ROOT / 'logs/p2'
TAGS = ['p2-0-correctness-final', 'p2-0-stress-03', 'p2-0-control-final',
        'p1-baseline-300', 'p2-0-real-300', 'p2-correctness-serial', 'p2-correctness-final',
        'p2-correctness-migration', 'p2-logits-probe',
        'p2-alp-300-transport', 'p2-static-300-transport', 'p2-control',
        'p2-pdmux-regression', 'p2-pp-regression', 'p2-ablation-off']
report = {'runs': {}, 'cpu_tests': 18, 'bench_cpu_tests': 7}
for tag in TAGS:
    result = json.loads((LOG / f'{tag}-result.json').read_text())
    assert result['status'] == 'PASS', tag
    log = (LOG / f'{tag}-server.log').read_text()
    row = {'status': result['status'], 'command': result['command']}
    if result['mode'] == 'muxpp':
        idle = {}
        for stage, free, evict, reqs in re.findall(
            r'idle self-check PASS stage=(\d+) free_kv=(\d+) evictable=(\d+) free_reqs=(\d+)', log):
            idle[int(stage)] = dict(free_kv=int(free), evictable=int(evict), free_reqs=int(reqs))
        assert set(idle) == set(range(4)), (tag,idle)
        assert all(v['free_reqs'] == 128 for v in idle.values()), (tag,idle)
        assert len({v['free_kv']+v['evictable'] for v in idle.values()}) == 1
        admissions, migrations = {}, {}
        for stage in range(4):
            admissions[stage] = re.findall(rf'PP{stage}\] MuxWise-PP admit (.*)', log)
            migrations[stage] = re.findall(rf'PP{stage}\] MuxWise-PP migrate (.*)', log)
        assert all(admissions[i] == admissions[0] for i in range(4)), tag
        assert all(migrations[i] == migrations[0] for i in range(4)), tag
        row.update(idle=idle, admission_batches=len(admissions[0]), migrations=len(migrations[0]))
    summary = LOG / f'{tag}_summary.json'
    if summary.exists():
        d = json.loads(summary.read_text())
        assert d['completed_requests'] == d['planned_requests'] and d['failed_requests'] == 0, tag
        row['benchmark'] = {k: d[k] for k in ['completed_requests','failed_requests','duration_sec',
                                             'input_tokens','output_tokens','throughput','latencies_sec','cached_tokens']}
    report['runs'][tag] = row
reference = json.loads((ROOT / 'logs/p1/normal-correctness-result.json').read_text())['outputs']
actual = json.loads((LOG / 'p2-correctness-serial-result.json').read_text())['outputs']
comparison = {'requests': len(actual), 'equal': [], 'first_difference': []}
for a,b in zip(reference, actual):
    comparison['equal'].append(a['output_ids'] == b['output_ids'])
    comparison['first_difference'].append(next((i for i,(x,y) in enumerate(zip(a['output_ids'], b['output_ids'])) if x != y), None))
assert len(actual)==len(reference)==20 and all(comparison['equal'])
assert report['runs']['p2-logits-probe']['migrations'] > 0
comparison['tokens_equal'] = sum(len(a['output_ids']) for a in actual)
comparison['migration_probe_commands_per_stage'] = report['runs']['p2-logits-probe']['migrations']
report['correctness'] = comparison
report['concurrent_repeat_equal_requests'] = 19
report['bitwise_batch_invariant'] = False
report['numerical_limitation_evidence'] = ['p2-correctness-final-comparison.json','p1-fixed-batch-probe-server.log']
(LOG/'acceptance-summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'correctness':comparison, 'runs':len(report['runs'])},indent=2))
