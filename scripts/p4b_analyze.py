#!/usr/bin/env python3
"""Read-only development run summaries, GPU samples and lane accounting."""
import ast
import collections
import csv
import datetime
import json
import pathlib
import re
import sys


def stage_delta(first, last):
    result = {}
    def delta(a, b):
        return {k: b.get(k, 0) - a.get(k, 0) for k in a.keys() | b.keys()}
    for i in range(4):
        a, b = first[i]['timing'], last[i]['timing']
        elapsed = b['elapsed'] - a['elapsed']
        wall = {lane: delta(a['wall'][lane], b['wall'][lane]) for lane in a['wall']}
        cuda = delta(a['cuda'], b['cuda'])
        parts = delta(a['partitions'], b['partitions'])
        result[i] = {
            'seconds': elapsed,
            'wall_percent': {lane: {k: 100*v/elapsed for k,v in vs.items()} for lane,vs in wall.items()},
            'cuda_percent': {k: 100*v/elapsed for k,v in cuda.items()},
            'partition_percent': {k: 100*v/elapsed for k,v in parts.items()},
            'mape': last[i]['mape'],
            'prefill_slice_mape': last[i].get('prefill_slice_mape'),
            'prefill_legacy_batch_mape': last[i].get('prefill_legacy_batch_mape'),
            'slices': b.get('slices', {}),
        }
    return result


def gpu_stats(samples):
    result = {}
    for i, vals in samples.items():
        result[i] = {'util_mean': sum(v[0] for v in vals)/len(vals),
                     'memory_max_MiB': max(v[1] for v in vals), 'samples': len(vals)}
    return result


def analyze(path):
    summary = json.loads(next(path.glob('*_summary.json')).read_text())
    start = summary['run_started_unix']
    end = start + summary['duration_sec']
    arrival_end = start + summary['scheduled_duration_sec']
    result = {'summary': summary}
    gpu = collections.defaultdict(list)
    gpu_arrival = collections.defaultdict(list)
    for row in csv.reader((path/'gpu.csv').open()):
        try:
            at = datetime.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()
        except ValueError:
            continue
        if start <= at <= end:
            value = [int(row[4].split()[0]), int(row[2].split()[0])]
            gpu[int(row[1])].append(value)
            if at <= arrival_end:
                gpu_arrival[int(row[1])].append(value)
    result['gpu'] = gpu_stats(gpu)
    result['gpu_arrival'] = gpu_stats(gpu_arrival)
    log = path/'pp4.log'
    if log.exists():
        samples = []
        for line in log.read_text().splitlines():
            if 'MuxWise-PP summary' not in line:
                continue
            at = datetime.datetime.strptime(line[1:20], '%Y-%m-%d %H:%M:%S').timestamp()
            data = ast.literal_eval(re.search(r'stages=(.*?) mbs=', line).group(1))
            samples.append((at, data, line))
        before = [v for v in samples if v[0] <= start]
        during = [v for v in samples if start <= v[0] <= end]
        result['summary_counts'] = {'samples': len(during), 'required_inf': sum('required=inf' in v[2] for v in during)}
        for key in ('budget', 'inflight'):
            result[key+'_summary_counts'] = dict(collections.Counter(int(re.search(key+r'=(\d+)', v[2]).group(1)) for v in during))
        result['partition_summary_counts'] = {i: dict(collections.Counter(v[1][i]['partition'] for v in during)) for i in range(4)}
        if len(during) > 1 and 'timing' in during[-1][1][0]:
            first = before[-1] if before else during[0]
            result['timing_window_unix'] = [first[0], during[-1][0]]
            result['timing'] = stage_delta(first[1], during[-1][1])
            result['timing_strict_window_unix'] = [during[0][0], during[-1][0]]
            result['timing_strict'] = stage_delta(during[0][1], during[-1][1])
            active = [v for v in during if first[0] < v[0] <= arrival_end]
            if active:
                result['timing_arrival_window_unix'] = [first[0], active[-1][0]]
                result['timing_arrival'] = stage_delta(first[1], active[-1][1])
                if active[-1][0] > during[0][0]:
                    result['timing_arrival_strict_window_unix'] = [during[0][0], active[-1][0]]
                    result['timing_arrival_strict'] = stage_delta(during[0][1], active[-1][1])
    (path/'analysis.json').write_text(json.dumps(result, indent=2))
    print(path.name, 'analyzed', summary['completed_requests'], 'completed')
    return result


if __name__ == '__main__':
    for name in sys.argv[1:]:
        analyze(pathlib.Path(name))
