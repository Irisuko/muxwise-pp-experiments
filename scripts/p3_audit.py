#!/usr/bin/env python3
"""Read-only audit of P3b results; never launches a model or benchmark."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import psutil

ROOT = Path('/root/workspace/muxpp_work')
LABELS = ['chunked', 'muxwise', 'sglang_pd', 'muxwise_pp']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    trace = ROOT / 'bench/toolagent_600_240_x2.5.jsonl'
    records = {r['idx']: r for r in map(json.loads, trace.open())}
    checks = []

    def check(name, value):
        checks.append({'name': name, 'passed': bool(value)})

    check('trace_sha256_unchanged', sha(trace) == (ROOT / 'logs/p3/setup/trace-sha256.txt').read_text().strip())
    check('trace_has_1551_unique_requests', len(records) == 1551)
    check('exactly_four_formal_markers', sorted(p.name for p in (ROOT / 'results').glob('*.formal-started.json')) == sorted(f'{l}.formal-started.json' for l in LABELS))
    runs = {}
    previous_end = 0
    previous_closed = ''
    for label in LABELS:
        marker = read(ROOT / f'results/{label}.formal-started.json')
        directory = Path(marker['log_dir'])
        summary_path = ROOT / f'results/{label}_summary.json'
        details_path = ROOT / f'results/{label}_requests.jsonl'
        summary = read(summary_path)
        rows = list(map(json.loads, details_path.open()))
        meta = read(directory / 'run.json')
        jobs = read(directory / 'commands.json')
        successes = [r for r in rows if r['success']]
        check(label + ':one_formal_command', len(list((ROOT / 'logs/p3' / label).glob('*/formal-command.json'))) == 1)
        check(label + ':marker_command_matches', marker['command'] == read(directory / 'formal-command.json'))
        check(label + ':all_requests_sent_once', len(rows) == len({r['idx'] for r in rows}) == 1551 and {r['idx'] for r in rows} == set(records))
        check(label + ':requests_match_trace', all(all(r[k] == records[r['idx']][k] for k in ['arrival', 'input_len', 'output_len']) for r in rows))
        check(label + ':unmodified_formal_options', summary['trace'] == str(trace) and summary['time_scale'] == 1 and summary['limit'] is None and summary['warmup'] == 2 and summary['force_close'] and summary['timeout_sec'] == 3600 and not summary['interrupted'])
        check(label + ':counts_match', summary['planned_requests'] == summary['sent_requests'] == 1551 and summary['unsent_requests'] == 0 and summary['completed_requests'] == len(successes) and summary['failed_requests'] == len(rows) - len(successes))
        check(label + ':successful_outputs_complete', all(r['output_tokens'] == r['output_len'] and r['done_received'] and r['http_status'] == 200 for r in successes))
        passed = sum(r['ttft'] <= 5 and r['tpot'] is not None and r['tpot'] <= .1 for r in successes)
        check(label + ':joint_slo_recomputed', summary['slo']['ttft_le_5s_and_tpot_le_100ms']['passed'] == passed and summary['slo']['ttft_le_5s_and_tpot_le_100ms']['total'] == 1551)
        check(label + ':throughput_recomputed', all(summary[k] == sum(r[field] for r in successes) for k, field in [('input_tokens', 'input_len'), ('output_tokens', 'output_tokens')]) and abs(summary['throughput']['output_tokens_per_sec'] - sum(r['output_tokens'] for r in successes) / summary['duration_sec']) < 1e-10)
        check(label + ':chronological_order_and_no_overlap', summary['run_started_unix'] > previous_end and meta['started_utc'] > previous_closed)
        previous_end = summary['run_started_unix'] + summary['duration_sec']
        previous_closed = meta['finished_utc']
        configs = {}
        for name, info in meta['server_info'].items():
            expected = {'model_path': '/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct', 'dtype': 'bfloat16', 'context_length': 33792, 'disable_overlap_schedule': True, 'schedule_policy': 'fcfs', 'tp_size': 1, 'pp_size': 4 if label == 'muxwise_pp' else 1, 'disable_radix_cache': label == 'sglang_pd' and name in ['worker2', 'worker3']}
            check(label + ':' + name + ':common_config', all(info.get(k) == v for k, v in expected.items()))
            configs[name] = {k: info.get(k) for k in [*expected, 'mem_fraction_static', 'max_total_num_tokens', 'chunked_prefill_size', 'attention_backend', 'disable_cuda_graph', 'disaggregation_mode']}
        visible = {int(g) for job in jobs if job['name'] != 'router' for g in job['env']['CUDA_VISIBLE_DEVICES'].split(',')}
        check(label + ':all_four_gpus', visible == {0, 1, 2, 3})
        smoke = read(directory / 'smoke' / f'{label}_smoke_summary.json')
        check(label + ':smoke_passed', smoke['completed_requests'] == 20 and smoke['failed_requests'] == 0 and smoke['duration_sec'] <= 60 and smoke['time_scale'] == .2 and smoke['limit'] == 20 and smoke['force_close'])
        warmups = read(directory / 'long-warmup.json')
        check(label + ':long_warmup_passed', all(r['input_len'] == 31663 and r['response']['meta_info']['completion_tokens'] == 8 for r in warmups))
        timestamps = collections.defaultdict(list)
        for line in (directory / 'gpu.csv').read_text().splitlines()[1:]:
            fields = line.split(',')
            timestamps[int(fields[1])].append(datetime.datetime.strptime(fields[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp())
        intervals = {g: [b - a for a, b in zip(ts, ts[1:])] for g, ts in timestamps.items()}
        check(label + ':gpu_sampling', set(timestamps) == {0, 1, 2, 3} and all(4.9 < statistics.median(v) < 5.5 for v in intervals.values()))
        runs[label] = {'start_utc': summary['run_started_at'], 'duration_sec': summary['duration_sec'], 'closed_utc': meta['finished_utc'], 'completed': len(successes), 'failed': len(rows) - len(successes), 'joint_slo_passed': passed, 'formal_returncode': meta['formal_returncode'], 'log_dir': str(directory), 'summary_sha256': sha(summary_path), 'details_sha256': sha(details_path), 'gpu_sampling_interval_median_sec': {g: statistics.median(v) for g, v in intervals.items()}, 'gpu_sampling_interval_max_sec': {g: max(v) for g, v in intervals.items()}, 'actual_server_config': configs}
    pd_dir = Path(read(ROOT / 'results/sglang_pd.formal-started.json')['log_dir'])
    correct = read(pd_dir / 'correctness-comparison.json')
    check('pd_exact_output_match_20', correct['passed'] == correct['total'] == 20 and all(correct['matches']))
    commit = command(['git', '-C', '/root/workspace/sglang', 'rev-parse', 'HEAD'])
    status = command(['git', '-C', '/root/workspace/sglang', 'status', '--porcelain'])
    check('source_commit_unchanged', commit == 'bd849d628cd6332defee24945ce71db3fa662f90')
    check('source_tree_clean', not status)
    gpu_processes = command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'])
    check('gpu_processes_clean', not gpu_processes)
    servers = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        args = p.info['cmdline'] or []
        if any(a in ['sglang.launch_server', 'sglang_router.launch_router'] or a.startswith('sglang::') for a in args):
            servers.append(p.info)
    check('server_processes_clean', not servers)
    final_gpu = command(['nvidia-smi'])
    (ROOT / 'logs/p3/final-gpu.txt').write_text(final_gpu + '\n')
    result = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL', 'trace_sha256': sha(trace), 'source_commit': commit, 'source_git_status': status, 'checks': checks, 'runs': runs, 'remaining_gpu_processes': gpu_processes, 'remaining_server_processes': servers}
    (ROOT / 'logs/p3/final-audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'failures': [c['name'] for c in checks if not c['passed']]}))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
