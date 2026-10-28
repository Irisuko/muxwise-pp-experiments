#!/usr/bin/env python3
"""Read-only P4a provenance, gate, trace and cleanup audit."""
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import psutil

ROOT = Path('/root/workspace/muxpp_work')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})

    before = read(ROOT/'logs/p4a/diagnosis/original-sha256.json')
    changed = [p for p, h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p) != h]
    check('all_original_raw_results_summaries_markers_and_p3_logs_unchanged', not changed)
    check('formal_markers_are_original_four_plus_one_r2',
          {p.name for p in (ROOT/'results').glob('*.formal-started.json')} ==
          {label+'.formal-started.json' for label in ['chunked', 'muxwise', 'sglang_pd', 'muxwise_pp', 'sglang_pd_r2']})
    marker = read(ROOT/'results/sglang_pd_r2.formal-started.json')
    directory = Path(marker['log_dir'])
    meta = read(directory/'run.json')
    gate_dir = Path(meta['gate'])
    gate = read(gate_dir/'run.json')
    check('one_r2_formal_command', len(list((ROOT/'logs/p4a').glob('*/formal-command.json'))) == 1)
    check('formal_command_matches_marker', marker['command'] == read(directory/'formal-command.json'))
    check('configuration_matches_passed_gate', gate['status'] == 'PASS' and gate['configuration_sha256'] == meta['configuration_sha256'])
    trace = list(map(json.loads, (ROOT/'bench/toolagent_600_240_x2.5.jsonl').open()))
    check('original_trace_unchanged', sha(ROOT/'bench/toolagent_600_240_x2.5.jsonl') == (ROOT/'logs/p3/setup/trace-sha256.txt').read_text().strip())
    for name, path, expected in [('gate', gate_dir/'precheck300', trace[:300]), ('formal', ROOT/'results', trace)]:
        label = 'precheck300' if name == 'gate' else 'sglang_pd_r2'
        summary = read(path/f'{label}_summary.json')
        rows = list(map(json.loads, (path/f'{label}_requests.jsonl').open()))
        by_id = {r['idx']: r for r in expected}
        success = [r for r in rows if r['success']]
        check(name+':every_expected_request_once', len(rows) == len({r['idx'] for r in rows}) == len(expected) and {r['idx'] for r in rows} == set(by_id))
        check(name+':trace_lengths_and_arrivals_match', all(all(r[k] == by_id[r['idx']][k] for k in ['input_len', 'output_len', 'arrival']) for r in rows))
        check(name+':client_settings', summary['time_scale'] == 1 and summary['warmup'] == 2 and summary['force_close'] and summary['timeout_sec'] == 3600 and not summary['interrupted'] and summary['limit'] == (300 if name == 'gate' else None))
        check(name+':counts_match', summary['completed_requests'] == len(success) and summary['failed_requests'] == len(rows)-len(success) and summary['unsent_requests'] == 0)
        check(name+':successful_outputs_complete', all(r['output_tokens'] == r['output_len'] and r['done_received'] and r['http_status'] == 200 for r in success))
        check(name+':throughput_recomputed', summary['output_tokens'] == sum(r['output_tokens'] for r in success) and abs(summary['throughput']['output_tokens_per_sec'] - summary['output_tokens']/summary['duration_sec']) < 1e-10)
        passed = sum(r['ttft'] <= 5 and r['tpot'] is not None and r['tpot'] <= .1 for r in success)
        check(name+':joint_slo_recomputed', summary['slo']['ttft_le_5s_and_tpot_le_100ms']['passed'] == passed)
        if name == 'gate':
            check('gate:300_success_zero_failure', len(success) == 300 and summary['failed_requests'] == 0)
    for name, path in [('gate', gate_dir), ('formal', directory)]:
        correctness = read(path/'correctness-comparison.json')
        check(name+':20_exact_matches', correctness['passed'] == correctness['total'] == 20 and all(correctness['matches']))
        warmup = read(path/'long-warmup.json')
        check(name+':two_31663_token_successes', len(warmup) == 2 and all(r['input_len'] == 31663 and r['response']['meta_info']['completion_tokens'] == 8 for r in warmup))
    original_info = read(ROOT/'logs/p3/sglang_pd/formal-mini/run.json')['server_info']
    common_keys = ['model_path', 'dtype', 'context_length', 'disable_overlap_schedule', 'schedule_policy', 'tp_size', 'pp_size', 'disable_radix_cache', 'mem_fraction_static', 'chunked_prefill_size', 'attention_backend', 'disable_cuda_graph', 'max_running_requests', 'cuda_graph_max_bs']
    for worker, info in meta['server_info'].items():
        check(worker+':P3b_common_settings_preserved', all(info.get(k) == original_info[worker].get(k) for k in common_keys))
    sample_times = {}
    for line in (directory/'gpu.csv').read_text().splitlines()[1:]:
        fields = line.split(',')
        sample_times.setdefault(int(fields[1]), []).append(datetime.datetime.strptime(fields[0].strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp())
    intervals = {gpu: statistics.median([b-a for a, b in zip(ts, ts[1:])]) for gpu, ts in sample_times.items()}
    check('four_gpus_sampled_every_five_seconds', set(intervals) == {0, 1, 2, 3} and all(4.9 < x < 5.5 for x in intervals.values()))
    source = meta['source_dir']
    source_commit = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    changed_source = subprocess.check_output(['git', '-C', source, 'diff', '--name-only', 'bd849d628cd6332defee24945ce71db3fa662f90', 'HEAD'], text=True).splitlines()
    check('inference_source_identical_to_P3b_only_PD_profile_added', changed_source == ['benchmark/pd_4090/p4a_profiles.json'])
    check('isolated_source_tree_clean_and_matches_manifest', source_commit == meta['git_commit'] and not subprocess.check_output(['git', '-C', source, 'status', '--porcelain'], text=True).strip())
    gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True).strip()
    check('gpu_processes_clean', not gpu_processes)
    servers = []
    for proc in psutil.process_iter(['pid', 'cmdline']):
        args = proc.info['cmdline'] or []
        if any(x in args for x in ['sglang.launch_server', 'sglang_router.launch_router']) or (args and args[0].startswith('sglang::')):
            servers.append(proc.info)
    check('service_processes_clean', not servers)
    output = {'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL',
              'checks': checks, 'changed_original_files': changed, 'gpu_processes': gpu_processes,
              'service_processes': servers, 'gpu_sample_median_intervals': intervals,
              'formal_log_dir': str(directory), 'gate_dir': str(gate_dir),
              'source_dir': source, 'source_commit': source_commit}
    (ROOT/'logs/p4a/final-audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(output['status'], f'{sum(c["passed"] for c in checks)}/{len(checks)} checks')
    for c in checks:
        if not c['passed']:
            print(c)
    assert output['status'] == 'PASS'


if __name__ == '__main__':
    main()
