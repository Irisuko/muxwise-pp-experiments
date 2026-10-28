"""P1 server lifecycle, token equality and PP stress/regression evidence."""
import argparse
import hashlib
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import requests

ROOT = Path('/root/workspace/muxpp_work')
LOG = ROOT / 'logs/p9a/regressions'
LOG.mkdir(parents=True, exist_ok=True)
MODEL = '/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct'
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['normal', 'pdmux', 'pp', 'muxpp'])
p.add_argument('--suite', choices=['smoke', 'correctness', 'stress', 'control'], default='smoke')
p.add_argument('--tag')
p.add_argument('--tp', type=int, default=1)
p.add_argument('--pp', type=int)
p.add_argument('--long-warmup', action='store_true')
p.add_argument('--prompt-indices')
p.add_argument('--batch-request', action='store_true')
p.add_argument('--cold-cache', action='store_true', help='Flush before each serial correctness request')
p.add_argument('--force-close', action='store_true')
p.add_argument('--concurrency', type=int, default=1)
p.add_argument('--limit', type=int, default=200)
p.add_argument('--time-scale', type=float, default=0.02)
p.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
a = p.parse_args()
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
tag = a.tag or f'{a.mode}-{a.suite}'
BASE = 'http://127.0.0.1:30000'
if any(c.status == psutil.CONN_LISTEN and c.laddr.port == 30000 for c in psutil.net_connections(kind='tcp')):
    raise RuntimeError('port 30000 occupied')
cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', MODEL,
       '--tp', str(a.tp), '--port', '30000', '--disable-overlap-schedule',
       '--context-length', '33792', '--mem-fraction-static', '0.65' if a.mode == 'muxpp' else '0.8',
       '--cuda-graph-max-bs', '32', '--max-running-requests', '128',
       '--watchdog-timeout', '180', '--log-level', 'debug']
if a.mode in ('pp', 'muxpp'):
    cmd += ['--pp-size', '4' if a.mode == 'muxpp' else '2']
if a.mode in ('pdmux', 'muxpp'):
    cmd += ['--enable-pdmux', '--chunked-prefill-size', '-1', '--pdmux-config-path',
            str(ROOT / 'configs' / ('pdmux_pp_4090.yml' if a.mode == 'muxpp' else 'pdmux_4090.yml'))]
if a.suite == 'correctness':
    cmd += ['--skip-tokenizer-init']
if a.pp: cmd += ['--pp-size', str(a.pp)]
if a.tp > 1: cmd += ['--disable-custom-all-reduce']
cmd += ['--attention-backend', 'flashinfer']
cmd += a.extra
gpus = a.tp * (a.pp or (4 if a.mode == 'muxpp' else 2 if a.mode == 'pp' else 1))
env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, range(gpus))))
result = {'command': cmd, 'mode': a.mode, 'suite': a.suite, 'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'], 'cold_cache': a.cold_cache}
source = Path(os.getenv('P5B_SOURCE', '/root/workspace/sglang'))
if os.getenv('P5B_SOURCE'): env['PYTHONPATH'] = str(source / 'python')
result['commit'] = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
result['source_sha256'] = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'python/sglang/srt/multiplex/pp_decode.py', source/'python/sglang/srt/multiplex/multiplexing_pp.py', source/'python/sglang/srt/models/llama.py', source/'python/sglang/srt/managers/scheduler.py'] if p.exists()}
indices = [i for i, value in enumerate(cmd) if value == '--pdmux-config-path']
if indices:
    result['pdmux_config'] = Path(cmd[indices[-1] + 1]).read_text()
(LOG / f'{tag}-source.patch').write_bytes(subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--binary']))
assert not (LOG / f'{tag}-server.log').exists(), 'Never overwrite evidence; choose a new tag'
log = open(LOG / f'{tag}-server.log', 'w', buffering=1)
proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(LOG / f'{tag}.pid').write_text(str(proc.pid))
gpu_log = open(LOG / f'{tag}-gpu.csv', 'w')
gpu_monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu', '--format=csv', '--loop=1'], stdout=gpu_log, stderr=subprocess.DEVNULL)
start = time.monotonic()
print('Launching', ' '.join(cmd), flush=True)

def generate(ids, n=64):
    resp = requests.post(BASE + '/generate', json={'input_ids': ids,
        'sampling_params': {'temperature': 0, 'max_new_tokens': n, 'ignore_eos': True}}, timeout=300)
    resp.raise_for_status()
    body = resp.json()
    assert body['meta_info']['completion_tokens'] == n, body
    return body

try:
    while time.monotonic() - start < 600:
        if proc.poll() is not None:
            raise RuntimeError(f'server exited {proc.returncode}')
        try:
            if requests.get(BASE + '/health', timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        raise TimeoutError('startup timeout')
    result['startup_s'] = time.monotonic() - start
    result['gpu_ready'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv'], text=True)
    result['server_info'] = requests.get(BASE + '/get_server_info', timeout=30).json()
    print('Ready', result['startup_s'], flush=True)
    if a.long_warmup:
        import random
        rng = random.Random(730031)
        result['long_warmup'] = generate([128000] + [rng.randrange(128000) for _ in range(31662)], 8)
        requests.post(BASE + '/flush_cache', timeout=30).raise_for_status()
    if a.suite == 'smoke':
        result['outputs'] = [generate([128000, 3923, 374, 220, 17, 489, 220, 17, 30], 16)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            jobs = [([128000] + [791, 1401, 374, 264, 1296, 13] * n, 64) for n in [20, 150, 600, 900, 1200, 100]]
            result['outputs'] += list(pool.map(lambda job: generate(*job), jobs))
    elif a.suite == 'control':
        def cancellable(i, length):
            payload = {'rid': f'p1-abort-{i}', 'input_ids': [128000] + [791, 1401, 374, 264, 1296, 13] * length,
                       'sampling_params': {'temperature': 0, 'max_new_tokens': 1000, 'ignore_eos': True}}
            r = requests.post(BASE + '/generate', json=payload, timeout=120)
            return {'status': r.status_code, 'body': r.json()}
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures = []
            for i in range(4):
                futures.append(pool.submit(cancellable, i, 20 + i))
                time.sleep(0.15)
            for i in range(4, 16):
                futures.append(pool.submit(cancellable, i, 4500 + i))
            time.sleep(0.3)
            requests.post(BASE + '/abort_request', json={'rid': 'p1-abort-', 'abort_all': True}, timeout=20).raise_for_status()
            result['aborts'] = [f.result() for f in futures]
        for r in result['aborts']:
            assert r['status'] == 400 or r['body']['meta_info']['finish_reason']['type'] == 'abort', r
        time.sleep(2)
        response = requests.post(BASE + '/generate', json={'input_ids': [128000, 791, 1296], 'return_logprob': True,
                                  'sampling_params': {'max_new_tokens': 2}}, timeout=20)
        result['unsupported_status'] = response.status_code
        assert response.status_code == 400, response.text
        result['after_abort'] = generate([128000, 791, 1401, 374, 264, 1296, 13], 16)
        time.sleep(1)
        flushed = requests.post(BASE + '/flush_cache', timeout=20)
        result['flush_status'] = flushed.status_code
        assert flushed.status_code == 200, flushed.text
    elif a.suite == 'correctness':
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        lengths = [100, 128, 256, 512, 768, 1000, 1500, 2000, 2500, 3000,
                   3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
        prompts = []
        for i, n in enumerate(lengths):
            # Four families share long prefixes; suffixes make the tasks distinct.
            base = tok.encode(('The garden contains apple and pear trees. Water the plants daily. '
                               if i % 4 == 0 else f'Report family {i % 4}: the robot records temperature and rainfall. ') * 1200,
                              add_special_tokens=False)
            suffix = tok.encode(f'\nSummarize report number {i} in two sentences.\nAnswer:', add_special_tokens=False)
            prompts.append([128000] + base[:n - len(suffix) - 1] + suffix)
        if a.prompt_indices:
            result['prompt_indices'] = [int(i) for i in a.prompt_indices.split(',')]
            prompts = [prompts[i] for i in result['prompt_indices']]
        (LOG / f'{tag}-prompts.json').write_text(json.dumps(prompts))
        if a.cold_cache:
            assert a.concurrency == 1 and not a.batch_request
            result['outputs'], result['cold_cache_flushes'] = [], []
            for ids in prompts:
                flushed = requests.post(BASE + '/flush_cache', timeout=30)
                flushed.raise_for_status()
                result['cold_cache_flushes'].append(flushed.text)
                body = generate(ids)
                assert body['meta_info'].get('cached_tokens') == 0, body['meta_info']
                result['outputs'].append(body)
        elif a.batch_request:
            response = requests.post(BASE + '/generate', json={'input_ids': prompts, 'sampling_params': {'temperature': 0, 'max_new_tokens': 64, 'ignore_eos': True}}, timeout=300)
            response.raise_for_status()
            result['outputs'] = response.json()
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                result['outputs'] = list(pool.map(generate, prompts))
        for i, (ids, body) in enumerate(zip(prompts, result['outputs'])):
            assert len(body['output_ids']) == 64
            print('correctness', i, len(ids), 'cached', body['meta_info'].get('cached_tokens'), flush=True)
    else:
        bench_cmd = [sys.executable, str(ROOT / 'bench/bench_client.py'), '--trace',
                     str(ROOT / 'bench/dev_toolagent_1200_240_x2.5.jsonl'), '--base-url', BASE,
                     '--label', tag, '--out-dir', str(LOG), '--limit', str(a.limit),
                     '--time-scale', str(a.time_scale), '--warmup', '2', '--drain-timeout', '300', '--progress-interval', '15']
        if a.force_close:
            bench_cmd.append('--force-close')
        result['bench_command'] = bench_cmd
        subprocess.run(bench_cmd, check=True, timeout=1800)
        summary = json.loads((LOG / f"{tag}_summary.json").read_text())
        assert summary["completed_requests"] == a.limit and summary["failed_requests"] == 0, summary
    time.sleep(3)
    assert proc.poll() is None, 'server failed after workload'
    result['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv'], text=True)
    result['status'] = 'PASS'
except BaseException as exc:
    result['status'] = 'FAIL'
    result['error'] = repr(exc)
    raise
finally:
    gpu_monitor.terminate()
    gpu_monitor.wait(timeout=10)
    gpu_log.close()
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    for child in children:
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=10)
    result['total_s'] = time.monotonic() - start
    result['gpu_cleanup'] = subprocess.check_output(['nvidia-smi'], text=True)
    (LOG / f'{tag}-result.json').write_text(json.dumps(result, indent=2, default=str))
    print(result['status'], result.get('error', ''), flush=True)
