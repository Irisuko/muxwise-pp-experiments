"""Launch, exercise, and always stop one SGLang smoke server.

Run with env.sh sourced, e.g. python scripts/smoke.py pdmux.
Any extra CLI arguments are appended to the server command.
"""
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
MODEL = '/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct'
MODE = sys.argv[1]
assert MODE in {'normal', 'pdmux', 'pp'}
BASE = 'http://127.0.0.1:30000'
if any(c.status == psutil.CONN_LISTEN and c.laddr.port == 30000
       for c in psutil.net_connections(kind='tcp')):
    raise RuntimeError('Port 30000 is already occupied')
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1' if MODE == 'pp' else '0')
cmd = [sys.executable, '-m', 'sglang.launch_server', '--model-path', MODEL,
       '--tp', '1', '--port', '30000', '--disable-overlap-schedule']
if MODE == 'pdmux':
    cmd += ['--enable-pdmux', '--pdmux-config-path', str(ROOT / 'configs/pdmux_4090.yml'),
            '--chunked-prefill-size', '-1', '--mem-fraction-static', '0.8',
            '--log-level', 'debug']
if MODE == 'pp':
    cmd += ['--pp-size', '2']
cmd += sys.argv[2:]
result = {'mode': MODE, 'command': cmd, 'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'],
          'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'requests': []}
log = open(ROOT / f'logs/{MODE}-server.log', 'w', buffering=1)
print('Launching:', ' '.join(cmd), flush=True)
proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(ROOT / f'logs/{MODE}-server.pid').write_text(str(proc.pid) + '\n')
start = time.monotonic()

def generate(label, payload):
    t = time.monotonic()
    response = requests.post(BASE + '/generate', json=payload, timeout=600)
    response.raise_for_status()
    body = response.json()
    assert isinstance(body.get('text'), str) and body['text'].strip(), body
    assert body.get('meta_info', {}).get('completion_tokens', 0) > 0, body
    answer = {'label': label, 'elapsed_s': round(time.monotonic()-t, 3), 'response': body}
    print(json.dumps(answer, ensure_ascii=False), flush=True)
    return answer

try:
    deadline = start + 1200
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f'Server exited during startup: {proc.returncode}')
        try:
            if requests.get(BASE + '/health', timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Startup exceeded 1200 seconds')
        time.sleep(2)
    result['startup_s'] = round(time.monotonic() - start, 3)
    print('Server ready after', result['startup_s'], 'seconds', flush=True)
    result['server_info'] = requests.get(BASE + '/get_server_info', timeout=30).json()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    def payload(content, output_tokens=64):
        return {'text': tok.apply_chat_template([{'role': 'user', 'content': content}],
                       tokenize=False, add_generation_prompt=True),
                'sampling_params': {'temperature': 0, 'max_new_tokens': output_tokens}}

    result['requests'].append(generate('basic', payload('What is 2 + 2? Reply with the number only.')))
    assert '4' in result['requests'][-1]['response']['text']
    if MODE == 'pdmux':
        text = ('This is a test log entry. The garden contains apple trees and pear trees. ' * 1000)
        prompt_ids = tok.encode(text, add_special_tokens=False)[:5950]
        long_payload = payload(tok.decode(prompt_ids) + '\nSummarize the repeated log entry in one sentence.', 96)
        result['long_prompt_token_count'] = len(tok.encode(long_payload['text'], add_special_tokens=False))
        assert 5900 <= result['long_prompt_token_count'] <= 6100
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(generate, 'decode-seed', payload('Write a detailed story about a robot gardener.', 512))]
            time.sleep(0.15)
            jobs = [('long-6000', long_payload),
                    ('short-1', payload('Name three colors.')),
                    ('short-2', payload('What is the capital of France?')),
                    ('medium', payload('Explain the idea in this text: ' + 'A tree grows toward sunlight. ' * 200, 128)),
                    ('short-3', payload('Count from one to ten.'))]
            futures += [pool.submit(generate, name, p) for name, p in jobs]
            result['requests'] += [f.result() for f in futures]
        long_response = next(r for r in result['requests'] if r['label'] == 'long-6000')
        assert 5900 <= long_response['response']['meta_info']['prompt_tokens'] <= 6100
    result['status'] = 'PASS'
except BaseException as exc:
    result['status'] = 'FAIL'
    result['error'] = repr(exc)
    raise
finally:
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=20)
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
    log.close()
    result['total_elapsed_s'] = round(time.monotonic() - start, 3)
    result['server_exit_code'] = proc.returncode
    (ROOT / f'logs/{MODE}-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print('Result:', result['status'], flush=True)
