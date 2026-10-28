#!/usr/bin/env python3
"""Validate the exact HF-layout snapshot, then repair the RAM copy if needed."""
import argparse, datetime, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from safetensors import SafetensorError, safe_open

ROOT = Path('/root/workspace/muxpp_work')
NAME = 'Mistral-Small-24B-Instruct-2501'
DISK = Path('/root/autodl-pvt-A40-hdd-1/models') / NAME
RAM = Path('/dev/shm/models') / NAME
MANIFEST = ROOT / 'configs/p10a_model_manifest.json'
LOG = ROOT / 'logs/p10a'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def validate(path, files, full=False):
    for name, info in files.items():
        p = path / name
        if not p.is_file() or p.stat().st_size != info['size']:
            raise ValueError(f'Incomplete model file: {p}')
        if full and sha(p) != info['sha256']:
            raise ValueError(f'SHA256 mismatch: {p}')
    index = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
    seen = {}
    for name in sorted(set(index.values())):
        with safe_open(path/name, framework='pt', device='cpu') as f:
            for key in f.keys():
                # SafeTensors validates header, offsets and final file size on open.
                seen[key] = name
                f.get_slice(key).get_shape()
    if seen != index: raise ValueError('Tensor keys do not match the shard index')
    return len(seen)


def download(files):
    """Recover volatile RAM storage using only the pinned manifest allowlist."""
    RAM.mkdir(parents=True, exist_ok=True)
    pending = RAM/'.p10a-download-pending'
    pending.touch()
    # ModelScope can skip an existing file. Remove only known corrupt artifacts
    # so the final SHA256 validation cannot repeatedly encounter the same file.
    for name, info in files.items():
        path = RAM/name
        if path.exists() and (path.stat().st_size != info['size'] or sha(path) != info['sha256']):
            path.unlink()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_path = LOG/f'autodownload-{stamp}-{os.getpid()}.log'
    command = [str(Path(sys.executable).with_name('modelscope')), 'download',
               'mistralai/'+NAME, '--revision', 'master', '--local-dir', str(RAM),
               '--max-workers', '5', '--include', *files]
    env = dict(os.environ, MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS='8')
    print(f'No persistent model available; downloading to volatile RAM. Log: {log_path}', flush=True)
    with log_path.open('x') as log:
        log.write(json.dumps({'command':command,'range_workers':8,'manifest':str(MANIFEST)})+'\n')
        log.flush()
        for attempt in range(1, 4):
            log.write(f'\nAttempt {attempt}\n'); log.flush()
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode == 0:
                break
            if attempt == 3:
                raise RuntimeError(f'ModelScope download failed; see {log_path}')
            time.sleep(10)
        count = validate(RAM, files, full=True)
        pending.unlink(missing_ok=True)
        result = {'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'path':str(RAM), 'persistent':False, 'files':len(files),
                  'bytes':sum(v['size'] for v in files.values()), 'tensors':count,
                  'sha256_verified':True, 'manifest_sha256':sha(MANIFEST),
                  'download_log':str(log_path)}
        log.write('\n'+json.dumps(result)+'\n')
    (LOG/f'model-recovery-{stamp}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Downloaded model passed all SHA256 and safetensors checks', flush=True)


def ensure():
    LOG.mkdir(parents=True, exist_ok=True)
    with (LOG/'model-recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _ensure_locked()


def _ensure_locked():
    manifest = json.loads(MANIFEST.read_text())
    files = manifest['files']
    try:
        pending = RAM/'.p10a-download-pending'
        validate(RAM, files, full=pending.exists())
        pending.unlink(missing_ok=True)
        return RAM
    except (ValueError, OSError, KeyError, SafetensorError) as exc:
        print(f'RAM model needs repair: {exc}', flush=True)
    try:
        validate(DISK, files)
    except (ValueError, OSError, KeyError, SafetensorError) as exc:
        print(f'Persistent model unavailable: {exc}', flush=True)
        download(files)
        return RAM
    RAM.mkdir(parents=True, exist_ok=True)
    for name, info in files.items():
        dest = RAM/name
        if dest.is_file() and dest.stat().st_size == info['size'] and sha(dest) == info['sha256']:
            continue
        temp = dest.with_name(name+'.copying')
        shutil.copyfile(DISK/name, temp)
        if sha(temp) != info['sha256']: raise ValueError(f'Copy checksum mismatch: {temp}')
        temp.replace(dest)
    validate(RAM, files, full=True)
    return RAM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['manifest', 'ensure', 'validate', 'persist'])
    p.add_argument('--path', type=Path, default=RAM)
    a = p.parse_args()
    if a.mode == 'manifest':
        remote = json.loads((ROOT/'logs/p10a/modelscope-raw-files.json').read_text())
        wanted = [r for r in remote if r['Path'].startswith('model-') or r['Path'] in (
            'config.json','generation_config.json','model.safetensors.index.json',
            'tokenizer.json','tokenizer_config.json','special_tokens_map.json')]
        files = {}
        for row in wanted:
            file = a.path/row['Path']
            assert file.stat().st_size == row['Size'], row
            digest = sha(file)
            assert digest == row['Sha256'], f'ModelScope SHA256 mismatch: {file}'
            files[row['Path']] = {'size': row['Size'], 'sha256': digest,
                                  'source_file_revision': row['Revision']}
            print(row['Path'], files[row['Path']], flush=True)
        count = validate(a.path, files, full=False)
        MANIFEST.write_text(json.dumps({'source':'https://modelscope.cn/models/mistralai/'+NAME,
            'revision':'master','files':files,'tensor_count':count}, indent=2)+'\n')
    elif a.mode == 'ensure': print(ensure())
    elif a.mode == 'validate':
        print(validate(a.path, json.loads(MANIFEST.read_text())['files'], full=True))
    else:
        files = json.loads(MANIFEST.read_text())['files']
        validate(RAM, files)
        DISK.mkdir(parents=True, exist_ok=True)
        for name, info in files.items():
            dest = DISK/name
            if dest.exists() and dest.stat().st_size == info['size'] and sha(dest) == info['sha256']: continue
            temp = dest.with_name(name+'.copying')
            shutil.copyfile(RAM/name, temp)
            if sha(temp) != info['sha256']: raise ValueError(f'Copy checksum mismatch: {temp}')
            temp.replace(dest)
            print('persisted', name, flush=True)
        validate(DISK, files, full=True)
        shutil.copyfile(MANIFEST, DISK/'p10a_manifest.json')
        print('Persistent copy verified', DISK, flush=True)

if __name__ == '__main__': main()
