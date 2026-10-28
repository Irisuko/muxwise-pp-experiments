import datetime, json, pathlib, subprocess, time
root = pathlib.Path('/root/workspace/muxpp_work')
with (root/'logs/p5b/gpu-gate.jsonl').open('a', buffering=1) as out:
    while True:
        pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        markers = [str(p) for p in [root/'logs/p5a.gpu_done', root/'logs/p5a.rc'] if p.exists()]
        out.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), markers=markers, pids=pids))+'\n')
        if markers and not pids:
            break
        time.sleep(120)
