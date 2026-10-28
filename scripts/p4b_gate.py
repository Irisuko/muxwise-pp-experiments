import subprocess, time, pathlib, json, datetime
root=pathlib.Path('/root/workspace/muxpp_work')
with (root/'logs/p4b/gpu-gate.jsonl').open('a', buffering=1) as f:
 while True:
  pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  done=(root/'logs/p4a.rc').exists()
  f.write(json.dumps(dict(at=datetime.datetime.now().isoformat(),p4a_done=done,pids=pids))+'\n')
  if done and not pids: break
  time.sleep(120)
