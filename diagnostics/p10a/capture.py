import psutil, signal, sys, time
from pathlib import Path
tag=sys.argv[1]
for p in psutil.process_iter(['cmdline']):
    a=p.info['cmdline'] or []
    if len(a)>2 and a[1].endswith('/p10a_runner.py') and a[-1]==tag:
        root=p;break
else:raise SystemExit('runner missing')
log=Path('/root/workspace/muxpp_work/logs/p10a')/tag/'smoke-client.log'
for _ in range(240):
    if log.exists() and '[    30.0s]' in log.read_text():break
    if not root.is_running():raise SystemExit('runner exited')
    time.sleep(1)
else:raise SystemExit('no smoke30')
for p in root.children(recursive=True):
    if 'scheduler' in p.name():
        print(p.pid,p.name(),flush=True)
        p.send_signal(signal.SIGUSR1)
        p.send_signal(signal.SIGUSR2)
