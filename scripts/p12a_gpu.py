"""P12a strict GPU guard, with bracketed NVML memory sampling.

Device and process memory counters are not atomic. cudaFree during flush_cache
can make their difference arbitrarily large even when every process is ours.
Read device memory on both sides of the process query; the minimum is a lower
bound on unaccounted memory during a release. Identified foreign compute or
utilization PIDs are never ignored or retried.
"""
import json
import time
import psutil
import pynvml as nv
import p11b_runner as original

def snapshot(known=None,since=None):
    known=known or {};rows=[]
    for i in range(nv.nvmlDeviceGetCount()):
        h=nv.nvmlDeviceGetHandleByIndex(i)
        before=nv.nvmlDeviceGetMemoryInfo(h,version=nv.nvmlMemory_v2)
        apps=[]
        for a in nv.nvmlDeviceGetComputeRunningProcesses(h):
            memory=a.usedGpuMemory/2**20 if a.usedGpuMemory<2**60 else 0
            try:owned=a.pid in known and psutil.Process(a.pid).create_time()==known[a.pid]
            except psutil.NoSuchProcess:owned=a.pid in known
            apps.append(dict(pid=a.pid,memory_mib=memory,owned=owned,visible=psutil.pid_exists(a.pid)))
        after=nv.nvmlDeviceGetMemoryInfo(h,version=nv.nvmlMemory_v2)
        samples=[];error=None
        try:
            for u in nv.nvmlDeviceGetProcessUtilization(h,(since or {}).get(i,int((time.time()-2)*1e6))):
                samples.append(dict(pid=u.pid,sm=u.smUtil,memory=u.memUtil,timestamp_us=u.timeStamp,owned=u.pid in known))
        except nv.NVMLError as exc:error=str(exc)
        owned_memory=sum(a['memory_mib'] for a in apps if a['owned'])
        rows.append(dict(index=i,uuid=nv.nvmlDeviceGetUUID(h),memory_mib=after.used/2**20,
            memory_before_mib=before.used/2**20,memory_changed_during_sample=before.used!=after.used,
            total_mib=after.total/2**20,free_mib=after.free/2**20,reserved_mib=after.reserved/2**20,
            utilization=nv.nvmlDeviceGetUtilizationRates(h).gpu,processes=apps,
            external_estimate_mib=max(0,min(before.used,after.used)/2**20-owned_memory),
            process_utilization=samples,process_utilization_error=error))
    return dict(utc=original.now(),unix=time.time(),gpu=rows)
