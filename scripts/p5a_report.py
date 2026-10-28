#!/usr/bin/env python3
"""Append only P5a observations, retaining every earlier result."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path("/root/workspace/muxpp_work")
LOG = ROOT / "logs/p5a"
CANDIDATE = "nixl-ipc-host-1m"


def read(p):
    return json.loads(p.read_text())


def fmt(x, factor=1):
    return "—" if x is None else f"{x*factor:.3f}"


def row(items):
    return "| " + " | ".join(map(str, items)) + " |"


def resources(d, s):
    samples = collections.defaultdict(list)
    start = s["run_started_unix"]
    end = start + s["duration_sec"]
    for line in (d/"gpu.csv").read_text().splitlines()[1:]:
        f = [x.strip() for x in line.split(",")]
        t = datetime.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        if start <= t <= end:
            samples[f[1]].append((float(f[2].split()[0]), float(f[4].split()[0])))
    return {k: {"mean_util_pct":statistics.mean(v[1] for v in a),
                "peak_mem_MiB":max(v[0] for v in a), "samples":len(a)}
            for k, a in sorted(samples.items())}


def main():
    base_dir = LOG/"dev-mooncake-r2"
    cand_dir = LOG/f"dev-{CANDIDATE}"
    base, cand = read(base_dir/"run.json"), read(cand_dir/"run.json")
    b, a = base["dev300_summary"], cand["dev300_summary"]
    ratio = a["throughput"]["input_tokens_per_sec"]/b["throughput"]["input_tokens_per_sec"]
    ttft_ratio = a["latencies_sec"]["ttft"]["p50"]/b["latencies_sec"]["ttft"]["p50"]
    gates = {}
    for d, run in [(base_dir, base), (cand_dir, cand)]:
        s = run["dev300_summary"]
        gates[d.name] = {
            "lifecycle_pass":run["status"] == "PASS",
            "text20":read(d/"correctness-comparison.json")["passed"],
            "long_successes":len(read(d/"long-warmup.json")),
            "completed":s["completed_requests"], "failed":s["failed_requests"],
            "unsent":s["unsent_requests"],
        }
    correct = all(g["lifecycle_pass"] and g["text20"] == 20 and g["long_successes"] == 2
                  and g["completed"] == 300 and g["failed"] == g["unsent"] == 0 for g in gates.values())
    performance = ratio >= 1.5 or ttft_ratio <= 0.5
    formal_path = ROOT/"results/sglang_pd_r3_summary.json"
    decision = {"candidate":CANDIDATE, "throughput_ratio":ratio, "ttft_p50_ratio":ttft_ratio,
                "correctness_pass":correct, "performance_pass":performance,
                "formal_allowed":correct and performance, "formal_run":formal_path.exists(),
                "gates":gates, "source_commit":cand["git_commit"],
                "rule":"Input throughput >=1.5x OR TTFT p50 <=0.5x, AND all correctness gates"}
    (LOG/"decision.json").write_text(json.dumps(decision, indent=2)+"\n")
    util = {d.name:resources(d, r["dev300_summary"]) for d, r in [(base_dir,base),(cand_dir,cand)]}
    (LOG/"resources.json").write_text(json.dumps(util, indent=2)+"\n")
    lines = [
        "", "## P5a：PD 传输带宽评估与 r3 决策", "",
        f"时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。隔离 worktree `/root/workspace/sglang-p4a`，开发冻结提交 `{cand['git_commit']}`。沿用 2P2D TP1（P=GPU0/1、D=GPU2/3）、P3b/P4a 模型、调度、显存和 CUDA graph 参数。新增可关闭的单调时钟日志及传输配置，没有修改模型推理、KV 内容或其它系统代码。", "",
        "环境为 Mooncake 0.3.13.post1、NIXL 1.5.0、其 CUDA12 wheel 内 UCX 1.23.0（revision 65e342a）。本次实际 `nvidia-smi topo -m` 和 CUDA UUID 对照显示 GPU0 在 NUMA0、GPU1/2/3 在 NUMA1，与背景/P4a文字中的编号分布不同；保留原 P0/1、D2/3，不改变设备顺序或绑核。P2P 全 CNS。证据：[environment.txt](logs/p5a/environment.txt)、[cuda-device-map.txt](logs/p5a/cuda-device-map.txt)。", "",
        "**微基准方法。** 每方案保持四个模型实例同时在线，对 0→2、0→3、1→2、1→3 四条路径逐条请求；每对先64-token预热，再测4096/16384/31663 tokens，每条前flush四后端cache，生成8 tokens。输入由固定随机种子生成，不读取正式或开发trace。各尺寸每路径一次，表中为四路径均值，不是重复实验置信区间。每token KV=32层×2(K/V)×8头×128维×2 bytes=131072 bytes。", "",
        "日志在每块 `kv_sender.send` 前（现有GPU索引到CPU同步之后）和decode轮询确认全部KV成功时记录同机 `monotonic_ns`，不新增CUDA同步。剩余等待=最后块发出前至decode收齐；完整窗口=首块发出前至decode收齐。4k只有一块，可直接观察发送/收齐窗口；16k/31k包含中间prefill计算和提交开销。有效带宽=完整KV bytes/完整窗口，**不是纯链路峰值**。不能把全量KV bytes除以“剩余等待”，否则会把已重叠传完的部分重复计入带宽。", "",
        row(["方案","tokens","prefill结束后剩余等待(s)","首块→收齐(s)","有效GB/s","请求E2E(s)"]),
        row(["---"]*6),
    ]
    order = ["mooncake-r2","nixl-host","nixl-host-1m","nixl-ipc-host-1m","mooncake-8m"]
    for name in order:
        d = LOG/f"micro-{name}"
        values = read(d/"micro-summary.json")
        for n in [4096,16384,31663]:
            arr = [x for x in values if x["tokens"] == n]
            lines.append(row([name,n]+[fmt(statistics.mean(x[k] for x in arr)) for k in
                ["prefill_complete_to_received_s","first_send_to_received_s","effective_total_kv_GBps","e2e_s"]]))
    lines += [
        "", "原始逐路径时间戳、输出、带宽、5秒GPU采样及完整启动命令均保存在各 `logs/p5a/micro-*/` 目录的 `micro-summary.json`、`micro-requests.json`、`gpu.csv`、`commands.json`。", "",
        row(["候选","关键环境","实测可用性"]),
        row(["---"]*3),
        row(["mooncake-r2","P4a mooncake-tuned 原参数，1MiB slice","四路径、三尺寸全部成功"]),
        row(["nixl-host","UCX_TLS=cuda_copy,sm,tcp,self；NET_DEVICES=lo","全部成功；默认缓冲慢"]),
        row(["nixl-host-1m","上述TLS；TCP_TX/RX_SEG_SIZE、POSIX/SYSV_SEG_SIZE均1M","全部成功；接近r2"]),
        row(["nixl-ipc-host-1m","增加cuda_ipc，其余同host-1m","全部成功；新候选中最快；开发日志intra-node的RMA为rma_am(tcp/lo)"]),
        row(["nixl-ipc-sm","cuda_copy,cuda_ipc,sm,self，无TCP","初始化失败：no active messages transport / no peer failure handler"]),
        row(["nixl-sm-1m","cuda_copy,sm,self；POSIX_ERROR_HANDLING=y；共享内存分片1M","64-token通过；首条4k时cuMemcpyAsync invalid argument → UCX fatal mem type unpack → abort/segfault，淘汰"]),
        row(["mooncake-8m","仅把MC_TCP_SLICE_SIZE从1MiB改8MiB","全部成功，但无带宽收益"]),
        "", "没有发现需要最小修补的 SGLang NIXL Python 层错误。共享内存失败记录定位到 UCX CUDA copy/unpack；本轮未冒险替换UCX或加入未经验证的自建传输后端。Mooncake的其它GPU直传协议不能仅凭同机就假定能规避本机P2P/RDMA缺失，因此未扩展该路线。", "",
        "开发轮开启C/Python日志立即刷新后，UCX跨进程endpoint明确记录 `rma_am(tcp/lo)`；同时列出 `device(cuda_ipc/cuda)` 能力，不能将配置中出现cuda_ipc等同于KV WRITE已经采用GPU直传。该候选仍属于TCP回环中转路线。[路径日志](logs/p5a/dev-nixl-ipc-host-1m/ucx-path-evidence.txt)。", "",
        "调参依据：UCX文档说明主机收发分片会影响无NVLink/RDMA时的CUDA中转成本，增大缓冲可能改善大消息；本机实测将默认NIXL的4k窗口从约1秒缩短至约0.22秒。共享内存错误处理选项依据UCX源码，开启后仍以实测失败淘汰，未宣称通用可用。[UCX配置说明](https://ucx-py.readthedocs.io/en/latest/configuration.html)、[共享内存参数源码](https://raw.githubusercontent.com/openucx/ucx/v1.20.0/src/uct/sm/mm/base/mm_iface.c)。", "",
        "**开发对比。** 根据全部传输窗口的总字节/总时间，以及4k单块带宽，选定新候选 `nixl-ipc-host-1m`；选择记录在 [selection.json](logs/p5a/selection.json)。单请求测量仍略落后r2，但并发队列行为可能不同，因此按要求各回放开发负载一次。只用 `bench/dev_toolagent_1200_240_x2.5.jsonl` 前300条，`--time-scale 1 --warmup 2 --force-close`，无重试；正式trace未用于调参、预检或冒烟。两个开发结果分别位于 [r2](logs/p5a/dev-mooncake-r2/) 和 [新候选](logs/p5a/dev-nixl-ipc-host-1m/)。", "",
        row(["方案","成功/失败/未发","时长(s)","输入tok/s","TTFT p50(s)","TTFT p99(s)","TPOT p50(ms)","联合SLO"]),
        row(["---"]*8),
    ]
    for name, s in [("r2配置",b),("NIXL 1MiB + IPC/TCP",a)]:
        lines.append(row([name, f"{s['completed_requests']}/{s['failed_requests']}/{s['unsent_requests']}",
            fmt(s["duration_sec"]),fmt(s["throughput"]["input_tokens_per_sec"]),
            fmt(s["latencies_sec"]["ttft"]["p50"]),fmt(s["latencies_sec"]["ttft"]["p99"]),
            fmt(s["latencies_sec"]["tpot"]["p50"],1000),fmt(s["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"],100)+"%"]))
    lines += ["", "正确性使用原P4a固定 [text20](logs/p4a/text20-cases.json) 和普通TP1参考，不重新挑选用例；比较完整text及output_ids。", "",
              row(["方案","text20","31663-token长请求","开发300"]),
              row(["---"]*4)]
    for name,g in gates.items():
        lines.append(row([name,f"{g['text20']}/20",f"{g['long_successes']}/2",
                          f"{g['completed']}成功 / {g['failed']}失败 / {g['unsent']}未发"]))
    lines += ["", "回放窗口GPU平均利用率（GPU0/1为P，GPU2/3为D；每5秒采样）：", "",
              row(["方案","GPU0 %","GPU1 %","GPU2 %","GPU3 %"]), row(["---"]*5)]
    for name,u in util.items():
        lines.append(row([name]+[fmt(u[str(i)]["mean_util_pct"]) for i in range(4)]))
    lines += ["", "现有阶段日志补充（各请求传输窗口及decode日志快照，非时间加权）：", "",
              row(["方案","传输请求数","首块→收齐 p50(s)","剩余传输 p50(s)","decode运行请求数中位","KV使用率中位","预分配使用率中位","P未完成bootstrap请求数最大"]),
              row(["---"]*8)]
    for d in [base_dir,cand_dir]:
        ph = read(d/"phase-analysis.json")
        lines.append(row([d.name,ph["rooms"],fmt(ph["full_span_p50_s"]),fmt(ph["residual_p50_s"]),
                          fmt(ph["running_reqs_median"]),fmt(ph["token_usage_median"],100)+"%",
                          fmt(ph["preallocated_usage_median"],100)+"%",ph["prefill_unbootstrapped_max"]]))
    lines += ["", "这些记录显示传输、decode KV预分配/驻留和bootstrap准入相互影响；低prefill利用率或“输入tok/s×每token KV字节数”本身不能证明传输引擎已经达到独立带宽上限。当前结果不能把网络拷贝、CUDA同步及KV容量造成的背压分别定量归因。", ""]
    lines += ["", f"**决策：** 新方案输入吞吐为r2的 **{ratio:.3f}倍**；TTFT p50为 **{ttft_ratio:.3f}倍**（降低{(1-ttft_ratio)*100:.2f}%）。正确性门槛{'通过' if correct else '未通过'}；性能门槛{'通过' if performance else '未通过'}（要求吞吐≥1.5倍或TTFT p50降低≥50%）。[机器可读决策](logs/p5a/decision.json)。", ""]
    report = (ROOT/"RESULTS.md").read_text()
    report = re.sub(r"\n## P5a：[^\n]*\n.*?(?=\n## |\Z)", "", report, flags=re.S)
    if formal_path.exists():
        s = read(formal_path)
        assert decision["formal_allowed"]
        marker = read(ROOT/"results/sglang_pd_r3.formal-started.json")
        d = Path(marker["log_dir"])
        main_row = row(["sglang_pd_r3",s["completed_requests"],s["failed_requests"],s["unsent_requests"],
            fmt(s["duration_sec"]),fmt(s["throughput"]["input_tokens_per_sec"]),fmt(s["throughput"]["output_tokens_per_sec"]),
            fmt(s["slo"]["ttft_le_5s"]["rate"],100)+"%",fmt(s["slo"]["tpot_le_100ms"]["rate"],100)+"%",
            fmt(s["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"],100)+"%",fmt(s["cached_tokens"]["token_weighted_ratio"],100)+"%"])
        report = "\n".join(x for x in report.splitlines() if not x.startswith("| sglang_pd_r3 |"))+"\n"
        report = report.replace("\n\ncache来自成功响应", "\n"+main_row+"\n\ncache来自成功响应",1)
        latrows = []
        for metric in ["ttft","tpot","e2e"]:
            v = s["latencies_sec"][metric]
            factor = 1000 if metric == "tpot" else 1
            latrows.append(row(["sglang_pd_r3",metric.upper()+(" (ms)" if factor==1000 else " (s)"),v["count"]]+
                               [fmt(v[k],factor) for k in ["mean","p50","p90","p95","p99","max"]]))
        report = report.replace("\n\n全部2/5/10s","\n"+"\n".join(latrows)+"\n\n全部2/5/10s",1)
        lines += [
            f"满足全部门槛后，通过 `scripts/run_experiment.sh sglang_pd_r3 --profile {CANDIDATE} --tag sglang_pd_r3-formal --gate {cand_dir} --baseline {base_dir}` 启动唯一正式回放。入口核验配置指纹、源码提交、两轮开发结果和性能门槛；排他创建正式标记防止重复。重启后的长请求/text20及开发20条流式冒烟通过，flush四后端cache并重启router后正式回放。正式trace与P3b/P4a完全相同，warmup2/force-close、无重试。", "",
            f"正式 **{s['completed_requests']}/1551成功、{s['failed_requests']}失败、{s['unsent_requests']}未发送**；总时长{fmt(s['duration_sec'])}秒，输入{fmt(s['throughput']['input_tokens_per_sec'])} tok/s，TTFT p50={fmt(s['latencies_sec']['ttft']['p50'])}秒，联合SLO={fmt(s['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'],100)}%。",
            "", f"完整命令和服务端/GPU日志：[正式目录]({d.relative_to(ROOT)})；正式请求：[sglang_pd_r3_requests.jsonl](results/sglang_pd_r3_requests.jsonl)。旧sglang_pd和r2行及原始数据继续保留。", "",
        ]
    else:
        assert not decision["formal_allowed"], "Qualified candidate requires formal completion before final report"
        assert not (ROOT/"results/sglang_pd_r3.formal-started.json").exists()
        lines += ["**未运行 sglang_pd_r3。** 新候选没有同时满足授权门槛，不能以微基准局部改善代替开发负载改善。没有创建r3正式标记或结果，已有五个正式label及原始结果均保留。", ""]
    lines += [
        "单请求带宽与并发输入吞吐口径不同：首块→收齐窗口包含后续prefill，输入吞吐还包含缓存命中token、调度排队、decode驻留与尾部排空。结果支持当前传输栈存在显著开销，但不能把测得GB/s当硬件PCIe峰值，也不能由本次结果推断PD架构上限。所有开发和正式配置均为单轮，无置信区间。", "",
        "最终GPU、服务进程及端口清理证据见 [cleanup.json](logs/p5a/cleanup.json)，GPU交接标记为 `logs/p5a.gpu_done`；原始结果SHA256核验见 [audit.json](logs/p5a/audit.json)。已重新运行 `bench/summarize.py`，只纳入正式label，不混入开发微基准。", "",
    ]
    (ROOT/"RESULTS.md").write_text(report+"\n".join(lines))
    print(json.dumps(decision,indent=2))


if __name__ == "__main__":
    main()
