#!/usr/bin/env python3
"""Generate final comparison and P6 chapter from immutable formal measurements."""
import datetime
import json
import pathlib
import re
import shlex

ROOT = pathlib.Path("/root/workspace/muxpp_work")
FINAL = ["chunked", "muxwise_r2", "sglang_pd_r2", "muxwise_pp_v3"]
ALL = ["chunked", "muxwise", "muxwise_r2", "sglang_pd", "sglang_pd_r2", "muxwise_pp", "muxwise_pp_v3"]


def read(p):
    return json.loads(p.read_text())


def f(value, scale=1):
    return "—" if value is None else f"{value*scale:.3f}"


def pct(value):
    return f(value, 100) + "%"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"]*len(headers)) + " |"] + [
            "| " + " | ".join(map(str, row)) + " |" for row in rows]


def quantiles(s, metric):
    return "/".join(f(s["latencies_sec"][metric][q], 1000 if metric == "tpot" else 1)
                    for q in ["p50", "p90", "p95", "p99"])


def main():
    runs = {l: read(ROOT/f"results/{l}_summary.json") for l in ALL}
    paths = {l:pathlib.Path(read(ROOT/f"results/{l}.formal-started.json")["log_dir"]) for l in ALL}
    metas = {l:read(paths[l]/"run.json") for l in ALL}
    analysis = {l:read(paths[l]/"analysis.json") for l in ["muxwise_pp_v3","muxwise_r2"]}
    pp = runs["muxwise_pp_v3"]
    r2 = runs["muxwise_r2"]
    oldpp = runs["muxwise_pp"]
    oldmux = runs["muxwise"]
    ch = runs["chunked"]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    lines = ["# 四系统最终结果（P6）", "", "## 最终对比", "",
             f"更新时间：{now}。最终版本固定为 **Chunked=`chunked`、MuxWise=`muxwise_r2`、SGLang-PD=`sglang_pd_r2`、MuxWise-PP=`muxwise_pp_v3`**。",
             "", "4×RTX4090 24GB，Llama3.1-8B-Instruct BF16；同一正式trace共1551条、到达窗口599.9975秒。"
             "每个版本只测量一次，无置信区间。旧版本不是同配置重复样本，未覆盖或补跑。"
             "主SLO为 **TTFT≤5s 且 TPOT≤100ms**，分母是全部1551条计划请求；失败、未发送和TPOT未定义均算未达标。"
             "延迟及吞吐只统计成功请求，吞吐分母包含尾部排空；输入吞吐包含缓存命中token。", ""]
    rows = []
    for l in FINAL:
        s = runs[l]
        rows.append([l, f'{s["completed_requests"]}/{s["failed_requests"]}/{s["unsent_requests"]}',
                     f(s["duration_sec"]), f(s["throughput"]["input_tokens_per_sec"]),
                     f(s["throughput"]["output_tokens_per_sec"]),
                     pct(s["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"]),
                     pct(s["cached_tokens"]["token_weighted_ratio"])])
    lines += table(["最终版本","完成/失败/未发","总时长(s)","输入tok/s","输出tok/s","主联合SLO","响应token加权cache"],rows)
    lines += ["", "缓存为成功响应的`meta_info.cached_tokens`按输入token加权；PD r2响应为0，"
              "但P4a prefill日志实际命中约39.342%，不能据此称PD完全无缓存。其余系统也可能受回撤、日志边界和路由影响。", "",
              "延迟分位按 **p50 / p90 / p95 / p99** 排列：", ""]
    lines += table(["最终版本","TTFT(s)","TPOT(ms)","E2E(s)"],
                   [[l,quantiles(runs[l],"ttft"),quantiles(runs[l],"tpot"),quantiles(runs[l],"e2e")] for l in FINAL])
    single_keys = ["ttft_le_2s","ttft_le_5s","ttft_le_10s","tpot_le_50ms","tpot_le_100ms","tpot_le_200ms"]
    lines += ["", "单项阈值达成率（同样以1551为分母）：", ""]
    lines += table(["最终版本","TTFT≤2s","TTFT≤5s","TTFT≤10s","TPOT≤50ms","TPOT≤100ms","TPOT≤200ms"],
                   [[l]+[pct(runs[l]["slo"][k]["rate"]) for k in single_keys] for l in FINAL])
    lines += ["", "全部九组联合阈值：", ""]
    joint_keys = [f"ttft_le_{t}s_and_tpot_le_{p}ms" for t in [2,5,10] for p in [50,100,200]]
    lines += table(["最终版本"]+[f"{t}s/{p}ms" for t in [2,5,10] for p in [50,100,200]],
                   [[l]+[pct(runs[l]["slo"][k]["rate"]) for k in joint_keys] for l in FINAL])
    pp_slo = pp["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"]
    ch_slo = ch["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"]
    lines += ["", "### 结论与解释边界", "",
              f"- v3相对原PP：主SLO **{pct(oldpp['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'])}→{pct(pp_slo)}**；"
              f"TTFT p50 **{f(oldpp['latencies_sec']['ttft']['p50'])}→{f(pp['latencies_sec']['ttft']['p50'])}s**，"
              f"TPOT p50 **{f(oldpp['latencies_sec']['tpot']['p50'],1000)}→{f(pp['latencies_sec']['tpot']['p50'],1000)}ms**。"
              "这反映异步decode、分区校准/预取、MLP分块和容量等联合变化，不能用一次端到端测量分离各项贡献。",
              f"- 与Chunked相比，v3主SLO相差 **{(pp_slo-ch_slo)*100:+.3f}个百分点**，"
              f"输出吞吐比为 **{pp['throughput']['output_tokens_per_sec']/ch['throughput']['output_tokens_per_sec']:.3f}×**。"
              f"TTFT p99为{f(pp['latencies_sec']['ttft']['p99'])}/{f(ch['latencies_sec']['ttft']['p99'])}s，"
              f"TPOT p99为{f(pp['latencies_sec']['tpot']['p99'],1000)}/{f(ch['latencies_sec']['tpot']['p99'],1000)}ms（v3/Chunked）。"
              f"但TTFT≤2s达成率仅{pct(pp['slo']['ttft_le_2s']['rate'])}，低于Chunked的{pct(ch['slo']['ttft_le_2s']['rate'])}；"
              f"TPOT p50也略高（{f(pp['latencies_sec']['tpot']['p50'],1000)}对{f(ch['latencies_sec']['tpot']['p50'],1000)}ms），"
              "因此不是所有阈值和延迟分位都占优。"
              "应分别看首token等待、生成稳定性和完成率；未做重复测量，不能据小幅吞吐差异声称稳定优势。",
              f"- MuxWise r2相对r1：完成/失败由{oldmux['completed_requests']}/{oldmux['failed_requests']}变为"
              f"{r2['completed_requests']}/{r2['failed_requests']}，主SLO由{pct(oldmux['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'])}"
              f"变为{pct(r2['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'])}。"
              "r1采用Triton且关闭graph；r2的最终后端、graph和KV容量见P6配置表。"
              "最终r2 mem仍为0.826、KV约32.4k/卡，与r1相近；mem0.84通过预检但开发并发负载OOM，故只调整一轮降回0.826。"
              "FlashInfer已恢复，graph仅bs1，不能声称恢复了graph32或提高了KV容量。"
              "这些变化使两版本的差值不能被解释为纯调度改进。",
              "- PD r2解决了首轮TCP传输失败，但TTFT p50仍约797s、主SLO约0.516%。"
              "P5a显示decode KV预分配/驻留、bootstrap准入和传输互相制约，典型decode并发仅1；"
              "低prefill利用率不能单独证明传输带宽饱和。P5a新传输候选未达到预定门槛，因此未运行r3。",
              "- Chunked/MuxWise有四个完整模型副本、独立radix及cache_aware路由；PP为一个四stage模型、统一逻辑KV；"
              "PD为2P2D、random mini_lb，decode使用强制chunk cache。容量、路由、缓存驻留、graph档位和通信均不同。"
              "PP的每stage KV容量不能乘四后当作四倍独立请求容量。",
              "- 旧Chunked/PD结果来自早先运行，P6没有同期开新对照；v3经历开发调优，r2按指定显存选择并在开发健全性检查失败后仅调整一轮复核。"
              "时间、GPU状态和调优机会并未做随机化控制。保留预热后的JIT/在线模型状态，正式前flush后端cache并重启DP router。"
              "NVML利用率包含NCCL等待kernel，不等于SM有效计算占用。", "",
              "### 版本演进", ""]
    lines += table(["版本","完成/失败","TTFT p50/p99(s)","TPOT p50/p99(ms)","输出tok/s","主联合SLO"],
                   [[l,f'{runs[l]["completed_requests"]}/{runs[l]["failed_requests"]}',
                     "/".join(f(runs[l]["latencies_sec"]["ttft"][q]) for q in ["p50","p99"]),
                     "/".join(f(runs[l]["latencies_sec"]["tpot"][q],1000) for q in ["p50","p99"]),
                     f(runs[l]["throughput"]["output_tokens_per_sec"]),
                     pct(runs[l]["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"])] for l in ALL])
    mem = metas["muxwise_r2"]["server_info"]["worker0"]["mem_fraction_static"]
    attention = analysis["muxwise_r2"]["effective_attention_backend"]["worker0"]
    graph_disabled = metas["muxwise_r2"]["server_info"]["worker0"]["disable_cuda_graph"]
    graph_bs = metas["muxwise_r2"]["server_info"]["worker0"]["cuda_graph_max_bs"]
    lines += [""] + table(["版本","代码/配置差异"], [
        ["chunked","P3b `bd849d628`；TP1×4、FlashInfer、graph max-bs32、mem0.88、chunk2048、mixed-chunk。"],
        ["muxwise","P3b `bd849d628`；TP1×4、整请求prefill、五组SM分区、Triton/no-graph、mem0.826、无MLP分块；KV约32.4k/卡。"],
        ["muxwise_r2",f"P6 `f91e8a18d`；原pp=1调度，启用默认关闭的MLP token分块4096（`b0a107c21`）；"
                       f"{attention}、disable_graph={graph_disabled}、graph max-bs{graph_bs}、mem{mem}；cache_aware、五分区、65536 slice预算不变。"],
        ["sglang_pd","P3b `bd849d628`；2P2D、Mooncake TCP默认队列/准入参数；正式出现批量传输及session失效。"],
        ["sglang_pd_r2","隔离 `19261937a524bc67216d8b925a8ea5e50bfda0f3`；Mooncake队列65535/准入60s/8 lanes/1MiB slice，"
                        "SGLang传输队列及线程池4、等待3600s；原模型、2P2D、mem0.88、graph、KV内容不变。完整环境见P4a。"],
        ["muxwise_pp","P3b `bd849d628`；PP4、五分区、同步decode环、slice预算8192、inflight4、mem0.65、TPOT系数1.0、无MLP分块。"],
        ["muxwise_pp_v2（仅开发）","`ebdf39aa0`；八分区校准、RLS冷启动修正、双接收槽prefetch与READY直接发送、"
                                 "slice65536、inflight6、graph档1/4/8/16/32、系数0.75、mem0.63；没有v2正式label。"],
        ["muxwise_pp_v3","P6 `f91e8a18d`；在v2上加真实allocator/阻塞插桩、异步decode环与控制预取/槽生命周期修正"
                        "（`da74a65a6/e50f75191/714fd92a5`）、MLP分块4096与峰值记录（`b0a107c21/f91e8a18d`）；"
                        "选定R3 mem0.71、inflight6，其余严格见p5b_v3_serve.sh；KV359498/stage。"],
    ])
    lines += ["", "所有原始版本仍在[全部版本summary](results/summary.md)和[CSV](results/summary.csv)。"
              "最终四版本独立产物：[final_summary.md](results/final_summary.md)、[final_summary.csv](results/final_summary.csv)。", "",
              "![最终四版本TTFT CDF](results/final_ttft_cdf.png)", "",
              "![最终四版本TPOT CDF](results/final_tpot_cdf.png)", "",
              "[E2E CDF](results/final_e2e_cdf.png) · [TTFT时间线](results/final_ttft_over_time.png) · "
              "[全部SLO图](results/final_slo_attainment.png)。CDF和时间线采用对数延迟轴以展示PD与其它系统的尺度差异。", "", "---", ""]
    final_section = "\n".join(lines)
    lines = ["", "## P6：MuxWise-PP v3 与 MuxWise r2 正式运行", "",
             f"本轮源码固定`f91e8a18da502106a07bb67c67dbf351a933bfd9`，分支`muxwise-pp`；"
             "P6未修改推理/调度源码。仅新增生命周期、只读分析/审计/报告脚本，并扩展汇总及入口。"
             "环境快照见[environment.json](logs/p6/environment.json)，原结果SHA256见[before.json](logs/p6/before.json)。", "",
             "### 正式唯一性与流程", "",
             "先v3、再r2。所有启动前检查空卡；每5秒采样GPU。长预热输入为固定种子随机31,663 token、输出8；"
             "随后20条冒烟（P3b同一trace前20，time-scale0.2、timeout55s，进程上限60s）。"
             "r2在冒烟后逐卡重复长预热，检查完整kernel加载后的余量。正式前flush所有后端；DP重启router清除路由前缀树。"
             "正式客户端统一warmup2、force-close、time-scale1、timeout3600s、无重试。"
             "在客户端启动前以`open('x')`建立永久正式标记，异常也不删除。"
             "GPU/JIT与PP在线RLS保留预热状态。", "",
             "生命周期PASS只表示流程结束，完成/失败以客户端summary为准。日志时间是UTC+8，客户端为UTC；分析按Unix时间对齐。", ""]
    rows = []
    for l in ["muxwise_pp_v3","muxwise_r2"]:
        m = metas[l]
        smoke = read(paths[l]/"smoke"/f"{l}_smoke_summary.json")
        warm = read(paths[l]/"long-warmup.json")
        rows.append([l,m["started_utc"],runs[l]["run_started_at"],m["finished_utc"],
                     f(m["startup_s"]),str(len(warm)),f(smoke["duration_sec"]),m["formal_returncode"]])
    lines += table(["版本","生命周期开始UTC","正式开始UTC","清理完成UTC","启动(s)","长预热条数","冒烟20/20(s)","正式退出码"],rows)
    lines += ["", "### MuxWise r2 显存选择与开发健全性检查", "",
              "[完整选择过程](logs/p6/SELECTION.md)。初始候选选择仅使用启动、31,663-token长请求和≤1分钟冒烟；"
              "开发健全性检查失败后仅触发下述一轮显存调整，未用正式结果调参。"
              "最终选择门槛目录由`--gate`传入，正式启动前核对argv与已通过候选完全一致。", ""]
    candidates = []
    for p in (ROOT/"logs/p6").glob("*/run.json"):
        m = read(p)
        if m.get("mode") == "preflight" and m.get("label") == "muxwise_r2":
            candidates.append((m["started_utc"],p.parent,m))
    rows = []
    for _,p,m in sorted(candidates):
        cmd = read(p/"commands.json")[0]["argv"]
        option = lambda k:cmd[cmd.index(k)+1]
        infos = m.get("server_info",{}).get("worker0",{})
        outcome = m["status"]
        text = (p/"worker0.log").read_text()
        oom = re.search(r"torch.OutOfMemoryError: CUDA out of memory\. Tried to allocate ([^.]+?\.\d+ \w+|[\d]+ \w+)\.",text)
        if oom:
            outcome += "; OOM申请" + oom[1]
        first = len(read(p/"long-warmup.json")) if (p/"long-warmup.json").exists() else 0
        second = len(read(p/"long-warmup-after-smoke.json")) if (p/"long-warmup-after-smoke.json").exists() else 0
        rows.append([p.name,option("--mem-fraction-static"),option("--cuda-graph-max-bs"),
                     "triton" if "--attention-backend" in cmd and option("--attention-backend")=="triton" else "FlashInfer默认",
                     "--disable-cuda-graph" in cmd,infos.get("max_total_num_tokens","—"),f"{first}/4; {second}/4",outcome])
    lines += table(["候选目录","mem","graph max-bs","attention","禁用graph","GPU0 KV","首次/复查长预热完成","结果"],rows)+[""]
    dev = read(paths["muxwise_r2"]/"dev300"/f"{metas['muxwise_r2'].get('dev_label','muxwise_r2_dev300')}_summary.json")
    initial_path = ROOT/"logs/p6/muxwise_r2-formal/dev300/muxwise_r2_dev300_summary.json"
    if metas["muxwise_r2"].get("sanity_round",0):
        initial = read(initial_path)
        lines += [f"初次开发300（mem0.84、graph1）为**{initial['completed_requests']}成功、"
                  f"{initial['failed_requests']}失败、{f(initial['duration_sec'])}s**；"
                  "四worker在FlashInfer attention merge输出分配处OOM。初次目录虽然名为`muxwise_r2-formal`，"
                  "但尚未创建正式锁、没有正式回放。按授权只调整一轮：mem降为0.826，其余不变，"
                  "再次预检后仅做一次开发300复核。保留[首次失败检查](logs/p6/muxwise_r2-formal/dev300/)"
                  "及其完整服务日志，不将失败检查计作正式结果。", ""]
    lines += [f"选定配置后的开发健全性检查：`bench/dev_toolagent_1200_240_x2.5.jsonl`前300条，time-scale=1，"
              f"warmup2/force-close；**{dev['completed_requests']}成功、{dev['failed_requests']}失败、"
              f"{dev['unsent_requests']}未发，{f(dev['duration_sec'])}s**。"
              f"TTFT p50/p99={f(dev['latencies_sec']['ttft']['p50'])}/{f(dev['latencies_sec']['ttft']['p99'])}s，"
              f"TPOT p50/p99={f(dev['latencies_sec']['tpot']['p50'],1000)}/{f(dev['latencies_sec']['tpot']['p99'],1000)}ms，"
              f"主SLO={pct(dev['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'])}。"
              "这是最终选定配置的一次健全性检查；初次失败后只调整一轮，复核后不再调参。开发结果不混入正式summary。", "",
              "```bash",shlex.join(read(paths["muxwise_r2"]/"dev-command.json")),"```", "",
              "### 实际配置与精确命令", ""]
    rows = []
    for l in ["muxwise_pp_v3","muxwise_r2"]:
        for name,info in metas[l]["server_info"].items():
            cap = analysis[l]["capacity_by_worker"].get(name,[])
            rows.append([l,name,analysis[l]["effective_attention_backend"][name],info["disable_cuda_graph"],info["cuda_graph_max_bs"],
                         info["mem_fraction_static"],"/".join(map(str,cap)),info["pdmux_prefill_mlp_chunk_tokens"]])
    lines += table(["版本","实例","attention","禁用graph","graph max-bs","mem fraction","KV token容量","MLP chunk"],rows)
    if not graph_disabled and graph_bs < 32:
        lines += ["",f"MuxWise r2为通过显存门槛将graph max-bs缩小为{graph_bs}；大于该档的decode批次不能采用这组CUDA graph，"
                  "因此没有恢复到Chunked的graph32条件。这是性能比较中的显式混杂因素。"]
    for l in ["muxwise_pp_v3","muxwise_r2"]:
        p = paths[l]
        rel = p.relative_to(ROOT)
        lines += ["",f"#### {l}","",f"完整日志、服务端解析参数和配置快照：[{rel}]({rel})。"
                  "以下服务命令由生命周期脚本并发管理；不要逐条前台执行。", "",
                  "生命周期精确入口：", "```bash","source /root/workspace/muxpp_work/env.sh",
                  shlex.join(metas[l]["entry_command"]),"```", "",
                  "服务命令：", "```bash",(p/"commands.sh").read_text().strip(),"```", "",
                  "正式客户端：", "```bash",shlex.join(read(p/"formal-command.json")),"```"]
    lines += ["", "### P6 正式结果与资源观察", ""]
    lines += table(["版本","完成/失败/未发","TTFT p50/p90/p95/p99(s)","TPOT p50/p90/p95/p99(ms)","E2E p50/p90/p95/p99(s)"],
                   [[l,f'{runs[l]["completed_requests"]}/{runs[l]["failed_requests"]}/{runs[l]["unsent_requests"]}',
                     quantiles(runs[l],"ttft"),quantiles(runs[l],"tpot"),quantiles(runs[l],"e2e")] for l in analysis])
    rows = []
    for l,a in analysis.items():
        for gpu,v in a["gpu"].items():
            rows.append([l,gpu,v["samples"],f(v["util_mean"]),v["memory_max_MiB"]])
    lines += [""]+table(["版本","GPU","正式窗口5s样本数","NVML利用率均值(%)","采样峰值MiB"],rows)
    for l,a in analysis.items():
        c = a["server_cache"]
        lines += ["",f"- {l}：服务端正式窗口prefill批数={c['batches']}，new={c['new_tokens']}、cached={c['cached_tokens']}，"
                  f"日志口径命中={pct(c['ratio'])}；响应口径命中={pct(runs[l]['cached_tokens']['token_weighted_ratio'])}。"
                  f"send_lag p99/max={f(runs[l]['latencies_sec']['send_lag']['p99'],1000)}/"
                  f"{f(runs[l]['latencies_sec']['send_lag']['max'],1000)}ms。"
                  f"客户端错误={json.dumps(a['response_errors'],ensure_ascii=False)}；"
                  f"服务日志异常匹配{len(a['server_error_evidence'])}条，正式窗口retract相关日志{len(a['formal_retraction_evidence'])}条。"]
        lines += [f"  decode批次日志中的graph标志计数：`{json.dumps(a['decode_graph_log_counts'])}`；"
                  "这是日志采样次数，不是全部decode步数或时间加权比例。"]
    lines += ["", "服务端cache通常按秒级日志边界汇总，可能漏计或混入边界请求、包含回撤重算；不强制等于响应成功子集。"
              "v3首个正式准入与两个客户端16-token预热同处一秒，已按首请求输入长度恢复唯一对应批次、排除两个预热；"
              "最终准入1551条且new+cached=11,245,637，响应cached总量也一致，原始边界行见`cache_boundary_recovered`。"
              "r2同样恢复了起始秒内的8个正式prefill批次，排除同秒唯一16-token客户端预热；"
              "首秒正式输入均大于16，边界原始行及排除记录均已保存。修正后服务端cached总量4,027,591与响应一致，"
              "new+cached比原始输入多21,375 token；同期两条日志各回撤1请求，重计算口径不能直接当作独立请求输入量。"
              "异常及回撤原始行保存在各`analysis.json`；回撤日志行数不等于唯一请求数。", "",
              "r2尾部快照（2026-10-20 06:53:48 UTC）显示GPU0/1利用率约97%，日志等待队列25/24；"
              "GPU2/3利用率0%，其最后decode日志队列为0。720秒时客户端仍有52条在途。"
              "这支持本轮尾部存在实例间排空不均衡；未做路由/容量消融，不能将其单独归因于cache_aware或某项调度。"
              f"证据：[{paths['muxwise_r2'].relative_to(ROOT)}/progress-snapshots.jsonl]"
              f"({paths['muxwise_r2'].relative_to(ROOT)}/progress-snapshots.jsonl)。", "",
              "### v3：准入阻塞与真实 allocator", ""]
    a = analysis["muxwise_pp_v3"]
    reasons = ["inflight","kv","decode_reserve","budget","max_running_requests","other"]
    rows = []
    for name, key in [("到达窗口","admission_arrival"),("完整回放","admission")]:
        d = a[key]
        bounds = [v-pp["run_started_unix"] for v in d["strict_window"]]
        lines += [f"{name}准入差分边界：客户端起点后{f(bounds[0])}–{f(bounds[1])}s（完全落在窗口内的summary边界）。"]
        total_wall = sum(d["wall"].get(k,0) for k in reasons)
        for k in reasons:
            rows.append([name,k,d["blocked"].get(k,0),f(d["blocked_pct"].get(k,0)),
                         f(d["wall"].get(k,0)),f(100*d["wall"].get(k,0)/total_wall if total_wall else 0)])
    lines += [""]+table(["窗口","互斥首个原因","零准入次数","次数占比(%)","估算阻塞时间(s)","阻塞时间占比(%)"],rows)
    lines += ["", "inflight检查优先；计数只记录第一个阻塞原因，不表示其它约束同时不存在。"
              "wall为两次决策间沿用上一状态的时间加权估计，不是硬件因果计时。部分准入后停止另记stops，"
              "不属于零准入次数。", "",
              f"到达窗口部分停止计数：`{json.dumps(a['admission_arrival']['stops'],ensure_ascii=False)}`。", "",
              "![v3准入阻塞直方图](logs/p6/muxwise_pp_v3-formal/admission_histogram.png)", ""]
    rows = []
    for window,key in [("到达窗口","admission_arrival"),("完整回放","admission")]:
        for metric,d in a[key]["allocator"].items():
            rows.append([window,metric]+[f(d.get(q)) for q in ["p10","p50","p90","max"]])
    lines += table(["窗口","allocator字段(token；waiting为请求数)","p10","p50","p90","max"],rows)
    d = a["admission_arrival"]
    blocked_wall = sum(d["wall"].get(k,0) for k in reasons)
    capacity_wall = sum(d["wall"].get(k,0) for k in ["kv","decode_reserve"])
    lines += ["",f"正式到达窗口中，inflight限制累计估算{f(d['wall'].get('inflight',0))}s；"
              f"KV与完整未来输出预留合计{f(capacity_wall)}s，占全部阻塞时间"
              f"{f(100*capacity_wall/blocked_wall if blocked_wall else 0)}%。"
              "P5b开发R3在相同参数下未观察到KV/预留阻塞，因此容量改善不等于任意更长负载下容量约束消失。"
              "这些是观察到的首个准入限制，不能据此单独量化其对最终TTFT的因果贡献。"]
    lines += ["",f"仅waiting>0时：protected p50/p90/max={f(d['waiting_allocator']['protected']['p50'])}/"
              f"{f(d['waiting_allocator']['protected']['p90'])}/{f(d['waiting_allocator']['protected']['max'])} token；"
              f"free+evictable p10/p50={f(d['waiting_available']['p10'])}/{f(d['waiting_available']['p50'])}；"
              f"inflight新输入p50/max={f(d['waiting_allocator']['inflight_new_tokens']['p50'])}/"
              f"{f(d['waiting_allocator']['inflight_new_tokens']['max'])}。", "",
              "`used=capacity−free`、`protected=used−evictable`。在途新输入已分配并包含在protected内，不能再次扣除；"
              "共享radix前缀只在allocator中计一次。各字段分位独立，不能将中位数相加；Σctx不等于真实KV占用。"
              "allocator约0.5秒采样并随10秒summary输出，尾端不足summary周期的样本可能未输出。", "",
              "![v3 allocator与等待队列](logs/p6/muxwise_pp_v3-formal/allocator.png)", "",
              "### v3：分区驻留与decode环分解", ""]
    bounds = [v-pp["run_started_unix"] for v in a["timing_arrival_strict_window_unix"]]
    lines += [f"到达窗口计时边界为起点后{f(bounds[0])}–{f(bounds[1])}s。下表是时间加权驻留，"
              "不是summary瞬时快照次数；分区编号对应prefill/decode SM数。", ""]
    parts = ["128/0","120/8","112/16","104/24","96/32","80/48","64/64","0/128"]
    lines += table(["stage"]+[f"{i}:{p}(%)" for i,p in enumerate(parts)],
                   [[stage]+[f(d["partition_percent"].get(str(i),0)) for i in range(8)]
                    for stage,d in a["timing_arrival_strict"].items()])
    rows = []
    for stage,d in a["timing_arrival_strict"].items():
        w = d["wall_percent"]
        rows.append([stage,f(d["cuda_percent"].get("prefill",0)),f(w["prefill"].get("empty",0)),
                     f(w["prefill"].get("ready",0)),f(d["cuda_percent"].get("decode",0))] +
                    [f(w["decode"].get(k,0)) for k in ["send","token","control_recv","control","launch","retire"]])
    lines += [""]+table(["stage","prefill CUDA%","prefill empty%","prefill READY%","decode CUDA%",
                        "send%","token%","control_recv%","control%","launch%","retire%"],rows)
    lines += ["", "send/token/control_recv/control等是主调度线程墙钟分类，CUDA为事件区间；prefill、decode和通信可重叠，"
              "CPU与GPU类别不可相加成100%。异步下游control_recv包含retired token等待，类别迁移不是独立收益。"
              "严格采用推荐正式命令，没有启用开发逐步round trace，故本轮不报告逐microbatch环周期分位或shape匹配加速比；"
              "这里给出summary可支持的环分解。完整回放计时、MAPE、快照频数也保存在analysis.json。", ""]
    rows = []
    for stage,d in a["cuda_memory_process_peak_MiB"].items():
        rows.append([stage,f(d.get("max_allocated")),f(d.get("max_reserved"))])
    lines += table(["stage","PyTorch max allocated(MiB)","PyTorch max reserved(MiB)"],rows)
    lines += ["", "上述PyTorch进程峰值包含启动和预热至summary的历史峰值；与5秒NVML采样峰值定义不同。"
              f"正式窗口迁移指令={len(a['formal_migration_evidence'])}，不能把收益归功于未发生的迁移。", "",
              "最终idle self-check：", "", "```text"]+[d["text"] for _,d in sorted(a["final_idle"].items())]+["```", "",
              "四级free_kv+evictable均为359498、free_reqs=128，接收槽归还。"
              "这些自检中的累计prefill/decode/cache计数包含启动和预热，不作为正式请求计数。", "",
              "### 产物重建与最终清理", "",
              "```bash",
              "source /root/workspace/muxpp_work/env.sh",
              "python scripts/p6_analyze.py",
              "python bench/summarize.py 'results/*_summary.json' --final-labels chunked muxwise_r2 sglang_pd_r2 muxwise_pp_v3",
              "python scripts/p6_audit.py",
              "python scripts/p6_report.py",
              "```", "",
              "[最终只读审计](logs/p6/final-audit.json)核验原结果哈希、两次正式唯一性、全部逐条指标、"
              "31k预热、冒烟、开发300、四卡5s采样、精确R3命令、源码提交及最终进程/端口。"
              "结束GPU快照：[final-gpu.txt](logs/p6/final-gpu.txt)。"
              "本任务启动的GPU服务、router和客户端已清理，30000/31000–31003端口已释放。", ""]
    chapter = "\n".join(lines)
    original = (ROOT/"logs/p6/RESULTS.before.md").read_text()
    (ROOT/"RESULTS.md").write_text(final_section + original + chapter)
    (ROOT/"logs/p6/final-comparison.md").write_text(final_section.replace("](results/", "](../../results/"))
    print(ROOT/"RESULTS.md")


if __name__ == "__main__":
    main()
