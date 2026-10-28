#!/usr/bin/env python3
"""Build the final-only P7b report from the complete, cutoff-aware matrix."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from summarize import HISTORICAL, MAIN_SLO, MATRIX, format_number, load_runs, matrix_table, prepare_matrix, select_best


def main():
    results = ROOT / "results"
    paths = [str(results / f"{label}{'_cut300' if label in HISTORICAL else ''}_summary.json") for label in MATRIX]
    runs = prepare_matrix(load_runs(paths), results)
    indexed = {s["label"]: s for s, _ in runs}
    best = select_best(runs)
    audit = json.loads((ROOT / "logs/p7b/final-audit.json").read_text())
    analysis = json.loads((ROOT / "logs/p7b/analysis.json").read_text())
    selected = json.loads((ROOT / "configs/p7a_selected.json").read_text())

    def value(label, metric, q="p50", scale=1):
        return format_number(indexed[label]["latencies_sec"][metric][q], scale)

    def slo(label):
        return format_number(indexed[label]["slo"][MAIN_SLO]["rate"], 100, "%")

    def overview(system):
        return "；".join(f"{s['matrix']['configuration']}：{slo(s['label'])} / "
                        + value(s['label'], 'ttft')
                        + ("s" if isinstance(s["latencies_sec"]["ttft"]["p50"], (int, float)) else "")
                        for s,_ in runs if s["matrix"]["system"] == system)

    lines = [
        "# P7b：四系统最终并行配置矩阵", "",
        "本报告只包含四系统的最终版本：4个既有正式配置的300秒截断重算，以及11个新增配置各一次正式运行。"
        "不含已归档的旧版本；预检数据不参与性能排名。", "",
        "## 实验设置", "",
        "- 硬件：4×RTX4090 24GB，所有卡对CUDA P2P=false。GPU0位于NUMA0，GPU1/2/3位于NUMA1。"
        "本次新增配置的双卡实例分组为1+2和0+3；四卡顺序1,2,0,3，使用UUID固定映射。"
        "[拓扑](logs/p7a/topology.txt) · [P2P核验](logs/p7a/cuda-device-map.json)。",
        "- 模型：本地Llama3.1-8B-Instruct，BF16。统一disable-overlap-schedule、context-length33792、"
        "FCFS、max-running-requests128、enable-cache-report。普通多实例用cache_aware router，PD用冻结mini_lb/random及Mooncake TCP。",
        "- 新增非PD源码冻结为`f91e8a18da502106a07bb67c67dbf351a933bfd9`；PD冻结为"
        "`19261937a524bc67216d8b925a8ea5e50bfda0f3`。未修改推理/调度源码、选定启动参数及统计口径。"
        "既有Chunked DP4来自原正式提交`bd849d628`，并非与新增配置同期运行。",
        "- 正式负载：`bench/toolagent_600_240_x2.5.jsonl`，1551条请求，最后计划到达599.9975s；"
        "总输入11,245,637 token、请求输出261,776 token，最大输入31,663 token。开环到达，time-scale=1，"
        "`--warmup 2 --force-close --drain-timeout 300`。",
        "- 本次新增配置先做31,663-token长预热、开发trace前20条≤1分钟冒烟、清缓存；多实例配置重启router。"
        "MuxWise另按既定流程清缓存后复查长预热。正式过程有排他锁、完整日志及5秒GPU采样，结束清理。"
        "本次11个配置依指定顺序运行，各一次，重启次数0。", "",
        "[冻结启动命令和预检选择](P7A_NOTES.md) · [顺序执行进度](logs/p7b/progress.md) · "
        "[运行入口](scripts/run_experiment.sh)", "",
        "## 统计口径", "",
        "截止时刻为最后请求实际发出后300s；提前全部结束则立即结束。吞吐分母为"
        "`min(实际终态时刻, 截止时刻)`，包括开始空闲及排空时间。实时客户端到时关闭在途连接，记为unfinished；"
        "HTTP/协议错误记为failed。既有四个运行从原始SSE记录事后截断，使用`*_cut300_summary.json`，原始文件不改。",
        "",
        "- 所有SLO的分母均为全部1551条；未完成、失败、未发送及指标未定义均算未达标。"
        "主SLO为TTFT≤5s且TPOT≤100ms。TTFT≤2s单项也要求成功完成。",
        "- TTFT/E2E分位包含全部1551条。截止前无首token的TTFT取+∞；未成功完成的E2E取+∞。"
        "未完成/失败但已收到首token的请求保留真实TTFT。扩展实数线性分位中的+∞显示为`>cutoff`，"
        "它不是统一的单请求延迟阈值。",
        "- TPOT=(E2E−TTFT)/(输出token−1)，仅计算截止前成功完成且输出>1的请求，表中标明样本数n。"
        "单token输出不进入TPOT样本，也不达成联合SLO。",
        "- 输出吞吐计入截止前收到的全部SSE累计输出，含未完成/失败的部分输出。"
        "输入吞吐只计已交付首token的请求输入量，包含缓存，每请求计一次；不计排队输入及未观测的部分prefill。",
        "- cache为截止前成功完成且提供cached_tokens的响应中，缓存token/输入token的加权比例。"
        "PD的decode响应不能代表prefill实际缓存命中。TP/PP的KV容量为同一逻辑请求空间的分片容量，不能乘rank数；DP副本分别列出。",
        "- TTFT/E2E CDF以1551为分母，未观测尾部不补成有限值；TPOT CDF仅以有效已完成样本为分母。"
        "事件恰好落在截止时刻算入。", "",
        "[统计实现](bench/metrics.py) · [完整定义](bench/README.md)", "",
        "## 完整矩阵：15个最终配置", "",
        matrix_table(runs), "",
        "† PD cache列仅为decode响应值，prefill实际命中率未由该字段测量。"
        "表中KV来自本次/原正式运行就绪后的get_server_info，不用预检近似值替代。", "",
        "[矩阵Markdown](results/matrix_summary.md) · [完整精度CSV](results/matrix_summary.csv)", "",
        "![系统×配置主SLO](results/matrix_slo_heatmap.png)", "",
        "| 不支持/不适用的组合 | 原因 |",
        "| --- | --- |",
        "| MuxWise × PP2DP2、PP4、TP2PP2 | 原版MuxWise无PP路径。 |",
        "| MuxWise-PP × TP2DP2、TP4、TP2PP2 | 当前MuxWise-PP仅支持TP1。 |",
        "| MuxWise-PP × DP4(PP1) | 回到原版MuxWise路径，已由MuxWise DP4覆盖，不重复计为独立配置。 |",
        "| PD × 普通DP/PP列 | 本矩阵按专用P/D配比和每实例TP数定义PD配置；空白不表示已测试失败。 |", "",
        "## 各系统最佳配置", "",
        "在每系统已测配置中按主SLO最高选择，同分取TTFT p50最低；仅代表本次工作负载及本机结果。", "",
        "| 系统 | 最佳配置 | 主SLO（达标/1551） | TTFT p50/p99(s) | TPOT p50/p99(ms); n | 完成/未完/失败 | 输出 tok/s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s, _ in best:
        label = s["label"]
        lines.append(f"| {s['matrix']['system']} | {s['matrix']['configuration']} | {slo(label)} "
                     f"({s['slo'][MAIN_SLO]['passed']}/1551) | {value(label,'ttft')}/{value(label,'ttft','p99')} | "
                     f"{value(label,'tpot',scale=1000)}/{value(label,'tpot','p99',1000)}; n={s['latencies_sec']['tpot']['count']} | "
                     f"{s['completed']}/{s['unfinished']}/{s['failed']} | {s['throughput']['output_tokens_per_sec']:.3f} |")
    lines += [
        "",
        "本次完整矩阵的主SLO最高为MuxWise TP2DP2（82.656%），比Chunked最佳PP2DP2（58.672%）高23.985个百分点，"
        "比MuxWise-PP最佳PP2DP2（40.232%）高42.424个百分点。"
        "Chunked PP2DP2的TTFT p50更低（0.783s对1.791s），因此主SLO排名不是所有指标的排名。"
        "三个完整完成的系统最佳输出吞吐均约422–426 tok/s；本次吞吐接近不意味着延迟分布接近。",
        "", "[最佳配置Markdown](results/best_summary.md) · [完整精度CSV](results/best_summary.csv)", "",
        "![最佳配置主SLO](results/best_main_slo.png)", "",
        "![最佳配置TTFT CDF](results/best_ttft_cdf.png)", "",
        "![最佳配置TPOT CDF](results/best_tpot_cdf.png)", "",
        "图均提供同名SVG；[E2E CDF](results/best_e2e_cdf.png) · [全部SLO对比](results/best_slo_attainment.png)。", "",
        "## 未完成与运行异常", "",
    ]
    for s, _ in runs:
        label = s["label"]
        if not s["failed"] and not s["unfinished"]:
            continue
        evidence = analysis[label]["formal_error_evidence"]
        lines.append(f"- **{label}**：完成/未完成/失败={s['completed']}/{s['unfinished']}/{s['failed']}。")
        if evidence:
            unique = {}
            for e in evidence:
                if "#new-token: -" in e["text"] or e["text"].startswith(("RuntimeError:", "torch.OutOfMemoryError:", "AssertionError:")):
                    unique.setdefault(e["text"], e)
            for e in list(unique.values())[:4] or evidence[:2]:
                lines.append(f"  约回放{e['at_sec_approx']:.1f}s：[日志第{e['line']}行]({e['file']}) "
                             f"`{e['text'].replace('`', chr(39))}`。")
        elif s["unfinished"]:
            lines.append("  截止前未完成的请求保留已观测首token与部分输出；截断未完成不等同于进程崩溃。")
    lines += [
        "", "发生正式错误的配置同样使用全部1551条分母，不删除失败请求、不补跑。"
        "已有数据的崩溃配置不会触发“无数据启动失败”的重启例外。"
        "[错误类型与服务日志证据](logs/p7b/analysis.json)。", "",
        "## 并行方式影响与解释边界", "",
        "**Chunked。** 各配置主SLO / TTFT p50依次为：" + overview("Chunked") + "。"
        "TP每层需要跨rank归约；本机无P2P，新TP配置显式禁用custom allreduce，NCCL日志记录SHM路径。"
        "因此模型分片增加KV容量的同时也引入host/PCIe通信代价，不能假定TP卡数越多越快。"
        "这些端到端测量没有单独计时allreduce，不能将全部差值归因于通信。"
        "Chunked PP4和TP2PP2的结果受到上述正式调度崩溃影响，不能当作相应并行方式的稳定吞吐/尾延迟测量。"
        "PP还引入stage间传递与流水线等待；Chunked的PP组合按冻结源码要求关闭mixed chunk，"
        "所以TP/PP比较同时包含这一实现差别。"
        "PP2DP2虽有最低TTFT p50，其TPOT p99达到2637.747ms；原始请求确实存在短输出长间隔，"
        "例如idx=4615输出3 token、TTFT=0.493s、E2E=7.332s，对应TPOT=3419.675ms。"
        "这部分长尾必须保留，不能从低TTFT推导出decode同样平稳。",
        "",
        "**MuxWise。** " + overview("MuxWise") + "。"
        "DP4使用mem0.826、graph max-bs1和MLP分块4096；TP2DP2使用mem0.76，TP4使用mem0.80，"
        "两者graph max-bs32且无MLP分块（该功能仅TP1）。分片后KV容量、模型副本数、路由缓存驻留、"
        "整请求prefill排队和通信均发生变化。因此结果是选定完整配置的比较，不是纯TP单因素消融。",
        "",
        "**PD。** " + overview("PD") + "。"
        "decode预分配在接收KV前需要完整prompt及decode预留空间；可用KV同时扣除运行、传输、等待等请求的预留，"
        "队首不能满足时停止FIFO准入。增加P实例不自动增加D端容量，反之增加D实例会减少P端计算资源。"
        "1P1D(TP2)还同时改变每实例KV和TP通信；3P1D采用预检选定mem0.90，其他新增PD为0.88，"
        "也不是纯配比单因素对照。1P3D完成749条最多，但主SLO仅1/1551；2P2D按主SLO以8/1551胜出。"
        "几个达标请求的差异不足以证明配比排名稳定。该机制由冻结"
        "[decode.py](../sglang-p4a/python/sglang/srt/disaggregation/decode.py)的pop_preallocated/_allocatable_tokens支持，"
        "与表中的容量和完成数一起解释；本轮未新增逐阶段插桩，不能量化KV预分配、网络传输与计算各自的因果占比。"
        "PD的低TPOT是已完成请求子集的统计，必须与未完成数、含+∞的TTFT/E2E及全分母SLO一起解读。",
        "",
        "**MuxWise-PP。** " + overview("MuxWise-PP") + "。"
        "PP4是一份四stage逻辑KV空间，PP2DP2是两份独立两stage空间；不能把stage容量相加。"
        "PP2DP2按预先选定规则把inflight从6缩放为每实例3，mem0.71及其余v3参数保持。"
        "主SLO从PP4的31.915%增至40.232%（+8.317个百分点），但TTFT p99从13.365s增至31.574s，"
        "TPOT p50从33.940ms增至39.622ms；输出吞吐421.120与421.610 tok/s接近。"
        "PP深度、每stage计算量、独立路由副本、可共享前缀和KV容量共同改变延迟分布，"
        "最佳配置按主SLO选择，不意味着所有分位及TTFT≤2s都最优。",
        "",
        "**测量限制。** 每配置只有一次测量，无重复实验、误差条或置信区间。"
        "11个新增配置按固定顺序运行，4个既有配置来自历史观测；没有随机化控制温度、系统负载或缓存路由差异。"
        "历史300s截断是事后投影，新运行则在截止时关闭连接。禁止根据本轮正式trace调参或补跑，"
        "故较小差异不构成稳定优势或可推广结论，也不能将跨配置差异直接解释为某个单独优化的收益。",
        "",
        "## 配置、原始证据与复现", "",
        "| label | mem fraction | 正式日志 | 请求明细 | 采用的summary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s, _ in runs:
        label = s["label"]
        log = Path(s["matrix"]["log_dir"]).relative_to(ROOT)
        suffix = "_cut300" if label in HISTORICAL else ""
        memory = "/".join(str(v) for v in s["matrix"]["mem_fraction"].values())
        lines.append(f"| {label} | {memory} | [日志]({log}) | [requests](results/{label}_requests.jsonl) | "
                     f"[summary](results/{label}{suffix}_summary.json) |")
    lines += [
        "", "重建汇总和报告（仅CPU读取已保存数据，不会启动推理或重跑）：", "", "```bash",
        "source /root/workspace/muxpp_work/env.sh",
        "cd /root/workspace/muxpp_work",
        "python bench/summarize.py --matrix --results-dir results --out-dir results",
        "python scripts/p7b_analyze.py",
        "python scripts/p7b_audit.py",
        "python scripts/p7b_report.py",
        "```", "",
        "## 完成核验", "",
        f"审计时间：{audit['completed_utc']}。新增正式运行{audit['new_formal_runs']}次，重启{audit['restarts']}次；"
        "命令与预检指纹一致，原始四份结果/锁及冻结输入哈希未变。15份summary从原始请求重算逐项一致，"
        "矩阵CSV15行、最佳CSV4行。CPU统计/配置/选择测试14项通过。",
        "结束时GPU计算进程、推理服务、router、客户端及测试端口均无残留。"
        "[最终审计](logs/p7b/final-audit.json) · [GPU快照](logs/p7b/final-gpu.txt) · "
        "[测试日志](logs/p7b/unit-tests.log)。", "",
    ]
    pressure_lines = [
        "## PD预分配压力的日志证据", "",
        "| 配置label | decode实例 | 日志快照n | 总KV占用p50 | 预分配占用p50/p99 | 运行请求数p50 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s, _ in runs:
        if s["matrix"]["system"] != "PD":
            continue
        for worker, p in analysis[s["label"]]["pd_pressure_log_samples"].items():
            if "preallocated_usage_p50" not in p:
                continue
            pressure_lines.append(
                f"| {s['label']} | {worker} | {p['log_samples']} | {p['token_usage_p50']:.2%} | "
                f"{p['preallocated_usage_p50']:.2%}/{p['preallocated_usage_p99']:.2%} | {p['running_requests_p50']:.1f} |")
    pressure_lines += [
        "", "这些是decode日志快照的分位，非等时间加权统计；只取整个日志秒都落在正式统计窗口内的记录，"
        "排除边界秒以避免混入预热/结束清理。原日志占用率仅有两位小数，因此分位精度受限。"
        "`pre-allocated usage`来自传输队列请求的已分配prompt KV除以实例总容量；"
        "它可同时包含等待prefill/传输就绪的请求，并非只测量网络正在传输的数据。"
        "高占用且少量decode运行请求支持“KV驻留与准入存在压力”的解释，不能据此独立判定"
        "Mooncake TCP、prefill计算或调度各自的瓶颈占比。", "",
    ]
    insertion = lines.index("## 配置、原始证据与复现")
    lines[insertion:insertion] = pressure_lines
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
