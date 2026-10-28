#!/usr/bin/env python3
"""Render the P7a audit from saved preflights (does not run experiments)."""
import json
from pathlib import Path

from p7a_runner import MATRIX, ROOT, UUIDS


def fmt(value):
    return "—" if value is None else f"{value:.3f}" if isinstance(value, (float, int)) else str(value)


def capacity(meta):
    return "; ".join(f"{name}: {info.get('max_total_num_tokens', '—')}"
                     for name, info in meta.get("server_info", {}).items()) or "未就绪"


def failure(path, meta):
    if meta.get("status") == "PASS":
        return "通过"
    texts = [log.read_text(errors="replace") for log in path.glob("worker*.log")]
    for text in texts:
        for line in text.splitlines():
            if "torch.OutOfMemoryError:" in line:
                return line.split("torch.OutOfMemoryError: ", 1)[1].split("Including non-PyTorch")[0].strip()
    for text in texts:
        for line in text.splitlines():
            if line.startswith(("AssertionError:", "AttributeError:", "RuntimeError:", "ValueError:", "NameError:")):
                if "cutlass" in line and "CACHE_FILE" in line:
                    continue
                return line[:300].replace("|", "/")
    return meta.get("error", "运行中").replace("|", "/")


def main():
    selected = json.loads((ROOT/"configs/p7a_selected.json").read_text())
    notes = [
        "# P7a 并行配置矩阵准备与预检",
        "",
        "本任务只执行启动、合成长请求预热、开发trace冒烟及指定的一次开发300条健全性检查。没有运行任何新正式实验。",
        "既有四个正式运行仅作只读截断重算，原始请求、summary和正式锁的SHA256未变；当前主报告为 [RESULTS.md](RESULTS.md)。",
        "",
        "## 环境与边界",
        "",
        "- 环境：`source /root/workspace/muxpp_work/env.sh`；模型 `/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct`，BF16。",
        "- 主仓库 `/root/workspace/sglang` 固定 `f91e8a18da502106a07bb67c67dbf351a933bfd9`，分支muxwise-pp；没有修改推理/调度源码。",
        "- PD固定 `/root/workspace/sglang-p4a` 的r2提交 `19261937a524bc67216d8b925a8ea5e50bfda0f3`。进入任务时该worktree在P5a提交 `2374302bb75443ee572927d35a309def4777c06d`，已保留为git引用 `p7a-preserved-p5a` 后切回冻结r2。没有把P5a插桩或候选传输参数带入本次PD配置。",
        "- 统一 `--disable-overlap-schedule --context-length 33792 --schedule-policy fcfs --max-running-requests 128 --enable-cache-report`。除明确记录的限制，采用FlashInfer和CUDA graph。",
        "- 普通多实例采用wheel sglang-router/cache_aware；PD采用冻结worktree的mini_lb PD/random模式，r2全部Mooncake TCP环境变量与参数保持，详见下方精确命令。",
        "- 每个新TP实例显式关闭自定义allreduce，使用NCCL；实际日志含 `via SHM/direct/direct`。这是无P2P的硬件条件，不作为不支持原因。",
        "",
        "## GPU拓扑和映射",
        "",
        "[nvidia-smi topo -m](logs/p7a/topology.txt) · [NVML UUID与PCI映射](logs/p7a/gpu-uuid.csv) · [CUDA UUID/P2P核验](logs/p7a/cuda-device-map.json)",
        "",
        "| nvidia-smi / CUDA GPU | UUID | NUMA |",
        "| --- | --- | --- |",
    ]
    for i, uuid in enumerate(UUIDS):
        notes.append(f"| {i} | {uuid} | {0 if i == 0 else 1} |")
    notes += [
        "",
        "所有卡对 CUDA P2P=false。双卡实例分组为GPU1+2（NUMA1）与GPU0+3（跨NUMA，四卡中只有一张在NUMA0，因此无法避免）。单个四卡实例顺序为1,2,0,3；TP2×PP2的第一个TP组在NUMA1。TP1四实例按GPU0/1/2/3分配；PD P实例在前、D实例在后。启动命令使用UUID，避免枚举顺序歧义。",
        "",
        "## 配置矩阵",
        "",
        "已有最终正式运行映射：chunked→Chunked DP4；muxwise_r2→MuxWise DP4；sglang_pd_r2→PD 2P2D(TP1)；muxwise_pp_v3→MuxWise-PP PP4。新配置仅有预检状态，不能当成正式性能结果。",
        "",
        "| 新label | TP×PP×实例数 | mem fraction | 每实例逻辑KV容量(token) | 结论 | 说明 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, (_, tp, pp, dp) in MATRIX.items():
        cfg = selected[label]
        meta = json.loads((Path(cfg["gate"])/"run.json").read_text())
        notes.append(f"| {label} | {tp}×{pp}×{dp} | {cfg['mem']} | {capacity(meta)} | "
                     f"{'通过' if cfg['status'] == 'PASS' else '不支持'} | {cfg.get('reason','')} |")
    notes += [
        "",
        "容量来自健康就绪后的get_server_info.max_total_num_tokens；TP/PP各rank存储同一逻辑KV的分片，不能乘rank数。DP副本则分别列出，不把PP stage容量相加。",
        "",
        "支持范围和例外：",
        "",
        "- Chunked的PP2DP2、PP4、TP2PP2关闭mixed chunk；冻结源码server_args.check_server_args明确断言PP不能与mixed chunk并用。其它Chunked参数仍为chunk2048。",
        "- 原MuxWise的PP组合不适用；TP2/TP4不启用MLP token分块，因为该功能只支持TP1。显存选择仅改变mem fraction，若有其它例外在表中记录。",
        "- PD prefill由既有源码自动关闭CUDA graph，decode保留graph；decode的radix也由源码强制关闭（chunk cache）。这些是沿用r2的条件。",
        "- MuxWise-PP只支持TP1；TP2DP2、TP4、TP2PP2均记为不支持。PP1DP4回到原MuxWise路径，不作为独立MuxWise-PP配置。新增PP2DP2为两个PP2实例，inflight=3（6/4×2），其余v3参数保持。",
        "",
        "## 显存选择过程",
        "",
        "遵循P3b SELECTION方法：Chunked/PD从0.88、MuxWise从已有0.826、MuxWise-PP从已有0.71开始；只在启动/长预热/≤1分钟冒烟失败后依据显存或KV准入证据调整（MuxWise下调，PD3P1D尝试增加KV容量）。通过参考值即可停止，不搜索极限，不用正式trace调参。MuxWise另在冒烟后清cache并复查长请求，防止完整kernel加载后的显存变化遗漏。",
        "",
        "长预热：固定随机普通ID，31,663输入token、8输出token，每个普通模型实例分别执行；PD用与mini_lb相同的bootstrap协议显式覆盖所有P/D配对，避免随机路由遗漏实例。预热后清cache。",
        "预检脚本复核：早期MuxWise复查在冒烟后保留缓存直接发第二次长请求（TP4 mem0.80通过，TP2DP2 mem0.78在worker1 OOM）。后续协议v2补齐P3b的二次长预热前flush步骤，写入long-warmup-after-smoke-cache-flush.json；旧日志未覆盖，显存选择保留对已观察OOM的余量。",
        "冒烟只读 `bench/dev_toolagent_1200_240_x2.5.jsonl` 前20条，`--time-scale 0.2 --warmup 2 --force-close`；为满足≤1分钟，客户端timeout与drain-timeout初轮为50秒，后续对齐P3b为55秒（实际值逐候选记录在smoke-command.json中），外层硬上限60秒。正式与开发300的排空均为300秒。",
        "",
        "| 候选日志 | mem | backend / graph | 长预热通过数 | 冒烟完成/未完/失败；秒 | 冒烟后长预热数 | 结论/证据 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in MATRIX:
        for path in sorted((ROOT/"logs/p7a"/label).glob("*/run.json")):
            meta = json.loads(path.read_text())
            if meta.get("mode") != "preflight":
                continue
            cfg = meta["configuration"]
            warm = path.parent/"long-warmup.json"
            after = path.parent/"long-warmup-after-smoke.json"
            smoke = meta.get("smoke_summary")
            smoke_text = (f"{smoke['completed']}/{smoke['unfinished']}/{smoke['failed']}; {smoke['duration_sec']:.3f}"
                          if smoke else "未进入/未完成")
            rel = path.parent.relative_to(ROOT)
            notes.append(f"| [{label}/{path.parent.name}]({rel}) | {cfg['mem']} | "
                         f"{cfg['attention']} / {'off' if cfg['no_graph'] else 'max-bs'+str(cfg['graph_bs'])} | "
                         f"{len(json.loads(warm.read_text())) if warm.exists() else 0} | {smoke_text} | "
                         f"{len(json.loads(after.read_text())) if after.exists() else 0} | {failure(path.parent, meta)} |")
    notes += ["", "## 每个配置的最终启动命令", "",
              "以下命令为已保存候选的实际argv/env；`launch_<label>.sh`由统一runner启动整个集群并负责清理。每个脚本默认采用 `configs/p7a_selected.json` 的选定参数。候选中不支持的配置，启动入口会给出原因并拒绝启动。",
              "每个配置已经接入 `scripts/run_experiment.sh <label>`，本任务没有调用该入口。入口校验冻结源码、通过的预检门槛及完整命令指纹；全局GPU flock防止矩阵并发占卡，正式开始前用排他创建 `<label>.formal-started.json` 防止重复正式运行（失败也不重试）。",
              ""]
    for label in MATRIX:
        cfg = selected[label]
        path = Path(cfg["gate"])
        notes += [f"### {label}", "",
                  f"启动脚本：[launch_{label}.sh](scripts/launch_{label}.sh)；选择门槛：[{path.name}]({path.relative_to(ROOT)})。",
                  "", "```bash", (path/"commands.sh").read_text().strip(), "```", ""]
    notes += ["## MuxWise-PP PP2DP2 开发健全性检查", ""]
    marker = ROOT/"logs/p7a/muxpp_pp2dp2.dev300-started.json"
    if marker.exists():
        info = json.loads(marker.read_text())
        path = Path(info["log_dir"])
        meta = json.loads((path/"run.json").read_text())
        summary = meta.get("dev300_summary")
        notes += [f"只执行一次，独立排他标记记录于 `{marker.relative_to(ROOT)}`。使用开发trace前300条，time-scale=1、warmup=2、force-close、drain-timeout=300；只记录，没有据此调参。",
                  f"日志：[{path.name}]({path.relative_to(ROOT)})。复用P3函数产生的router-formal.log及早期“formal replay”重置提示仅为历史命名，实际客户端命令明确使用开发trace。"]
        if summary:
            notes += [f"完成/未完成/失败={summary['completed']}/{summary['unfinished']}/{summary['failed']}；未发送={summary['unsent_requests']}；"
                      f"实际统计时长={summary['duration_sec']:.3f}s，截止时刻={summary['cutoff_time']:.3f}s。",
                      f"主SLO={100*summary['slo']['ttft_le_5s_and_tpot_le_100ms']['rate']:.3f}%；"
                      f"TTFT p50/p99={fmt(summary['latencies_sec']['ttft']['p50'])}/{fmt(summary['latencies_sec']['ttft']['p99'])}s；"
                      f"TPOT样本数={summary['latencies_sec']['tpot']['count']}。这些仅为开发健全性记录。"]
        else:
            notes += ["未产出完整summary：" + meta.get("error", "尚未结束")]
    else:
        notes += ["尚未执行（本文件生成时）。"]
    notes += [
        "",
        "## 工具、统计、归档与验证",
        "",
        "- `bench/bench_client.py --drain-timeout SEC`默认300秒，按最后实际发送计时；取消归unfinished，保留已观测首token和部分输出。summary新增completed/unfinished/failed/cutoff_time。",
        "- `bench/metrics.py`供客户端和summarize共用；SLO分母为全计划请求，TTFT/E2E含+∞，JSON/表显示`>cutoff`；TPOT仅已完成且输出>1并报告样本数；吞吐按截止前已交付token与截断时长计算。输入只计已经交付首token的请求，包含缓存，每请求一次。详细定义见 [bench/README.md](bench/README.md)。",
        "- `scripts/p7a_recompute.py`产生四个`*_cut300_summary.json`，原始文件不改；`summary.*`、`final_summary.*`与所有当前图只含四个最终版本并采用新口径。事后截止为最后实际发送+300秒。",
        "- 被替代版本的原始结果、日志及旧报告已移动到`archive/superseded/`，没有删除；归档manifest包含逐文件SHA256并在移动后复核。P*_NOTES.md保留为实现文档，其历史当前结论不维护。当前报告不引用被替代结果。",
        "- [13项CPU单测日志](logs/p7a/unit-tests.log)：mock验证实际最后发送deadline、取消关闭连接、保留部分输出、failed与unfinished区分、提前完成不空等、全计划分母、扩展实数分位、离线一致性及边界时刻；配置测试覆盖11项卡数/参数、v3参数保持与正式锁阻止客户端启动。",
        "",
        "## 最终审计与清理",
        "",
    ]
    audit_path = ROOT/"logs/p7a/final-audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        notes += ["[完整审计](logs/p7a/final-audit.json)。",
                  f"原始正式文件校验：{audit['original_final_hashes_unchanged']}；新增正式运行数：{audit['new_formal_runs']}。",
                  f"GPU计算进程：{audit['gpu_compute_processes'] or '无'}；残留服务/客户端进程：{audit['residual_processes'] or '无'}；占用测试服务端口：{audit['occupied_ports'] or '无'}。",
                  "主仓库与PD冻结worktree均通过提交与干净状态核验。"]
    notes.append("")
    (ROOT/"P7A_NOTES.md").write_text("\n".join(notes))


if __name__ == "__main__":
    main()
