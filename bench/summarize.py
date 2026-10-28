#!/usr/bin/env python3
"""Compare replay summaries and plot their per-request measurements (CPU only)."""

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import cutoff_records, summarize_records


# Only the final versions participate. Historical runs must use cut300 inputs.
MATRIX = {
    "chunked": ("Chunked", "DP4"),
    "chunked_tp2dp2": ("Chunked", "TP2DP2"),
    "chunked_pp2dp2": ("Chunked", "PP2DP2"),
    "chunked_tp4": ("Chunked", "TP4"),
    "chunked_pp4": ("Chunked", "PP4"),
    "chunked_tp2pp2": ("Chunked", "TP2PP2"),
    "muxwise_r2": ("MuxWise", "DP4"),
    "muxwise_tp2dp2": ("MuxWise", "TP2DP2"),
    "muxwise_tp4": ("MuxWise", "TP4"),
    "sglang_pd_r2": ("PD", "2P2D(TP1)"),
    "pd_1p1d_tp2": ("PD", "1P1D(TP2)"),
    "pd_3p1d": ("PD", "3P1D(TP1)"),
    "pd_1p3d": ("PD", "1P3D(TP1)"),
    "muxwise_pp_v3": ("MuxWise-PP", "PP4"),
    "muxpp_pp2dp2": ("MuxWise-PP", "PP2DP2"),
}
HISTORICAL = {"chunked", "muxwise_r2", "sglang_pd_r2", "muxwise_pp_v3"}
MAIN_SLO = "ttft_le_5s_and_tpot_le_100ms"


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif not isinstance(item, list):
            result[name] = item
    return result


def load_runs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no summary matches {pattern}")
        paths.extend(Path(p).resolve() for p in matches)
    runs = []
    labels = set()
    for path in dict.fromkeys(paths):
        summary = json.loads(path.read_text())
        if summary.get("schema_version") not in (1, 2):
            raise ValueError(f"unsupported summary schema: {path}")
        label = summary["label"]
        if label in labels:
            raise ValueError(f"duplicate label {label}; use distinct benchmark labels")
        labels.add(label)
        configured = Path(summary["request_details_file"])
        # Prefer a sibling file so an exported run directory remains portable.
        details = path.parent / configured.name
        if not details.exists():
            details = configured
        with details.open() as source:
            records = [json.loads(line) for line in source if line.strip()]
        if len(records) != summary["sent_requests"]:
            raise ValueError(f"request detail count does not match summary: {path}")
        cutoff = summary.get("cutoff_time")
        summary.update(summarize_records(records, summary["planned_requests"], cutoff))
        runs.append((summary, cutoff_records(records, cutoff)))
    return runs


def format_number(value, scale=1, suffix=""):
    if isinstance(value, str):
        return value
    return "—" if value is None else f"{value * scale:.3f}{suffix}"


def write_tables(runs, output, prefix=""):
    columns = ["label", "completed_requests", "unfinished_requests", "failed_requests", "duration_sec",
               "throughput.input_tokens_per_sec", "throughput.output_tokens_per_sec",
               "throughput.requests_per_sec", "latencies_sec.ttft.mean", "latencies_sec.ttft.p50",
               "latencies_sec.ttft.p90", "latencies_sec.ttft.p99", "latencies_sec.tpot.mean",
               "latencies_sec.tpot.p50", "latencies_sec.tpot.p90", "latencies_sec.tpot.p99",
               "latencies_sec.e2e.p99", "slo.ttft_le_5s_and_tpot_le_100ms.rate", "cached_tokens.token_weighted_ratio"]
    headers = ["系统", "完成", "未完成", "失败", "时长(s)", "输入 tok/s", "输出 tok/s", "req/s",
               "TTFT 均值(s)", "TTFT p50(s)", "TTFT p90(s)", "TTFT p99(s)",
               "TPOT 均值(ms)", "TPOT p50(ms)", "TPOT p90(ms)", "TPOT p99(ms)",
               "E2E p99(s)", "联合 SLO 5s/100ms", "输入token加权缓存比例"]
    flattened = [flatten(summary) for summary, _ in runs]
    lines = ["# Tool&Agent trace 回放对比", "", "| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in flattened:
        values = []
        for key in columns:
            value = row.get(key)
            if key == "label":
                values.append(str(value).replace("|", "\\|").replace("\n", " "))
            elif key in ("completed_requests", "unfinished_requests", "failed_requests"):
                values.append(str(value))
            elif "tpot." in key:
                values.append(format_number(value, 1000))
            elif key.endswith(".rate") or key.endswith("_ratio"):
                values.append(format_number(value, 100, "%"))
            else:
                values.append(format_number(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## 完整延迟分位", "",
                  "| 系统 | 指标 | 样本数 | mean | p50 | p90 | p95 | p99 | max |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for summary, _ in runs:
        for metric, unit, scale in (("ttft", "s", 1), ("tpot", "ms", 1000), ("e2e", "s", 1)):
            d = summary["latencies_sec"][metric]
            values = [format_number(d.get(k), scale) for k in ("mean", "p50", "p90", "p95", "p99", "max")]
            lines.append(f"| {summary['label']} | {metric.upper()} ({unit}) | {d['count']} | "
                         + " | ".join(values) + " |")
    keys = list(runs[0][0]["slo"])
    lines.extend(["", "## 全部 SLO 阈值", "",
                  "| 系统 | " + " | ".join(keys) + " |",
                  "| --- | " + " | ".join(["---"] * len(keys)) + " |"])
    for summary, _ in runs:
        lines.append(f"| {summary['label']} | " +
                     " | ".join(format_number(summary["slo"][key]["rate"], 100, "%") for key in keys) + " |")
    lines.extend(["", "TTFT/E2E 分位以全部计划请求为样本；缺少首token/未成功完成的值取+∞，显示为 >cutoff。"
                  "TPOT 仅采用截止前成功完成且输出>1的请求，上表列出样本数。SLO 分母为全部计划请求，unfinished/failed/未发均不达标。"
                  "吞吐计入截止前交付的全部输出（含部分输出）；输入只计截止前已收到首token的请求。分母为min(实际时长,截止时刻)。",
                  "", "ITL 是两个增加 completion_tokens 的 SSE 事件之间的观测间隔；合包时无法还原真实逐 token ITL。"
                  "缓存使用成功响应的输入token加权比例；字段缺失显示为 —。PD decode响应的缓存字段不代表prefill实际命中。"
                  "CSV 保留完整指标（延迟单位为秒，比例范围 0–1）。每版本仅一次正式测量，无置信区间。", "",
                  "## 运行条件", "", "| 系统 | time-scale | 请求数 | 预热 | 中断 | trace |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for summary, _ in runs:
        lines.append(f"| {summary['label']} | {summary['time_scale']} | {summary['planned_requests']} | "
                     f"{summary['warmup']} | {summary['interrupted']} | {summary['trace']} |")
    lines.extend(["", "比较要求相同 trace、time-scale 和请求数量；不同版本/系统的源码、后端、KV容量和配置差异必须结合主报告解释。",
                  "", f"![TTFT CDF]({prefix}ttft_cdf.png)", "", f"![TPOT CDF]({prefix}tpot_cdf.png)", "",
                  f"![E2E CDF]({prefix}e2e_cdf.png)", "",
                  f"![TTFT timeline]({prefix}ttft_over_time.png)", "", f"![SLO attainment]({prefix}slo_attainment.png)", ""])
    (output / f"{prefix}summary.md").write_text("\n".join(lines))
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with (output / f"{prefix}summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def save_figure(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / name, dpi=160)
    fig.savefig(output / Path(name).with_suffix(".svg"))
    plt.close(fig)


def plot_runs(runs, output, prefix=""):
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
    for metric, scale, xlabel in (("ttft", 1, "TTFT (s)"), ("tpot", 1000, "TPOT (ms)"), ("e2e", 1, "E2E (s)")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for summary, records in runs:
            values = np.sort([r[metric] * scale for r in records if (metric != "tpot" or r["success"]) and r[metric] is not None])
            if len(values):
                ax.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values) + 1) / (len(values) if metric == "tpot" else summary["planned_requests"])],
                        where="post", label=f"{summary['label']} (finite={len(values)}, n={len(values) if metric == 'tpot' else summary['planned_requests']})")
        ax.set(xlabel=xlabel, ylabel="CDF (completed TPOT samples)" if metric == "tpot" else "CDF (all planned; missing tail = >cutoff)", ylim=(0, 1.01))
        if prefix:
            ax.set_xscale("log")
            ax.set_xlabel(xlabel + ", log scale")
        if ax.lines:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No valid observations", transform=ax.transAxes, ha="center")
        save_figure(fig, output, f"{prefix}{metric}_cdf.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for summary, records in runs:
        valid = [r for r in records if r["ttft"] is not None]
        ax.scatter([r["arrival"] for r in valid], [r["ttft"] for r in valid], s=9, alpha=0.55, label=summary["label"])
    ax.set(xlabel="Scheduled arrival after replay origin (s)", ylabel="TTFT (s)")
    ax.legend()
    if prefix:
        ax.set_yscale("log")
        ax.set_ylabel("TTFT (s), log scale")
    save_figure(fig, output, f"{prefix}ttft_over_time.png")

    keys = list(runs[0][0]["slo"])
    fig, ax = plt.subplots(figsize=(16, 6))
    positions = np.arange(len(keys))
    width = 0.8 / len(runs)
    for i, (summary, _) in enumerate(runs):
        values = [100 * (summary["slo"][key]["rate"] or 0) for key in keys]
        ax.bar(positions - 0.4 + width * (i + 0.5), values, width=width, label=summary["label"])
    names = [key.replace("ttft_le_", "TTFT≤").replace("tpot_le_", "TPOT≤").replace("_and_", " & ") for key in keys]
    ax.set_xticks(positions, names, rotation=50, ha="right")
    ax.set(ylabel="SLO attainment (% of all planned requests)", ylim=(0, 105))
    ax.legend()
    save_figure(fig, output, f"{prefix}slo_attainment.png")


def prepare_matrix(runs, results_dir):
    indexed = {s["label"]: (s, records) for s, records in runs}
    if set(indexed) != set(MATRIX):
        raise ValueError(f"Matrix requires exactly 15 final labels; "
                         f"missing={set(MATRIX)-set(indexed)}, extra={set(indexed)-set(MATRIX)}")
    ordered = []
    for label, (system, configuration) in MATRIX.items():
        summary, records = indexed[label]
        if label in HISTORICAL and not summary.get("retrospective_cutoff"):
            raise ValueError(f"{label}: load the historical _cut300 summary")
        if summary["planned_requests"] != 1551 or summary["time_scale"] != 1:
            raise ValueError(f"{label}: inconsistent formal workload")
        if summary.get("drain_timeout_sec") != 300:
            raise ValueError(f"{label}: expected 300s drain")
        marker = json.loads((results_dir / f"{label}.formal-started.json").read_text())
        log_dir = Path(marker["log_dir"])
        meta = json.loads((log_dir / "run.json").read_text())
        workers = [c for c in meta["commands"] if c["name"] != "router"]
        kv = {}
        memory = {}
        for worker in workers:
            name, argv = worker["name"], worker["argv"]
            role = (argv[argv.index("--disaggregation-mode") + 1]
                    if "--disaggregation-mode" in argv else "model")
            kv[f"{name}({role})"] = meta["server_info"][name]["max_total_num_tokens"]
            memory[name] = float(argv[argv.index("--mem-fraction-static") + 1])
        summary = dict(summary)
        summary["matrix"] = {
            "system": system, "configuration": configuration,
            "kv_capacity_tokens_per_instance": kv, "mem_fraction": memory,
            "log_dir": str(log_dir),
            "retrospective_cutoff": label in HISTORICAL,
        }
        ordered.append((summary, records))
    return ordered


def select_best(runs):
    systems = dict.fromkeys(system for system, _ in MATRIX.values())
    best = []
    for system in systems:
        candidates = [run for run in runs if run[0]["matrix"]["system"] == system]
        def rank(run):
            s = run[0]
            p50 = s["latencies_sec"]["ttft"]["p50"]
            return (-s["slo"][MAIN_SLO]["rate"],
                    p50 if isinstance(p50, (float, int)) else math.inf)
        best.append(min(candidates, key=rank))
    return best


def matrix_table(runs):
    headers = ["系统", "配置", "label", "完成/未完/失败",
               "TTFT p50/p90/p99(s)", "TPOT p50/p99(ms); n", "E2E p50(s)",
               "输入/输出 tok/s", "主 SLO", "TTFT≤2s", "10s/100ms SLO",
               "cache命中率†", "每实例KV(token)"]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for summary, _ in runs:
        m, latency, slo = summary["matrix"], summary["latencies_sec"], summary["slo"]
        values = [
            m["system"], m["configuration"], summary["label"],
            "/".join(str(summary[k]) for k in ("completed", "unfinished", "failed")),
            "/".join(format_number(latency["ttft"][q]) for q in ("p50", "p90", "p99")),
            "/".join(format_number(latency["tpot"][q], 1000) for q in ("p50", "p99"))
            + f"; n={latency['tpot']['count']}",
            format_number(latency["e2e"]["p50"]),
            "/".join(format_number(summary["throughput"][k + "_tokens_per_sec"]) for k in ("input", "output")),
            format_number(slo[MAIN_SLO]["rate"], 100, "%"),
            format_number(slo["ttft_le_2s"]["rate"], 100, "%"),
            format_number(slo["ttft_le_10s_and_tpot_le_100ms"]["rate"], 100, "%"),
            format_number(summary["cached_tokens"]["token_weighted_ratio"], 100, "%")
            + ("†" if m["system"] == "PD" else ""),
            "; ".join(f"{k}={v}" for k, v in m["kv_capacity_tokens_per_instance"].items()),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_matrix_tables(runs, output, prefix):
    lines = [
        f"# {'完整配置矩阵（15行）' if prefix == 'matrix_' else '各系统最佳配置（4行）'}",
        "",
        "主SLO：TTFT≤5s且TPOT≤100ms；SLO分母始终为1551。按主SLO降序选择系统最佳，同分看TTFT p50。",
        "",
        matrix_table(runs), "",
        "TTFT/E2E分位包括+∞（显示 >cutoff）；TPOT只取截止前成功完成且输出>1的样本，n列为样本数。",
        "最后实际发送后排空300s。输入/输出吞吐按截止前客户端交付量统计；输入含缓存。",
        "cache为截止前完成且包含cached_tokens的响应的输入token加权比例。† PD此值来自decode响应，"
        "不能视作prefill实际命中率。KV按独立模型实例列出；TP/PP各rank的分片不能相加。",
        "每配置一次正式测量，无置信区间；历史四个配置使用_cut300重算。CSV延迟用秒，比例用0–1。",
        "",
    ]
    if prefix == "matrix_":
        lines += ["MuxWise原版不支持PP2DP2、PP4、TP2PP2。MuxWise-PP仅TP1，"
                  "不支持TP2DP2、TP4、TP2PP2；PP1DP4归原版MuxWise。PD使用P/D专用配置列。",
                  "", "![主SLO矩阵](matrix_slo_heatmap.png)", ""]
    else:
        lines += ["![最佳配置TTFT CDF](best_ttft_cdf.png)", "",
                  "![最佳配置TPOT CDF](best_tpot_cdf.png)", "",
                  "![最佳配置主SLO](best_main_slo.png)", ""]
    (output / f"{prefix}summary.md").write_text("\n".join(lines))
    flattened = [flatten(s) for s, _ in runs]
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with (output / f"{prefix}summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def plot_matrix(runs, best, output):
    systems = list(dict.fromkeys(system for system, _ in MATRIX.values()))
    configurations = ["DP4", "TP2DP2", "TP4", "PP2DP2", "PP4", "TP2PP2",
                      "1P1D(TP2)", "2P2D(TP1)", "3P1D(TP1)", "1P3D(TP1)"]
    values = np.full((len(systems), len(configurations)), np.nan)
    failed_cells = set()
    for summary, _ in runs:
        m = summary["matrix"]
        values[systems.index(m["system"]), configurations.index(m["configuration"])] = 100 * summary["slo"][MAIN_SLO]["rate"]
        if summary["failed"]:
            failed_cells.add((systems.index(m["system"]), configurations.index(m["configuration"])))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i, system in enumerate(systems):
        for j, configuration in enumerate(configurations):
            value = values[i, j]
            if np.isfinite(value):
                text = f"{value:.2f}%" + ("*" if (i, j) in failed_cells else "")
            elif ((system == "MuxWise" and configuration in ("PP2DP2", "PP4", "TP2PP2"))
                  or (system == "MuxWise-PP" and configuration in ("TP2DP2", "TP4", "TP2PP2"))):
                text = "Unsupported"
            elif system == "MuxWise-PP" and configuration == "DP4":
                text = "See MuxWise"
            else:
                text = "N/A"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(value) and value > 55 else "#222222")
    ax.set_xticks(range(len(configurations)), configurations, rotation=25, ha="right")
    ax.set_yticks(range(len(systems)), systems)
    ax.set_title("Main SLO: TTFT ≤ 5 s & TPOT ≤ 100 ms (all 1551 requests)")
    ax.set_xlabel("* Formal request failures occurred; retained in the full denominator.")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Attainment (%)")
    save_figure(fig, output, "matrix_slo_heatmap.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [s["matrix"]["system"] + "\n" + s["matrix"]["configuration"] for s, _ in best]
    values = [100 * s["slo"][MAIN_SLO]["rate"] for s, _ in best]
    bars = ax.bar(names, values, color=["#426b98", "#d69a36", "#9c625e", "#4b9a86"])
    for bar, (summary, _) in zip(bars, best):
        v = summary["slo"][MAIN_SLO]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                f"{100*v['rate']:.3f}%\n{v['passed']}/1551", ha="center", va="bottom")
    ax.set(ylabel="Main SLO attainment (%)", ylim=(0, 105),
           title="Best configuration per system (one measurement per configuration)")
    save_figure(fig, output, "best_main_slo.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="*", help="summary JSON paths or quoted glob patterns")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    parser.add_argument("--final-labels", nargs="+", help="Also generate final_summary.md/csv and final_*.png for exactly these labels")
    parser.add_argument("--matrix", action="store_true", help="Require all 15 final configurations; generate matrix/best tables and plots")
    parser.add_argument("--results-dir", type=Path, help="Directory containing formal locks (defaults to first input's directory)")
    args = parser.parse_args()
    if not args.summaries:
        if not args.matrix:
            parser.error("provide summaries or use --matrix for the 15 frozen final configurations")
        results_dir = args.results_dir or Path(args.out_dir)
        args.summaries = [str(results_dir / f"{label}{'_cut300' if label in HISTORICAL else ''}_summary.json")
                          for label in MATRIX]
    runs = load_runs(args.summaries)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.matrix:
        results_dir = args.results_dir or Path(glob.glob(args.summaries[0])[0]).resolve().parent
        runs = prepare_matrix(runs, results_dir)
        best = select_best(runs)
        write_matrix_tables(runs, output, "matrix_")
        write_matrix_tables(best, output, "best_")
        plot_runs(best, output, "best_")
        plot_matrix(runs, best, output)
        print(f"Matrix: {len(runs)} rows; best: {len(best)} rows -> {output}")
        return
    write_tables(runs, output)
    plot_runs(runs, output)
    print(f"Compared {len(runs)} runs: {output / 'summary.md'}, summary.csv and 5 PNG plots")
    if args.final_labels:
        assert len(args.final_labels) == len(set(args.final_labels)), "Duplicate final labels"
        indexed = {summary["label"]: (summary, records) for summary, records in runs}
        missing = set(args.final_labels) - indexed.keys()
        if missing:
            raise ValueError(f"Final labels have no formal result: {sorted(missing)}")
        final_runs = [indexed[label] for label in args.final_labels]
        write_tables(final_runs, output, prefix="final_")
        plot_runs(final_runs, output, prefix="final_")
        print(f"Final comparison: {', '.join(args.final_labels)} -> {output / 'final_summary.md'}")


if __name__ == "__main__":
    main()
