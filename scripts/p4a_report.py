#!/usr/bin/env python3
"""Append the P4a result to the preserved P3b report without rewriting raw data."""
import collections
import datetime
import json
from pathlib import Path
import re
import shlex
import statistics

ROOT = Path('/root/workspace/muxpp_work')


def read(path):
    return json.loads(path.read_text())


def fmt(x, scale=1):
    return '—' if x is None else f'{x*scale:.3f}'


def row(values):
    return '| '+' | '.join(map(str, values))+' |'


def main():
    s = read(ROOT/'results/sglang_pd_r2_summary.json')
    marker = read(ROOT/'results/sglang_pd_r2.formal-started.json')
    d = Path(marker['log_dir'])
    m = read(d/'run.json')
    gate_dir = Path(m['gate'])
    gate = read(gate_dir/'run.json')
    g = gate['precheck300_summary']
    # Preserve sections added by other workspace tasks while this long run drains.
    report = (ROOT/'RESULTS.md').read_text()
    report = re.sub(r'\n## P4a：sglang_pd_r2（用户批准的唯一一次修复重跑）\n.*?(?=\n## |\Z)', '', report, flags=re.S)
    report = '\n'.join(line for line in report.splitlines() if not line.startswith('| sglang_pd_r2 |'))+'\n'
    if 'P4a PD 修复重跑' not in report.splitlines()[0]:
        report = report.replace(report.splitlines()[0], report.splitlines()[0]+'与 P4a PD 修复重跑', 1)
    original_timestamp = re.search(r'生成时间：([^。]+)', (ROOT/'logs/p4a/diagnosis/RESULTS.before.md').read_text())[1]
    report = re.sub(r'^(?:生成时间|P3b原报告生成时间)：[^\n]+',
                    'P3b原报告生成时间：'+original_timestamp+'；P4a更新时间：'+datetime.datetime.now(datetime.timezone.utc).isoformat()+'。',
                    report, count=1, flags=re.M)
    report = report.replace('四系统各一次，不补跑。', 'P3b四系统各一次；P4a用户另批准PD修复后唯一一次`sglang_pd_r2`，其余系统结果不变。', 1)
    if 'P3b首轮PD传输包为' not in report:
        report = report.replace('PD传输包为', 'P3b首轮PD传输包为', 1)
    report = report.replace('未安装或使用NIXL。', 'P3b阶段未安装或使用NIXL；P4a候选验证见下文。', 1)
    report = report.replace('每系统只有一轮正式测量，无置信区间；', '原四系统各一轮正式测量，PD另有不同修复配置的一次r2；不能将这两轮作为同配置重复样本，无置信区间；', 1)
    old = '| sglang_pd | 28 | 1523 |'
    report = report.replace(old, '| sglang_pd（首轮失败，保留） | 28 | 1523 |', 1)
    main_row = row(['sglang_pd_r2', s['completed_requests'], s['failed_requests'], s['unsent_requests'], fmt(s['duration_sec']), fmt(s['throughput']['input_tokens_per_sec']), fmt(s['throughput']['output_tokens_per_sec']), fmt(s['slo']['ttft_le_5s']['rate'],100)+'%', fmt(s['slo']['tpot_le_100ms']['rate'],100)+'%', fmt(s['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'],100)+'%', fmt(s['cached_tokens']['token_weighted_ratio'],100)+'%'])
    anchor = '\n\ncache来自成功响应'
    report = report.replace(anchor, '\n'+main_row+anchor, 1)
    latency_rows = []
    for metric in ['ttft','tpot','e2e']:
        stats = s['latencies_sec'][metric]
        scale = 1000 if metric == 'tpot' else 1
        latency_rows.append(row(['sglang_pd_r2', metric.upper()+(' (ms)' if scale==1000 else ' (s)'), stats['count']]+[fmt(stats[k],scale) for k in ['mean','p50','p90','p95','p99','max']]))
    anchor = '\n\n全部2/5/10s'
    report = report.replace(anchor, '\n'+'\n'.join(latency_rows)+anchor, 1)
    start = s['run_started_unix']
    end = start+s['duration_sec']
    new = hit = batches = 0
    resources = collections.defaultdict(list)
    for job in read(d/'commands.json'):
        if job['name'] == 'router':
            continue
        for line in (d/(job['name']+'.log')).read_text(errors='replace').splitlines():
            match = re.search(r'#new-token: (\d+), #cached-token: (\d+)',line)
            if match:
                stamp = datetime.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').timestamp()
                if start <= stamp <= end+1:
                    new += int(match[1]); hit += int(match[2]); batches += 1
    for line in (d/'gpu.csv').read_text().splitlines()[1:]:
        fields = [x.strip() for x in line.split(',')]
        stamp = datetime.datetime.strptime(fields[0],'%Y/%m/%d %H:%M:%S.%f').timestamp()
        if start <= stamp <= end:
            resources[fields[1]].append((float(fields[2].split()[0]),float(fields[4].split()[0])))
    rows = list(map(json.loads,(ROOT/'results/sglang_pd_r2_requests.jsonl').open()))
    errors = collections.Counter(r.get('error') for r in rows if not r['success'])
    title = 'Mooncake TCP 调整队列/准入/分片/并发参数' if m['profile']=='mooncake-tuned' else 'NIXL 1.5.0 / UCX host中转 + TCP/shared memory'
    rel = d.relative_to(ROOT)
    grel = gate_dir.relative_to(ROOT)
    text = [
        '', '## P4a：sglang_pd_r2（用户批准的唯一一次修复重跑）', '',
        f'最终方案：**{title}**。2P2D、TP1，P=GPU0/1、D=GPU2/3。正式模型/调度/显存/attention/graph参数保持P3b；没有修改任何MuxWise-PP或模型推理源码。隔离源码提交：`{m["git_commit"]}`，路径`{m["source_dir"]}`，相对P3b仅新增PD参数文件；原muxwise-pp分支参数提交为`117c2c745`、`5eb455f5b`、`4055e4de9`。隔离用于避开另一个任务的并行源码修改，没有触碰其文件。', '',
        '根因：原Mooncake 0.3.13.post1 TCP每peer默认1024队列槽+1024待准入槽、准入超时1秒。SGLang将碎片化KV的64个K/V层合成大批次，首轮后续失败批次达到6016个描述符，并明确记录TCP lane queue-full rejection；一次底层失败又使SGLang将该session持续标记为失效（新KVArgs注册可清除，但普通后续请求不会自动清除），最终放大为1523条请求失败。最早1792/1984描述符批次疑似准入超时，但旧日志不能唯一确定该次底层状态；没有30秒API超时、GPU拷贝错误、单次字节尺寸限制或OOM的直接证据。TCP源码已有GPU→host→TCP→host→GPU中转，不要求P2P。', '',
        '完整版本匹配源码、原始日志行号与证据：[ROOT_CAUSE.md](logs/p4a/diagnosis/ROOT_CAUSE.md)、[失败批次提取](logs/p4a/diagnosis/first-run-failure-evidence.json)。MC_NUM_TCP_IO_THREADS等本机main新增但wheel未支持的参数没有使用。NUMA布局保持原样：random均匀路由下孤立GPU3不论作为P或D，约一半路径仍需跨NUMA。', '',
        '候选审计：先试Mooncake参数，后试NIXL（安装记录：[install-nixl.log](logs/p4a/install-nixl.log)）。两者最初都通过2条长请求，但旧随机token短请求各19/20。随后普通TP1自身也复现19/20，且Mooncake分歧文本与普通TP1分歧完全一致。旧测试结果完整保留，不能据此直接认定KV损坏。修正验证oracle后重新按a→b顺序选择首个通过全部门槛的方案。', '',
        '短请求门槛使用预先固定的20条118-token上下文记录选择/复制用例，输出32 tokens：[固定输入](logs/p4a/text20-cases.json)。在这组用例的第一次PD测试前写入，未按PD输出筛选；四个普通TP1实例均20/20全文和token ID一致：[普通TP1验证](logs/p4a/normal-text20/comparison.json)。原20条随机ID诊断与不匹配记录在[normal-diagnostic](logs/p4a/normal-diagnostic)、[Mooncake初试](logs/p4a/a-mooncake-tuned)、[NIXL初试](logs/p4a/b-nixl-host)。', '',
        f'通过的预检目录：[{grel}]({grel})。20/20短输出精确一致；2条31,663-token长请求成功；正式trace前300条、time-scale=1、warmup2/force-close：**{g["completed_requests"]}/300成功、{g["failed_requests"]}失败**，总时长{fmt(g["duration_sec"])}s。预检结果只存logs/p4a，不混入正式结果。', '',
        f'正式目录：[{rel}]({rel})。重新启动后再次通过20条短请求、2条长请求及P3b 20条流式冒烟；随后flush四后端cache、重启router，排他创建`sglang_pd_r2.formal-started.json`，只执行一次。完整命令如下（五个服务并发启动，由生命周期脚本管理）：', '',
        f'正式回放起点：`{s["run_started_at"]}`；生命周期结束：`{m["finished_utc"]}`。主机日志为CST，分析按本机时区转换为Unix时间后与客户端UTC对齐。', '',
        '```bash', (d/'commands.sh').read_text().strip(), '```', '',
        '生命周期入口（完整预检gate必须通过，已存在正式标记时拒绝重跑）：', '', '```bash', 'source /root/workspace/muxpp_work/env.sh', 'scripts/run_experiment.sh sglang_pd_r2 --profile '+m['profile']+' --tag '+d.name+' --gate '+str(gate_dir), '```', '', '正式客户端（默认time-scale=1，timeout=3600秒，无重试）：', '',
        '```bash', shlex.join(read(d/'formal-command.json')), '```', '',
        f'正式结果：**{s["completed_requests"]}/1551成功、{s["failed_requests"]}失败、{s["unsent_requests"]}未发送**；总时长{fmt(s["duration_sec"])}s，输入{fmt(s["throughput"]["input_tokens_per_sec"])} tok/s，输出{fmt(s["throughput"]["output_tokens_per_sec"])} tok/s；联合SLO={s["slo"]["ttft_le_5s_and_tpot_le_100ms"]["passed"]}/1551={fmt(s["slo"]["ttft_le_5s_and_tpot_le_100ms"]["rate"],100)}%。延迟分位已加入上方主表。', '',
        f'响应cached_tokens加权命中={fmt(s["cached_tokens"]["token_weighted_ratio"],100)}%；prefill侧正式窗口日志共{batches}批、新token={new}、命中token={hit}，服务端cache比例={fmt(hit/(new+hit) if new+hit else None,100)}%。PD decode响应cache口径不能替代prefill实际命中。', '',
        '正式窗口GPU采样（每5秒）：', '',
        row(['GPU','峰值显存 MiB','平均GPU利用率 %']), row(['---','---','---']),
    ]
    for gpu, samples in sorted(resources.items()):
        text.append(row([gpu,fmt(max(x[0] for x in samples)),fmt(statistics.mean(x[1] for x in samples))]))
    text += ['', f'客户端失败分类：`{dict(errors)}`。formal_returncode={m["formal_returncode"]}；生命周期PASS不替代上述请求失败统计。', '',
             f'客户端send_lag p99={fmt(s["latencies_sec"]["send_lag"]["p99"],1000)}ms、max={fmt(s["latencies_sec"]["send_lag"]["max"],1000)}ms。四worker日志均未出现批量传输失败、队列拒绝、KVTransferError、API传输超时或CUDA OOM：[传输错误核验]({rel}/transfer-error-audit.json)。', '',
             f'可靠性恢复后，性能仍受持续排队限制：TTFT p50={fmt(s["latencies_sec"]["ttft"]["p50"])}s，TPOT mean={fmt(s["latencies_sec"]["tpot"]["mean"],1000)}ms；不能将0失败理解为达到交互延迟目标。此处没有对各个TCP参数做消融，不能分离其各自贡献。', '',
             '结果比较中继续保留原`sglang_pd`首轮失败数据；r2是修复后的新配置，不能将修复增益单独归因于PD架构。`summarize.py`已重新包含全部五个label。', '',
             '只读核验：[P4a final-audit.json](logs/p4a/final-audit.json)。逐条核对1551请求、正式唯一标记、300条门槛、统一参数、原始文件SHA256、GPU采样和最终服务/GPU清理。', '']
    (ROOT/'RESULTS.md').write_text(report+'\n'.join(text))
    summary_path = ROOT/'results/summary.md'
    summary_text = summary_path.read_text().replace('| sglang_pd |', '| sglang_pd（首轮失败） |', 1)
    note = 'P4a说明：sglang_pd保留首轮失败配置（28/1551成功）；sglang_pd_r2为修复后用户批准的唯一一次正式运行。首轮成功子集延迟不能代表整体可用性。完整命令、根因、预检与代码隔离记录见[RESULTS.md](../RESULTS.md)。'
    if note not in summary_text:
        summary_text += '\n'+note+'\n'
    summary_path.write_text(summary_text)
    print(main_row)


if __name__ == '__main__':
    main()
