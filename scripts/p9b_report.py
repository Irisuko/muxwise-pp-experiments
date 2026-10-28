#!/usr/bin/env python3
"""CPU-only P9b tables/figures from immutable saved request events."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
AUDIT = ROOT / 'logs/p9b'
ARCHIVE = ROOT / 'archive/superseded/p9b'
sys.path.insert(0, str(ROOT / 'bench'))
import summarize as sm
import numpy as np

BASE_MATRIX = dict(sm.MATRIX)
BASE_MATRIX['muxpp_pp2tp2'] = ('MuxWise-PP', 'PP2×TP2')
CONV = {
    'conv_muxwise_tp2dp2': ('MuxWise', 'TP2DP2'),
    'conv_muxwise_tp4': ('MuxWise', 'TP4'),
    'conv_chunked_pp2dp2': ('Chunked', 'PP2DP2'),
    'conv_chunked_tp2dp2': ('Chunked', 'TP2DP2'),
    'conv_muxpp_v4_pp2tp2': ('MuxWise-PP', 'PP2×TP2 v4'),
    'conv_muxpp_v4_pp4': ('MuxWise-PP', 'PP4 v4'),
}
PAIRS = {'muxpp_pp2tp2':'muxpp_v4_pp2tp2', 'muxwise_pp_v3':'muxpp_v4_pp4',
         'muxpp_pp2dp2':'muxpp_v4_pp2dp2'}
NEW = {new:('MuxWise-PP', BASE_MATRIX[old][1]+' v4') for old,new in PAIRS.items()}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def resolve(path):
    if path.exists():
        return path
    return ARCHIVE / path.relative_to(ROOT)


def load(path):
    return json.loads(resolve(path).read_text())


def summary_path(label):
    return resolve(OUT / f'{label}{"_cut300" if label in sm.HISTORICAL else ""}_summary.json')


def gpu_stats(path, summary):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f, skipinitialspace=True):
            try:
                t = dt.datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(
                    tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
                if summary['run_started_unix'] <= t <= summary['run_started_unix'] + summary['duration_sec']:
                    rows.append((int(r['index']), float(r['utilization.gpu [%]'].split()[0]),
                                 float(r['memory.used [MiB]'].split()[0])))
            except (ValueError, KeyError):
                continue
    return {'sampling_interval_sec':5, 'window':'client measurement incl. drain, excluding warmup',
            'samples':len(rows), 'gpu_util_mean_pct':float(np.mean([r[1] for r in rows])) if rows else None,
            'per_gpu':{str(i):{'samples':len([r for r in rows if r[0]==i]),
                'util_mean_pct':float(np.mean([r[1] for r in rows if r[0]==i])),
                'memory_peak_mib':max(r[2] for r in rows if r[0]==i)} for i in sorted({r[0] for r in rows})}}


def prepare(mapping, expected):
    runs = sm.load_runs([str(summary_path(l)) for l in mapping])
    result = []
    for s, records in runs:
        raw = load(summary_path(s['label']))
        for key in ['completed','unfinished','failed','latencies_sec','slo','throughput','cached_tokens']:
            assert raw[key] == s[key], (s['label'], key)
        assert s['planned_requests'] == expected and s['time_scale'] == 1
        assert s['drain_timeout_sec'] == 300
        marker = load(OUT / f'{s["label"]}.formal-started.json')
        folder = resolve(Path(marker['log_dir']))
        run = load(folder / 'run.json')
        kv, mem = {}, {}
        for worker in run['commands']:
            name, cmd = worker['name'], worker['argv']
            if name == 'router':
                continue
            role = cmd[cmd.index('--disaggregation-mode')+1] if '--disaggregation-mode' in cmd else 'model'
            kv[f'{name}({role})'] = run['server_info'][name]['max_total_num_tokens']
            mem[name] = float(cmd[cmd.index('--mem-fraction-static')+1])
        system, config = mapping[s['label']]
        s['matrix'] = dict(system=system, configuration=config, kv_capacity_tokens_per_instance=kv,
                           mem_fraction=mem, log_dir=str(folder),
                           retrospective_cutoff=s['label'] in sm.HISTORICAL)
        s['gpu_measurement'] = gpu_stats(folder / 'gpu.csv', s)
        breakdown = dict(passed=0,ttft_only=0,tpot_only=0,both=0,undefined_tpot=0,not_completed=0,
                         unsent=s['planned_requests']-len(records))
        for record in records:
            if not record['success']:
                key='not_completed'
            elif record['tpot'] is None:
                key='undefined_tpot'
            else:
                tt=record['ttft'] is None or record['ttft']>5
                tp=record['tpot']>.1
                key='both' if tt and tp else 'ttft_only' if tt else 'tpot_only' if tp else 'passed'
            breakdown[key]+=1
        assert sum(breakdown.values())==s['planned_requests']
        assert breakdown['passed']==s['slo'][sm.MAIN_SLO]['passed']
        s['main_slo_breakdown']=breakdown
        result.append((s,records))
    return result


def best(runs):
    systems = list(dict.fromkeys(s['matrix']['system'] for s,_ in runs))
    def rank(run):
        s = run[0]
        p50 = s['latencies_sec']['ttft']['p50']
        return (-s['slo'][sm.MAIN_SLO]['rate'], p50 if isinstance(p50,(int,float)) else math.inf)
    return [min((r for r in runs if r[0]['matrix']['system']==system), key=rank) for system in systems]


def tables(runs, prefix, title):
    sm.write_tables(runs, OUT, prefix)
    detail = (OUT / f'{prefix}summary.md').read_text().split('## 完整延迟分位',1)[1]
    lines = [f'# {title}', '', sm.matrix_table(runs), '',
             '主SLO：TTFT≤5s且TPOT≤100ms。每配置一次测量，无置信区间。'
             f'SLO/TTFT/E2E分母为全部{s["planned_requests"] if (s:=runs[0][0]) else 0}条计划请求；'
             'TPOT仅取已完成且输出>1的样本。吞吐包含排空，输入包含缓存。'
             'KV为每实例逻辑容量，不乘TP/PP rank数。† PD cache仅为decode响应。', '',
             '## 运行配置与GPU采样', '', '| label | mem fraction | GPU均值% | 样本数 | 发送延迟p99/max(ms) | 正式日志 |',
             '| --- | --- | --- | --- | --- | --- |']
    for s,_ in runs:
        g,m = s['gpu_measurement'],s['matrix']
        lines.append(f'| {s["label"]} | {m["mem_fraction"]} | {sm.format_number(g["gpu_util_mean_pct"])} | '
                     f'{g["samples"]} | '+ '/'.join(sm.format_number(s['latencies_sec']['send_lag'][q],1000) for q in ['p99','max'])+
                     f' | {m["log_dir"]} |')
    lines += ['', 'GPU每5秒采样，仅取客户端测量窗口（包括排空，不含长预热/冒烟）；'
              'GPU利用率包含通信等待，不能等同于有效计算效率。', '', '## 完整延迟分位'+detail]
    (OUT / f'{prefix}summary.md').write_text('\n'.join(lines))
    save(OUT / f'{prefix}summary.json', [s for s,_ in runs])


def barplot(runs, name, title):
    fig, ax = sm.plt.subplots(figsize=(max(9,len(runs)*1.8),5.7))
    names = [s['matrix']['system']+'\n'+s['matrix']['configuration'] for s,_ in runs]
    values = [100*s['slo'][sm.MAIN_SLO]['rate'] for s,_ in runs]
    colors = {'MuxWise':'#d69a36','Chunked':'#426b98','MuxWise-PP':'#4b9a86','PD':'#9c625e'}
    bars = ax.bar(names, values, color=[colors[s['matrix']['system']] for s,_ in runs])
    for b,(s,_) in zip(bars,runs):
        v=s['slo'][sm.MAIN_SLO]
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,
                f'{100*v["rate"]:.3f}%\n{v["passed"]}/{v["total"]}',ha='center',fontsize=9)
    ax.set(ylabel='Main SLO attainment (%)',ylim=(0,110),title=title)
    ax.grid(axis='x',visible=False)
    sm.save_figure(fig,OUT,name)


def heatmap(runs):
    columns=['DP4','TP2DP2','TP4','PP2DP2','PP4','PP2×TP2','1P1D(TP2)','2P2D(TP1)','3P1D(TP1)','1P3D(TP1)']
    systems=list(dict.fromkeys(s['matrix']['system'] for s,_ in runs))
    values=np.full((len(systems),len(columns)),np.nan)
    texts={}
    for s,_ in runs:
        m=s['matrix']; config=m['configuration'].replace(' v3','').replace(' v4','').replace('TP2PP2','PP2×TP2')
        i,j=systems.index(m['system']),columns.index(config)
        val=100*s['slo'][sm.MAIN_SLO]['rate']
        values[i,j]=max(values[i,j],val) if np.isfinite(values[i,j]) else val
        version='v4: ' if 'v4' in s['label'] else 'v3: ' if s['label'] in PAIRS else ''
        texts.setdefault((i,j),[]).append(version+f'{val:.2f}%'+('*' if s['failed'] else ''))
    fig,ax=sm.plt.subplots(figsize=(15,5.4))
    cmap=sm.plt.get_cmap('YlGnBu').copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=0,vmax=100,aspect='auto')
    for i in range(len(systems)):
        for j in range(len(columns)):
            txt='\n'.join(texts.get((i,j),['N/A']))
            ax.text(j,i,txt,ha='center',va='center',fontsize=8,color='white' if values[i,j]>55 else '#222')
    ax.set_xticks(range(len(columns)),[c.replace('×',' x ') for c in columns],rotation=25,ha='right')
    ax.set_yticks(range(len(systems)),systems)
    ax.set_title('Tool & Agent: TTFT <= 5 s and TPOT <= 100 ms; all 1551 requests')
    ax.set_xlabel('One run per version/configuration; different dates/GPU mappings. * Request failures retained.\n'
                  'When versions coexist, cell color uses the higher listed rate; both measurements are shown.')
    ax.grid(False);fig.colorbar(im,ax=ax,label='Attainment (%)')
    sm.save_figure(fig,OUT,'matrix_slo_heatmap.png')


def comparison():
    mapping={**{k:BASE_MATRIX[k] for k in PAIRS},**NEW}
    runs=prepare(mapping,1551)
    indexed={s['label']:s for s,_ in runs}
    changes={}
    for old,new in PAIRS.items():
        a,b=indexed[old],indexed[new]
        changes[new]={'old_label':old,'main_slo_before':a['slo'][sm.MAIN_SLO]['rate'],
                      'main_slo_after':b['slo'][sm.MAIN_SLO]['rate'],
                      'main_slo_delta_pp':100*(b['slo'][sm.MAIN_SLO]['rate']-a['slo'][sm.MAIN_SLO]['rate']),
                      'before':a,'after':b}
    save(AUDIT/'v3-v4-comparison.json',changes)
    tables(runs,'p9b_v3_v4_','Tool&Agent v3 / v4（历史对照，不是同日消融）')
    return changes


def main():
    if '--compare-only' in sys.argv:
        comparison(); return
    decisions=load(AUDIT/'replacement-decisions.json')
    mapping={}
    for label,pair in BASE_MATRIX.items():
        if label in PAIRS:
            new=PAIRS[label]
            if not decisions[new]['replace']:
                mapping[label]=(pair[0],pair[1]+' v3')
            mapping[new]=NEW[new]
        else:
            mapping[label]=pair
    tool=prepare(mapping,1551)
    conv=prepare(CONV,735)
    tool_best,conv_best=best(tool),best(conv)
    for runs,prefix,title in [(tool,'matrix_',f'Tool&Agent 完整矩阵（{len(tool)}行）'),
                             (tool_best,'best_','Tool&Agent 各系统最佳配置'),
                             (conv,'conv_','Conversation 完整配置矩阵（6行）'),
                             (conv_best,'conv_best_','Conversation 三系统最佳配置')]:
        tables(runs,prefix,title)
    for runs,prefix in [(tool_best,'best_'),(conv,'conv_'),(conv_best,'conv_best_')]:
        sm.plot_runs(runs,OUT,prefix)
    heatmap(tool)
    barplot(tool_best,'best_main_slo.png','Tool & Agent: best per system (single measurement)')
    barplot(conv,'conv_main_slo.png','Conversation: six configurations (single measurement)')
    barplot(conv_best,'conv_best_main_slo.png','Conversation: best per system (single measurement)')
    # Refresh the older final-comparison aliases to the current best, avoiding stale v3 plots.
    tables(tool_best,'final_','Tool&Agent 各系统最佳配置（同best_summary）')
    tables(tool_best,'','Tool&Agent 各系统最佳配置（同best_summary）')
    sm.plot_runs(tool_best,OUT,'final_')
    sm.plot_runs(tool_best,OUT,'')
    comparison()
    save(AUDIT/'report-index.json',dict(tool_labels=list(mapping),
        tool_best=[s['label'] for s,_ in tool_best],conv_labels=list(CONV),
        conv_best=[s['label'] for s,_ in conv_best]))
    print(json.dumps({'matrix_rows':len(tool),'conv_rows':len(conv),
                     'tool_best':[s['label'] for s,_ in tool_best],
                     'conv_best':[s['label'] for s,_ in conv_best]},ensure_ascii=False))


if __name__=='__main__':
    main()
