#!/usr/bin/env python3
"""Render measured lane timings, admission histograms and residency from saved logs."""
import json
from pathlib import Path

ROOT = Path('/root/workspace/muxpp_work/logs/p9a')
parts = ['128/0', '120/8', '112/16', '104/24', '96/32', '80/48', '64/64', '0/128']
lines = ['# P9a stage 与准入证据', '',
    '严格到达窗口内的首/末10秒summary差分；每轮实际窗口见下表。',
    'CPU lane是状态驻留；CUDA事件含stream上的等待，prefill/decode会重叠，不能相加当作GPU利用率。',
    '每个`analysis.json`另含完整回放窗口、逐GPU利用率/显存、逐请求等待分布及分位分解。', '']
for p in sorted(ROOT.glob('r*-*/analysis.json')):
    a = json.loads(p.read_text())
    if 'arrival' not in a:
        continue
    window = a['arrival']
    lines += ['## ' + p.parent.name, '', f"窗口 Unix {window['window'][0]:.0f}–{window['window'][1]:.0f}", '',
              '| stage | 窗口s | CUDA累计s | prefill lane驻留s | decode lane驻留s |',
              '| --- | ---: | --- | --- | --- |']
    def fmt(values, scale=1):
        return ', '.join(f'{k}={v*scale:.3f}' for k,v in sorted(values.items())) or '—'
    for stage, d in window['stages'].items():
        factor = d['seconds']/100
        lines += [f"| {stage} | {d['seconds']:.3f} | {fmt(d['cuda_percent'],factor)} | {fmt(d['wall_percent']['prefill'],factor)} | {fmt(d['wall_percent']['decode'],factor)} |"]
    lines += ['', '| stage | ' + ' | '.join(parts) + ' |',
              '| --- | ' + ' | '.join(['---:']*len(parts)) + ' |']
    for stage, d in window['stages'].items():
        lines += ['| ' + stage + ' | ' + ' | '.join(f"{d['partition_percent'].get(str(i),0):.3f}%" for i in range(8)) + ' |']
    adm = window['admission']
    lines += ['', '| 准入原因 | blocked次数 | decisions次数 | wall s |', '| --- | ---: | ---: | ---: |']
    for key in sorted(adm['blocked'].keys() | adm['decisions'].keys() | adm['wall'].keys()):
        lines += [f"| {key} | {adm['blocked'].get(key,0)} | {adm['decisions'].get(key,0)} | {adm['wall'].get(key,0):.3f} |"]
    lines += ['', 'batch停止原因计数：`' + json.dumps(adm['stops'],sort_keys=True) + '`。', '']
    if a.get('slice_planning'):
        lines += ['slice计划计数：`' + json.dumps(a['slice_planning'],sort_keys=True) + '`。deferred_layers仅是本slice延后计算的层数，不是省略计算。', '']
    for stage,d in window['stages'].items():
        if 'admission_budget_eval' in d:
            lines += [f"stage {stage} 准入模型：`{json.dumps(d['admission_budget_eval'],sort_keys=True)}`。", '']
(ROOT / 'stage-details.md').write_text('\n'.join(lines)+'\n')
print('Wrote', ROOT / 'stage-details.md')
