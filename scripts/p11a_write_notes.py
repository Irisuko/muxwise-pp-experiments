#!/usr/bin/env python3
"""Render the final P11a report from audited development measurements."""
import json
from pathlib import Path
import shlex
import subprocess
import p11a_serve as serve
from p11a_analyze import ROOT, OUT, MAIN, LOOSE

LABELS = ['llama_pp4', 'llama_pp2tp2', 'big_pp2tp2', 'big_pp4']
NAMES = {'llama_pp4':'8B PP4', 'llama_pp2tp2':'8B PP2×TP2',
         'big_pp2tp2':'24B PP2×TP2', 'big_pp4':'24B PP4',
         'llama_muxwise_tp2dp2':'8B MuxWise TP2×DP2',
         'big_muxwise_tp4':'24B MuxWise TP4', 'llama_pp2dp2':'8B PP2×DP2'}


def main():
    selection=json.loads((ROOT/'configs/p11a_selected.json').read_text())
    audit=json.loads((OUT/'final-audit.json').read_text())
    gate=json.loads((OUT/'final-correctness-gate.json').read_text())
    assert not audit['git_status'] and audit['head']==selection['commit']
    assert all(not v[k]['unexpected_cases'] for v in gate.values() for k in ['serial20','cold20'])
    assert all(v['cold12288']['exact'] for v in gate.values())
    data=json.loads((OUT/'comparison.json').read_text())
    def row(rnd,label,rate):
        found=[v for v in data if v['rate']==rate and v['run'].endswith('-'+label)
               and (v['run'].startswith('baseline-') if rnd=='baseline' else v['run']==rnd+'-'+label)]
        assert len(found)==1,(rnd,label,rate,len(found))
        return found[0]
    def table(rnd,labels):
        lines=['| 配置 | 标准主/宽松SLO % | TTFT p50 s / TPOT p50 ms | 三档主SLO % | 三档输出 tok/s | 饱和时长 s | 三档goodput req/s | 标准GPU均值 % |',
               '| --- | --- | --- | --- | --- | ---: | --- | ---: |']
        for label in labels:
            rows=[row(rnd,label,rate) for rate in ['standard','double','saturated']]
            a=rows[0]['summary'];lat=a['latencies_sec']
            slo=' / '.join(f"{100*v['summary']['slo'][MAIN]['rate']:.3f}" for v in rows)
            out=' / '.join(f"{v['summary']['throughput']['output_tokens_per_sec']:.2f}" for v in rows)
            gp=' / '.join(f"{v['goodput']:.4f}" for v in rows)
            gpu=sum(v['util_mean'] for v in rows[0]['gpu'].values())/len(rows[0]['gpu'])
            lines.append(f"| {NAMES[label]} | {100*a['slo'][MAIN]['rate']:.3f} / {100*a['slo'][LOOSE]['rate']:.3f} | {lat['ttft']['p50']:.3f} / {1000*lat['tpot']['p50']:.3f} | {slo} | {out} | {rows[-1]['summary']['duration_sec']:.3f} | {gp} | {gpu:.2f} |")
        return '\n'.join(lines)
    lines=[f'''# P11a：MuxWise-PP v5 开发优化结果

最终代码 `{selection['commit']}`，分支 `muxwise-pp`；起点 `4c10135d2e2891d6fb262d4894f0769c86dd5751`。本次仅跑指定开发 trace，完成六轮优化矩阵，没有运行正式实验，没有 push。最终推荐按拓扑选择已测配置，见 [p11a_selected.json](configs/p11a_selected.json)。

## 方法、范围与可比性

已阅读 RESULTS.md 的 Tool&Agent、Conversation、大模型三部分以及 P9A_NOTES.md/P10B_NOTES.md。历史正式数字只用于定位瓶颈；下面均为本次重新测得的开发结果。

- 8B：`bench/dev_toolagent_1200_240_x2.5.jsonl` 前300条；24B：`bench/dev_big_toolagent_1200_80_x7.5.jsonl` 前150条。标准/2倍/饱和分别为 time-scale 1/0.5/0.05。客户端统一 `--warmup 2 --force-close --drain-timeout 300`。
- 主 SLO：TTFT≤5s 且 TPOT≤100ms；宽松：TTFT≤10s 且 TPOT≤100ms。goodput=主SLO达标请求数/测量总时长；吞吐和goodput分母均包含排空。输入tok/s包含cache命中，不能当作新计算prefill速率。
- 每配置启动一次，顺序为标准→2倍→饱和。每档额外31663-token固定输入/8输出预热，前后flush；DP的router按档重置。RLS在同一进程内继续学习。每项只有一次测量，没有置信区间，亚百分比差异不作统计显著性结论。
- 四张RTX4090独占；0/1在NUMA0、2/3在NUMA1。每次使用前检查GPU和进程，运行期间按PID及创建时间追踪自有进程；未发现容器外占用。KV、完整未来输出预留、mem fraction和原分区配置没有放宽。
- 原始argv/env、源码commit及patch、trace/config哈希、逐请求结果、GPU采样、逐stage lane/请求事件和decode批大小均在 `logs/p11a/`。完整所有速率的TTFT/TPOT p50/p90/p95/p99、输入/输出tok/s、SLO与GPU见 [comparison.md](logs/p11a/comparison.md) / [JSON](logs/p11a/comparison.json)；每stage计时、分区驻留、准入阻塞直方图和decode批大小见 [stage-details.md](logs/p11a/stage-details.md) 及每档 `analysis.json`。DP2的独立worker分别计算计时差分，避免跨worker相减。stage计时取测量窗口内首末summary差分，窗口两端不足10s不计；CUDA与双stream驻留会重叠，不能相加。

基线特例：v4 24B PP4标准档150/150完成后，下一档的31k预热分配256MiB时仅余253.81MiB而OOM，发生在double客户端启动之前。修正编排为预热前也flush，然后用同一冻结4c版本/配置另起进程续跑double、saturated，保留 `baseline-big_pp4` 原失败记录及 `baseline-resume-big_pp4` 续跑记录。因此该基线后两档RLS重置；不能声称三档同一进程。测量档本身均无失败/未完成。

## 有效实现与边界

stage0对完整准入批生成FCFS token块计划，短请求和长请求尾部可打包。每stage一次性分配完整输入KV、保留原radix锁；子块只持有张量视图，后块prefix包含前块KV，每stage FIFO执行。块内继续使用原按层切片和RLS/ALP分区控制。每块有独立wire ID与READY/接收槽生命周期，child metadata由CUDA event覆盖；末级完成后按序归还精确token额度。整父批全部完成后才在原MERGE点插入radix并返回首token。取消标志共享到子块，完成在途计算/通信后释放，避免KV或接收槽提前复用。

中间块跳过norm/head/采样，终结span按FCFS必为行前缀，用basic slice视图采样，避免Python列表GPU索引的同步。接收buffer按块预算分配。24B保留原MLP尾块padding；可选QKV/O投影padding只影响投影dummy行，不进入attention/KV。追加块改用完整上下文paged causal attention，并禁用自动KV split，减少BF16归约路径变化。

可选token在途上限替代批数上限；完整KV/output预留独立保留。单条超额度的FCFS请求仅在未完成工作为0时允许进入，避免饿死。可选refill门槛等待已有工作归还足够额度后再准入，门槛受当前ALP预算限制，不等待未来请求凑批。可选CPU元数据让整批prefill首次split无需GPU→CPU读回，同时用CPU前缀和、query/KV indptr及last-page长度供FlashInfer规划；KV索引仍在GPU构造，整数元数据等价。新增参数默认0/关闭，原pdmux、普通PP、普通Chunked路径保持原行为。

本次采用实测选择的固定4096/8192块预算，没有实现新的动态chunk-size控制器；块内沿用v4层切片，RLS/ALP继续用于原有分区与准入决策。KV容量和decode微批上限保持v4。剖析表明主要问题是长prefill气泡以及小批带来的CPU/READY固定开销。

## 本次开发基线（18档）
''',table('baseline',LABELS+['llama_muxwise_tp2dp2','big_muxwise_tp4'])]
    descriptions={
      'r1':('4096固定CPP，保留v4批数上限','c690b5d0d', '长输入气泡减少，但采样列表索引造成CPU同步，8B吞吐退化；24B冷cache出现新增数值分叉，本轮不推荐。'),
      'r2':('修复采样视图、MLP尾块、paged追加attention、KV split和接收buffer；8192 CPP/投影，保留v4批数准入','563b32fe9', '8B PP4及两种24B拓扑三档主SLO/输出吞吐均不低于v4；8B PP2×TP2主SLO改善，但容量仍略退。性能前数值预检通过无新增差异门槛。'),
      'r3':('8192 CPP，65536 token额度，准入budget8192；8B去投影padding','563b32fe9', '小批过多：两个PP2×TP2高负载吞吐低于v4；8B PP4标准SLO略升但容量低于R2。'),
      'r4':('恢复准入budget16384、token额度65536；8B PP2关闭CPP，24B PP4使用4096 CPP/投影','563b32fe9', 'PP4两组改善；PP2×TP2高负载容量仍退。8B PP2关闭CPP也未解决小批固定成本。'),
      'r5':('R4上增加16384 token回补门槛，cap/KV预留不变','7e1adaecf', '检查额度逐块归还时能否减少零碎准入；与R4的SLO/容量取舍保留在表中。'),
      'r6':('PP整批首split复用CPU长度元数据；按拓扑采用此前有效配置','1dcbb285e', '最后一轮。具体参数及选择理由见最终配置；不再追加调参轮次。'),
    }
    for rnd,(change,commit,conclusion) in descriptions.items():
        lines += [f'\n## {rnd.upper()}：{change}\n\n源码 `{commit}`。{conclusion}\n',table(rnd,LABELS)]
    lines += ['''
## 剖析与未采用的方案

R1将标准窗口末级等上游比例从8B PP4的39.2%降至19.5%、24B PP4的32.3%降至15.7%、24B PP2的17.5%降至9.8%，但末级prefill_launch CPU驻留分别由1.85/6.29/10.20s升至14.08/21.10/17.27s。无拷贝采样修复后R2的8B PP4与24B PP4该驻留降至1.87/3.46s。长请求流水线重叠有效，但固定CPU成本必须一起处理。

R3的8B PP2两倍速率准入批数从R2的73增至135，末级CPU launch 6.72→9.43s，stage0 READY 7.53→13.56s，末级等上游11.38→18.62s；decode平均bs反而9.70→9.28。R4整请求prefill仍有89批，CUDA prefill计算量与v4近似，但CPU/READY开销更大。R5的refill门槛针对这种零碎credit准入；R6进一步去掉整批首次长度、前缀求和以及FlashInfer planner索引指针的读回同步。

24B数值诊断使用独立插桩服务，未混入性能测量：R1 cold20在PP2/PP4均17/20一致。初始R2的MLP尾块修复不足（16/20与17/20）；逐层定位到CPP边界attention的ragged/paged BF16合并。改paged causal后为18/20与17/20。短尾块QKV shape变化由投影padding修复；自动KV split变化由禁用split修复，两条4500/5500输入的完整64步logits及采样层组件恢复完全一致。4096块的PP2在8000输入仍有O投影GEMM归约差异（第0层起约6.1e-5），导致输出26近似并列logits分叉；关闭全局BF16 reduced-precision反而增加差异，未采用。PP2因此采用8192块，≤8192保留整批fastpath。

8192候选的serial20/cold20：8B两组和24B PP2均20/20；24B PP4 serial20/20、cold19/20，所有输出与v4归档完全相同。已知PP4差异是cold请求12输出35起，与本次普通PP4相比1252/1280 token一致；不能宣称所有拓扑逐token完全等价。最终另测12288-token冷请求，覆盖真实跨块执行，避免把最长8000的cold20误当作CPP数学等价证明。数值证据见 `r2e-correctness-gate.json`、`numeric-r2*-*`；失败方案均留档，没有覆盖结果。

## 最终选择、v4对比与MuxWise差距

最终配置按拓扑选取各轮冻结源码的实测点，表中注明轮次。后续新增功能有独立开关，未选择的功能保持关闭。所选配置在最终HEAD统一执行正确性与资源回归；六轮之后不追加性能矩阵。
''', (OUT/'selected-comparison.md').read_text()]
    lines += ['\n'+str(v)+'\n' for v in selection.get('decisions',[])]
    lines += ['\n### 所选配置的标准窗口剖析\n',
              '| 配置 | 末级等上游占比 v4→v5 | 末级CPU prefill_launch占比 v4→v5 | decode平均bs v4→v5 | v5主要准入阻塞（次数/驻留s） |',
              '| --- | --- | --- | --- | --- |']
    for label,rnd in selection['performance_rounds'].items():
        a,b=[row(r,label,'standard') for r in ['baseline',rnd]]
        def tail(v):
            stages=v['stage_windows']['full']['stages']
            return stages[max(stages,key=int)]
        ta,tb=tail(a),tail(b)
        def percentage(stage,lane,key):return 100*stage['wall_seconds'][lane].get(key,0)/stage['seconds']
        upstream='→'.join(f"{percentage(v,'prefill','upstream'):.2f}%" for v in [ta,tb])
        cpu='→'.join(f"{percentage(v,'decode','prefill_launch'):.2f}%" for v in [ta,tb])
        bs='→'.join(f"{v['decode_rounds']['bs']['mean']:.2f}" for v in [a,b])
        adm=b['stage_windows']['full']['admission']
        reasons=sorted([k for k,n in adm['blocked'].items() if n and k != 'empty'],key=lambda k:adm['wall'].get(k,0),reverse=True)[:3]
        text='；'.join(f"{k}: {adm['blocked'][k]}/{adm['wall'].get(k,0):.2f}s" for k in reasons)
        lines.append(f"| {NAMES[label]} | {upstream} | {cpu} | {bs} | {text} |")
    lines += ['\n上述占比按各自stage观测窗口归一化；完整各stage、三种速率及准入直方图仍以stage-details.md/analysis.json为准。']
    lines += ['\n![开发SLO与饱和吞吐](logs/p11a/slo-throughput.png)\n\n可导出图：[PDF](logs/p11a/slo-throughput.pdf)。图仅表示本次单次开发测量。\n']
    dp=[v for v in data if v['run']=='v5-dp2-llama_pp2dp2']
    if dp:
        lines += ['\n### 8B PP2×DP2补充验证\n\n对最终配置仅做一次三档开发验证，无本次v4 PP2×DP2成对基线，不宣称相对v4提升。\n',table('v5-dp2',['llama_pp2dp2'])]
        lines += ['\n这组附测不作为8B默认推荐。饱和档GPU0/1利用率均值为87.0%/99.4%，GPU2/3仅56.9%/61.3%；worker0的KV/完整输出预留阻塞驻留为61.05/34.58s，worker1的empty状态为54.55s，伴随明显worker负载不均。两个worker分别计时，未跨worker相减；验证文件为analyzer-validation.json。该配置在标准档中位TTFT较低，但高负载吞吐落后PP4，未在六轮之外继续调参。']
    lines += ['\n## 最终回归与资源守恒\n',
              '| 配置 | serial20 exact | cold20 exact | 新增分叉 | 冷12288 exact | 取消数 | stress200 完成/失败/未完成 |\n| --- | --- | --- | --- | --- | ---: | --- |']
    for label,v in gate.items():
        s=v['stress'];lines += [f"| {NAMES[label]} | {v['serial20']['exact_plain']}/20 | {v['cold20']['exact_plain']}/20 | {v['serial20']['unexpected_cases'] + v['cold20']['unexpected_cases']} | {v['cold12288']['exact']} | {v['cancellation_requests']} | {s['completed']}/{s['failed']}/{s['unfinished']} |"]
    lines += [f'''
serial/cold每请求比较64个原始token，参照为同拓扑普通实例。取消16个请求后检查abort结果并再生成16-token请求；压力为指定开发trace前200条、time-scale0.02，客户端仍warmup2/force-close/drain300。所有PP/TP rank在排空后检查free KV+可驱逐KV=容量、128个请求槽全部归还、接收槽回收；保留 `final-verify-*/run.json` 和 [正确性门禁](logs/p11a/final-correctness-gate.json)。

原路径共12项冒烟：原pdmux的8B TP1/TP2、24B TP4；普通PP四拓扑；普通Chunked四拓扑；最终8B PP2×DP2。DP2包含router并发与两个worker分别执行12288-token请求。CPU最终相关套件141项全部通过，格式检查通过，见 `logs/p11a-unit-r6-full.log` / `logs/p11a-format-r6.log`。所有配置默认关闭新功能时仍走原路径。

结束审计见 [final-audit.json](logs/p11a/final-audit.json)：复核{audit['protected_files_verified']}个既有trace/结果/笔记哈希，重算每次测量指标，检查每rank资源守恒。服务器日志中的唯一执行异常是上述v4 PP4测量前预热OOM；数值预检中的失败方案已在上文单独记录。未发现外部GPU占用。结束时四卡空闲、无残留服务/监测进程，工作树干净。代码已commit，未push。

## 最终v5配置与完整启动命令

配置文件：[p11a_selected.json](configs/p11a_selected.json)。下面命令从最终配置生成；执行前确认四卡没有容器外占用。同一时刻只启动一组拓扑。带GPU独占检查和退出清理的入口为 `source /root/workspace/muxpp_work/env.sh && python /root/workspace/muxpp_work/scripts/p11a_serve.py launch <label>`；按Ctrl-C停止。以下是可核查的完整展开命令，DP2需先在两个终端启动workers、待ready后启动router。
''']
    lines += ['| 配置 | CPP token | 投影padding | 未完成token额度 | refill token | CPU元数据 | 实测来源 |',
              '| --- | ---: | ---: | --- | ---: | --- | --- |']
    for label,flags in selection['flags'].items():
        def value(key):return int(flags[flags.index(key)+1]) if key in flags else 0
        tokens=value('--pdmux-pp-max-inflight-tokens')
        lines.append(f"| {NAMES[label]} | {value('--pdmux-pp-prefill-chunk-size')} | {value('--pdmux-pp-prefill-projection-pad-to')} | {tokens if tokens else '沿用v4批数门控'} | {value('--pdmux-pp-token-refill-size')} | {'开启' if '--pdmux-pp-cpu-prefill-metadata' in flags else '关闭'} | {selection['performance_rounds'].get(label,'最终固定配置附测')} |")
    for label in ['llama_pp4','llama_pp2tp2','llama_pp2dp2','big_pp4','big_pp2tp2']:
        lines += [f'\n### {NAMES[label]}\n\n```bash\nsource /root/workspace/muxpp_work/env.sh']
        for name,cmd,env,_ in serve.commands(label,ROOT/'configs/p11a_selected.json'):
            lines += [f'# {name}',shlex.join(['env',*[k+'='+v for k,v in env.items()],*cmd])]
        lines += ['```']
    lines += ['\n## 代码与复现入口\n', '```text',subprocess.check_output(['git','log','--reverse','--oneline','4c10135d2..HEAD'],cwd=serve.r.MAIN,text=True).strip(),'```',
              '\n开发复现：`scripts/p11a_runner.py --round <新标签> --source <冻结worktree> --config configs/p11a_selected.json`。正确性/原路径入口分别为 `p11a_final_checks.py`、`p11a_legacy.py`。分析入口为 `p11a_analyze.py`、`p11a_compare.py`、`p11a_selection_report.py`、`p11a_plot.py`、`p11a_audit.py --final`。不调用任何正式trace。']
    destination=ROOT/'P11A_NOTES.md'
    backup=OUT/'P11A-progress-archive.md'
    if not backup.exists():backup.write_text(destination.read_text())
    destination.write_text('\n'.join(lines)+'\n')
    print(destination)


if __name__=='__main__':main()
