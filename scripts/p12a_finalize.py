#!/usr/bin/env python3
"""Finalize P12a only after the selected development matrix and cleanup pass."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/root/workspace/muxpp_work')
OUT=ROOT/'logs/p12a'
CONFIG=ROOT/'configs/p12a_selected.json'
MAIN='ttft_le_5s_and_tpot_le_100ms'
LOOSE='ttft_le_10s_and_tpot_le_100ms'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rate_rows(tag):
    paths=[p for p in (OUT/tag).glob('*/analysis.json') if p.parent.name in ['partial','full']]
    return [read(p) for p in sorted(paths,key=lambda p:['partial','full'].index(p.parent.name))]

def metric_table(tags):
    lines=['| run / 规模 | 完成/失败/未完成 | 主/宽松 SLO % | TTFT p50/p90/p95/p99 s | TPOT p50/p90/p95/p99 ms | 输入/输出 tok/s | GPU均值 % |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for tag in tags:
        for a in rate_rows(tag):
            s=a['summary'];lat=s['latencies_sec'];t=s['throughput']
            def q(k,f):return '/'.join(f'{lat[k][p]*f:.3f}' for p in ['p50','p90','p95','p99'])
            gpu=sum(v['util_mean'] for v in a['gpu'].values())/len(a['gpu'])
            lines.append(f"| {tag} / {a['rate']} | {s['completed']}/{s['failed']}/{s['unfinished']} | {100*s['slo'][MAIN]['rate']:.3f}/{100*s['slo'][LOOSE]['rate']:.3f} | {q('ttft',1)} | {q('tpot',1000)} | {t['input_tokens_per_sec']:.2f}/{t['output_tokens_per_sec']:.2f} | {gpu:.2f} |")
    return '\n'.join(lines)

def decomposition_table(tags):
    lines=['| run | 准入 / stage0队列 / stage0父批span / 下游暴露tail（均值s） | 各stage完整prefill CUDA累计s | 新prefill tokens | stage0/1切片数 |',
           '| --- | --- | --- | ---: | --- |']
    diagnostics={(v['run'],v['rate']):v for v in read(OUT/'diagnostics.json')}
    for tag in tags:
        a=read(OUT/tag/'full/analysis.json');d=diagnostics[tag,'full'];m=d['mean_components_seconds']
        values=[m[k] for k in ['admission_queue','stage0_queue','stage0_parent_prefill_span']]
        values.append(sum(v for k,v in m.items() if k.endswith('_exposed_prefill_tail')))
        cuda='/'.join(f'{v:.2f}' for v in d['prefill_cuda_total_seconds'].values())
        counts='/'.join(str(v['cuda_slices']['n']) for v in a['timeline_diagnostics'].values())
        lines.append(f"| {tag} | {' / '.join(f'{v:.3f}' for v in values)} | {cuda} | {d['new_prefill_tokens']:,} | {counts} |")
    return '\n'.join(lines)

def main():
    selected=read(CONFIG)
    assert selected['flags']==read(ROOT/'configs/p12a_r3.json')['flags']
    final=read(OUT/'final-audit.json')
    assert final['head']==selected['commit'] and not final['git_status']
    assert not final['residual_processes'] and not final['residual_listeners']
    assert not final['remaining_owned_processes'] and not final['foreign_pid_observations']
    assert final['formal_result_files_unchanged'] and final['protected_files_verified']==269
    assert not list(OUT.glob('STOP-*.json'))
    head=subprocess.check_output(['git','-C','/root/workspace/sglang','rev-parse','HEAD'],text=True).strip()
    assert head==selected['commit']
    assert not subprocess.check_output(['git','-C','/root/workspace/sglang','status','--porcelain'],text=True).strip()
    files=set()
    performance=selected['performance_runs']
    verification=selected['verification_runs']
    references={label:'reference-'+label for label in verification}
    legacy=[v['tag'] for v in read(OUT/'legacy-queue.json')]
    for tag in set(performance.values())|set(verification.values())|set(references.values())|set(legacy):
        path=OUT/tag;meta=read(path/'run.json');audit=read(path/'audit.json')
        assert meta['status']=='PASS' and meta['commit']==head,(tag,meta.get('status'))
        assert not audit['errors'] and audit['all_conserved'] is not False,tag
        files.update([path/'run.json',path/'commands.json',path/'audit.json'])
    for label,tag in performance.items():
        expected=['partial','full'] if label.endswith('pp2tp2') else ['full'] if label=='llama_pp4' else ['partial']
        for rate in expected:
            s=read(OUT/tag/rate/(rate+'_summary.json'))
            n=(543 if label.startswith('big') else 1620) if rate=='full' else (150 if label.startswith('big') else 300)
            assert (s['completed'],s['failed'],s['unfinished'])==(n,0,0),(tag,rate)
            analysis=read(OUT/tag/rate/'analysis.json')
            assert analysis['parent_matched']==n and not analysis['decomposition_errors'],(tag,rate)
            files.update([OUT/tag/rate/(rate+'_summary.json'),OUT/tag/rate/(rate+'_requests.jsonl')])
    regression=['| 配置 | serial20 | cold20 | cold12288 | 取消 | stress200 | rank资源守恒 |',
                '| --- | --- | --- | --- | --- | --- | --- |']
    for label,tag in verification.items():
        path=OUT/tag;gate=read(path/'correctness-gate.json');long=read(path/'cold12288-comparison.json')
        assert not any(v['unexpected_cases'] for v in gate.values()),tag
        assert long['exact'],tag
        s=read(path/'stress/stress_summary.json');assert (s['completed'],s['failed'],s['unfinished'])==(200,0,0)
        cancels=read(path/'cancellation.json');assert len(cancels)==16
        assert read(path/'audit.json')['all_conserved'] is True
        regression.append(f"| {label} | {gate['serial20']['exact_plain']}/20 | {gate['cold20']['exact_plain']}/20 | 64/64 | 16 abort + 恢复 | 200/0/0 | PASS |")
        for name in ['correctness-gate.json','serial20.json','cold20.json','serial20-comparison.json','cold20-comparison.json','cold12288.json','cold12288-comparison.json','cancellation.json','after-cancel.json','stress/stress_summary.json','stress/stress_requests.jsonl']:
            files.add(path/name)
        for name in ['prompts.json','serial20.json','cold20.json','cold12288.json']:
            files.add(OUT/references[label]/name)
    for tag in legacy:
        smoke=read(OUT/tag/'smoke.json');assert len(smoke)==7
        assert all(v['response']['meta_info']['completion_tokens']==32 for v in smoke)
        files.add(OUT/tag/'smoke.json')
    repeats=selected['control_repeat_runs']
    for tag in repeats:
        path=OUT/tag;meta=read(path/'run.json');audit=read(path/'audit.json')
        assert meta['status']=='PASS' and audit['all_conserved'] and not audit['errors'],tag
        s=read(path/'partial/partial_summary.json')
        assert (s['completed'],s['failed'],s['unfinished'])==(300,0,0)
        files.update([path/'run.json',path/'commands.json',path/'audit.json',path/'partial/partial_summary.json',path/'partial/partial_requests.jsonl'])
    for name,count in [('r1-mlp-unit.log',37),('r1-pp-unit.log',111)]:
        p=OUT/name;assert f'{count} passed' in p.read_text();files.add(p)
    selected.update(validated=True,validated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    reference_runs=references,legacy_runs=legacy,
                    validation_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    tables8=metric_table(['ab-v4-llama_pp2tp2','ab-v4-full-llama_pp2tp2','ab-v5-llama_pp2tp2','r1-llama_pp2tp2','r3-llama_pp2tp2','r4-llama_pp2tp2'])
    tables24=metric_table(['ab-v4-big_pp2tp2','ab-v5-big_pp2tp2','r3-big_pp2tp2','r4-big_pp2tp2'])
    controls=metric_table(['ab-v5-llama_pp4','r4-llama_pp4','ab-v5-big_pp4','r4-big_pp4','ab-v5-llama_pp2dp2','r4-llama_pp2dp2',*repeats])
    dp={tag:read(OUT/tag/'partial/partial_summary.json')['slo'][MAIN]['rate']*100
        for tag in ['ab-v5-llama_pp2dp2','r4-llama_pp2dp2',*repeats]}
    dp_baseline=(dp['ab-v5-llama_pp2dp2']+dp[repeats[0]])/2
    dp_selected=(dp['r4-llama_pp2dp2']+dp[repeats[1]])/2
    assert [v['run'] for v in read(OUT/'dp2-routing-review.json')]==['ab-v5-llama_pp2dp2','r4-llama_pp2dp2',*repeats]
    decomp=decomposition_table(['ab-v4-full-llama_pp2tp2','ab-v5-llama_pp2tp2','r1-llama_pp2tp2','r3-llama_pp2tp2','r4-llama_pp2tp2','ab-v4-big_pp2tp2','ab-v5-big_pp2tp2','r3-big_pp2tp2','r4-big_pp2tp2'])
    body=f'''# P12a：PP2×TP2 v5退步定位与v5.1开发验证

代码提交：`{head}`，未push。共四轮候选，最终8B/24B均选择R3；没有运行正式trace。所选配置、启动入口和验证文件哈希见[配置](configs/p12a_selected.json)，全部原始证据位于[logs/p12a](logs/p12a)。

## 结论与边界

本机可复现v5的8B短集改善、全量退步。主导因素是当前SHM TP通信瓶颈叠加v5对CPP尾块无效padding行的归约；全量负载积压放大了这部分工作。裁剪通信的单因素消融已恢复到接近同机v4，8B再去除不必要投影padding后，全量主SLO由22.346%升至36.914%，TTFT p50由11.413s降到7.163s。短集主SLO78.667%→75.333%，有3.333个百分点取舍；不能宣称所有指标全面改善。

24B保留数值必需的padded GEMM，只归约有效行；全量主SLO19.705%→20.810%，TTFT p50 10.825→9.644s。主SLO仅多6个达标请求，单次开发测量不足以证明明显收益。当前机器的v4也明显弱于历史正式成绩，缺少历史PCIe链路实测，不能把历史全部差额归为v5代码缺陷。

建议先正式确认8B PP2×TP2的Tool&Agent及Conversation两个新label；不承诺恢复历史70.1%/65.9%。本轮没有回放Conversation开发trace，其收益不能由Tool&Agent直接外推。24B主SLO证据较弱，暂不建议仅为性能结论正式重跑。PP4/PP2×DP2无需因本次改动重跑正式。本任务只准备入口，未执行这些命令。

## 方法与硬件

只回放`bench/dev_toolagent_1200_240_x2.5.jsonl`（前300/全1620）和`bench/dev_big_toolagent_1200_80_x7.5.jsonl`（前150/全543）。所有性能档time-scale=1、warmup=2、force-close、drain-timeout=300。每档flush、31663-token随机长预热再flush；吞吐包含排空。GPU利用率按5s采样平均，所有GPU所有权按1s监测。主SLO=TTFT≤5s且TPOT≤100ms；宽松=10s/100ms。PP2×TP2每个版本/规模只测一次，PP2×DP2另做一次同参数配对复核；不作统计显著性承诺。

8B v4冻结worktree `/root/workspace/sglang-p12a-v4`=`f6833f24e`；24B v4采用P11a安全基线`4c10135d2`（大模型显存/数值修复），不是将8B旧提交直接用于24B。v5冻结`/root/workspace/sglang-p12a-v5`=`1dcbb285e`；修复冻结`/root/workspace/sglang-p12a-r1`=`638aa7941`。各性能run保存完整命令、源码commit、空source.patch、请求事件、准入/lane/CUDA计时、NCCL日志和GPU监测。R1/R3/R4代码相同，参数差异单独留档。

| GPU | UUID | 当前负载下PCIe | NUMA |
| --- | --- | --- | --- |
| 0 | GPU-3fe63c80-9f4f-2642-6ba5-bb3da949e9ca | Gen4×16 | 0 |
| 1 | GPU-ffc7795e-ed42-cc9b-ca45-067b5612bff9 | Gen4×16 | 0 |
| 2 | GPU-52777e54-80e7-9357-a595-86ce2678536f | Gen4×16 | 0 |
| 3 | GPU-65206bc3-a377-fc17-36e2-eb60af2a7cd3 | Gen4×8（max报告×16） | 0 |

TP组为[0,1]/[2,3]。v4/v5均关闭custom allreduce，NCCL 2.27.3均走SHM/direct/direct，没有观察到两版之间transport切换，也没有P2P。线程affinity和抽查共享内存页均在NUMA0。硬件记录见[hardware](logs/p12a/hardware)。

真实TP communicator、BF16 4096宽、112SM流的独立探针：8192行两组4.465/8.604ms，16384行8.893/17.105ms；128/64SM也保留接近2倍差距，与GPU3×8一致。Gloo MIN int32中位67–71µs，p95约166–169µs；这是隔离探针，不当作负载下同步耗时。copy engine探针大张量快约20–23%，但模型初始化失败，未采用。变量默认与实现参见[NCCL对应版本源码](https://raw.githubusercontent.com/NVIDIA/nccl/v2.27.3-1/src/transport/shm.cc)。

## 8B PP2×TP2：全部轮次

{tables8}

## 24B PP2×TP2：全部轮次

{tables24}

R2在服务就绪前停滞并终止，未发送trace请求；TTFT/TPOT/SLO/吞吐均为N/A，不用零值替代，失败候选单独留档。

## 根因证据与修复

8B v4/v5全量新prefill token为6,736,772/6,734,212，近似相同。v5的481个父批中383个分块，额外O投影padding为1,426,478行，相当于有效行21.18%，每stage每rank额外allreduce接口payload约174.13GiB（16层×4096宽×BF16；不是测得的物理总线双向字节）。24B v5为457,207额外行/87.21GiB。

R1仅改变O投影padding且TP>1的分支：GEMM仍按原padding形状执行，skip原内部allreduce，取连续有效行视图，再执行一次原TP归约。有效行数来自同一chunk计划，collective次数/顺序不变。TP1、普通decode/PP/Chunked和原pdmux路径保持原算子行为；FCFS、KV/未来输出预留、stage0唯一决策、TP一致性、READY和接收槽协议未改。

R1单因素消融：末级完整prefill CUDA536.67→506.41s，平均准入8.929→5.774s；约610s周期汇总窗口中，inflight阻塞489.90→440.59s、stage0 READY118.11→99.75s。只去除无效通信便同步降低设备时间和排队，说明inflight3阻塞主要是下游容量不足的表现。原PP2×TP2两模型都未启用token credit/refill，不能把退步归于这两个未开启的功能。

原始v4→v5末级CPU prefill_launch约45.99→47.45s、control58.06→56.11s；不支持块级CPU共识为原始退步的主要增量。短档CUDA切片329→304，数量也非必然增加。生命周期/READY次数确实增加，但单次READY、接收槽等待未变慢，更多总等待与下游设备服务时间增长一致。

短集与全量的输出到达负载约410.6→449.3 tok/s，缓存命中约48.6%→42.8%，缺失prefill计算速率约+8.5%；120s后与420s后的密集区间触发持续积压。见[完整CDF与到达窗口图](logs/p12a/dev-ttft.png)及[PDF](logs/p12a/dev-ttft.pdf)。不能把前300条结果外推为全量稳定性。

最终R3：8B关闭投影padding（数值预检通过），保留MLP4096 padding；24B保留8192投影padding和CPU元数据，仅归约真实行。两模型CPP8192、原inflight3、等层数、direct SHM，静态显存比例和KV安全门槛均不变。

### 分解口径及完整全量数据

逐请求采用父批完成门槛：stage0父批span + 下游依次暴露的queue/tail，重叠部分不重复相加；分项总和严格还原客户端TTFT。所有性能请求均匹配。CUDA累计来自完整请求事件，包含设备通信/等待；lane周期汇总窗口另存，不能将并行lane和CUDA直接相加。短档及每个分位点的分解、READY/接收槽分布详见[diagnostics](logs/p12a/diagnostics.json)和各档`analysis.json`/`ttft-requests.jsonl`。

{decomp}

## 四轮选择记录

1. R1：仅裁剪TP无效行。8B全量恢复，但短集主SLO70.333%低于v5；完成两档后正常清理，在END边界停止后续R1队列。
2. R2：R1加`NCCL_SHM_USE_CUDA_MEMCPY=1`。日志确认CE/direct；末级CUDA graph初始化超过4分钟无进展，GPU2/3与对应自有进程100%，stage0已完成。没有READY、没有trace请求。主动终止并正常清理，保留FAIL、日志、startup-stall和termination-reason。py-spy因系统ptrace权限被拒绝，未获取栈，因此不宣称已经证明具体死锁环。纯通信探针通过不能代替模型启动验证。
3. R3：8B取消投影padding，24B维持padding并裁剪归约。8B先完成普通参照、serial/cold20、12288冷请求、取消/压力再测性能。两模型前缀与全量完成后，进入最后一轮。最终两模型均选R3。
4. R4：现有层分配变量试验8B18/14、24B22/18，保持全部安全约束。两组均不选。8B设备CUDA435.87/433.00s已均衡，但stage0切片1882→2586、末级upstream35.02→100.89s，准入5.312→5.922s；新token只+1.03%，不能全归于KV缩小。KV容量390804→335549。24B容量203279→160130，KV/输出预留阻塞约1.04→68.54s；全量主SLO只多8条，TTFT和宽松明显更差。未为性能放松KV预留。第四轮后没有继续调参。

两次轮次切换仅停止后续队列，未中断正在测量的trace；身份及原因保存在`r1-handoff*.json`、`r3-handoff*.json`。原计划也保留，未覆盖结果。

## PP4/PP2×DP2不退化检查

这些配置的TP1算子及参数保持v5；`r4-`目录只是执行轮次标签，不表示对TP1采用层数调整。

{controls}

DP2首次主SLO95.000→91.667%，伴随热点从worker0（192条）切换到worker1（201条）；后者使用含×8卡的组。这是相关观测，不能据此单独证明波动原因。为确认而补做同参数成对复核，主SLO为{dp[repeats[0]]:.3f}→{dp[repeats[1]]:.3f}%；两次均值{dp_baseline:.3f}→{dp_selected:.3f}%。复测未复现首次下降，未发现稳定退化；两次均值仍低{dp_baseline-dp_selected:.3f}个百分点，不能宣称严格统计等效。全部[路由、缓存及分副本达标分布](logs/p12a/dp2-routing-review.json)保留，复核不改变参数、不算新增优化轮，也不选取最好的一次替代原结果。TP1路径的算子行为未改，样本仍不足以证明任意路由分配下严格零波动。

## 正确性、取消、压力与旧路径

使用当前机器、同拓扑普通实例重新生成参照，不把历史8B/24B其他拓扑当作golden。所选8B/24B TP2验证目录为`verify-r3-*`，PP4为`verify-r4-*`。

{chr(10).join(regression)}

24B PP4复现已知cold20请求12、输出35起的差异，与普通PP合计1252/1280 token相同；该请求的完整输出向量与P11a/v4归档完全相同，新增差异列表为空。不能宣称该拓扑所有请求与普通PP逐token全等。12288冷请求跨越实际CPP预算，避免把最长8000的cold20当作跨块证明。

全部取消均验证16条abort并恢复一次16-token生成；压力为开发trace前200、time-scale0.02、warmup2、force-close、drain300。各rank最后空闲检查验证free_kv+evictable=capacity、free_reqs=128、stage0接收槽0/其他stage2；无OOM、scheduler异常或illegal access。

普通PP由上述四个参照实例覆盖；原pdmux的8B TP1/TP2及24B TP4共3项，Chunked的两模型PP2×TP2/PP4及24B TP4共5项，均7条并发请求×32输出PASS。CPU相关测试37+111=148项通过，新增测试检查padded GEMM形状、只归约有效行且恰好一次、reduce_results=False与decode保持原行为。真实GPU BF16前1/32/511/2048/4096/8191行的裁剪/整块归约结果在四rank均exact。

## v5.1启动与run_experiment接入

最终配置显式固定R3实际默认的`NCCL_SHM_USE_CUDA_MEMCPY=0`及16/16、20/20层，避免继承R2/R4调试变量；这些值与R3未设置变量的语义一致，不是新一轮参数。配置绑定commit和验收文件SHA256。启动服务（不回放trace）：

```bash
cd /root/workspace/muxpp_work
source env.sh
python scripts/p12a_serve.py launch llama_pp2tp2
# 或24B（先结束上一个服务）
python scripts/p12a_serve.py launch big_pp2tp2
```

完整展开命令：[8B](logs/p12a/v51-8b-command.sh)、[24B](logs/p12a/v51-24b-command.sh)、[Conversation服务](logs/p12a/v51-conversation-command.sh)。`commands`模式只打印命令。CLI `launch`负责GPU占用检查、生命周期和退出清理。

以下为未来显式正式运行入口，本任务没有执行：

```bash
bash scripts/run_experiment.sh muxpp_v51_pp2tp2
bash scripts/run_experiment.sh conv_muxpp_v51_pp2tp2
bash scripts/run_experiment.sh big_muxpp_v51_pp2tp2
```

对应Tool&Agent/Conversation/大模型三个独立新label。入口要求validated、干净且匹配的commit、验证哈希、GPU独占和未使用label；正式客户端仍warmup2/force-close/drain300。旧准备锁仅在未来显式正式调用内临时移走并finally恢复。PP4/DP2行为未改，不新增相应正式label。

## 中断、审计与清理

v4前300完成后的flush触发一次旧监控未归属显存报警229MiB：设备/进程显存非原子读取恰逢本任务cudaFree。所有PID均为自有，清理后35s四卡约1MiB/0%，没有外部PID。原报警与复核完整保留。新监控在进程读取前后夹取设备显存，按两次设备读数较小者计算释放时未归属下界；阈值未提高，外部PID/利用率仍立即停止。这不是已确认的外部占用。v4全量因此另启进程，RLS未继承短档；v5与候选短/全量则同进程，这项方法差异保留。

首次通信探针误把PP作为第二位置参数（实际是EP），在world size校验处退出，未进入计时；修正为显式关键字后成功，失败日志保留。全部四轮候选及不利结果均保留。

最终审计：[final-audit.json](logs/p12a/final-audit.json)。核对269个受保护文件哈希，正式results的262个文件集合与内容均未变化；所有实际replay命令都在两个开发trace白名单中。无确认的容器外GPU占用，外部PID观测为0；四卡最终35s空闲检查通过，服务端口、服务/客户端、所有记录的自有进程均已清理。源码工作树干净，提交未push。大模型准备记录、GPU原始数据和完整指标可在日志目录复核。
'''
    backup=OUT/'progress-notes.md'
    if not backup.exists():backup.write_text((ROOT/'P12A_NOTES.md').read_text())
    (ROOT/'P12A_NOTES.md').write_text(body)
    CONFIG.write_text(json.dumps(selected,indent=2)+'\n')
    delivery=[ROOT/'P12A_NOTES.md',CONFIG,*sorted((ROOT/'scripts').glob('p12a_*.py')),ROOT/'scripts/run_experiment.sh',OUT/'v51-code.patch']
    (OUT/'delivery-manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in delivery},indent=2)+'\n')
    print(json.dumps(dict(version=selected['version'],commit=head,validated=selected['validated'],bound_validation_files=len(files),notes=str(ROOT/'P12A_NOTES.md')),indent=2))

if __name__=='__main__':main()
