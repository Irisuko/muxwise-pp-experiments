#!/usr/bin/env python3
"""Build the P3 report exclusively from preserved formal-run artifacts."""
import ast, collections, datetime, json, pathlib, re, shlex, statistics
ROOT=pathlib.Path('/root/workspace/muxpp_work')
LABELS=['chunked','muxwise','sglang_pd','muxwise_pp']
def read(p):return json.loads(p.read_text())
def fmt(x,k=1):return '—' if x is None else f'{x*k:.3f}'
def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]
def main():
    runs={l:read(ROOT/'results'/f'{l}_summary.json') for l in LABELS if (ROOT/'results'/f'{l}_summary.json').exists()}
    lines=['# P3b 四系统正式实验','',f'生成时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。', '',
    '模型：Llama3.1-8B-Instruct（本地路径见精确命令），bf16，4×RTX4090 24GB。统一代码基：`bd849d628cd6332defee24945ce71db3fa662f90`，分支 `muxwise-pp`，/root/workspace/sglang。', '',
    '正式 trace：`bench/toolagent_600_240_x2.5.jsonl`，1551条，输入11,245,637 tokens、目标输出261,776 tokens，最后到达599.9975秒。原窗口1603条，过滤总长度>32768的52条。到达输入速率约18,743 tok/s、输出需求约436 tok/s。四系统各一次，不补跑。', '',
    '统一 `--disable-overlap-schedule --context-length 33792 --schedule-policy fcfs`；FCFS是该代码默认值。max-running-requests=128。chunked、PD、MuxWise-PP为FlashInfer，graph最大batch参数32；其中PD prefill由既有代码自动关闭CUDA graph。MuxWise为通过31k预热所需的Triton、关闭CUDA graph，并启用expandable_segments。该后端/图捕获差异需纳入结果解释。客户端统一`--warmup 2 --force-close`，请求超时3600秒，无重试。预热与冒烟不计入正式结果。', '',
    '主SLO：TTFT≤5秒且TPOT≤100毫秒。分母为全部1551条计划请求；失败、未发送和TPOT未定义的单token输出计为未达标。延迟和吞吐只统计成功请求；吞吐分母包含10分钟到达与末尾排空，不能将成功子集的延迟直接理解为整体可用性。', '',
    '正式前清空后端cache并重启DP/PD router，清除冒烟的前缀路由记录。GPU/JIT保持预热；PP在线RLS保留预热样本。每次启动前确认无GPU计算进程，每5秒采样GPU，结束清理进程树。', '', '## 汇总比较','']
    lines[lines.index('## 汇总比较'):lines.index('## 汇总比较')]=['统一force-close的依据是[P2_NOTES.md](P2_NOTES.md)“客户端连接复用诊断”：此前客户端复用已超过服务端5秒keepalive的空闲连接，出现请求尚未进入调度器就断开的现象。每请求新建连接用于规避该竞争，不增加重试。', '']
    rows=[]
    for l,s in runs.items():
        rows.append([l,s['completed_requests'],s['failed_requests'],s['unsent_requests'],fmt(s['duration_sec']),fmt(s['throughput']['input_tokens_per_sec']),fmt(s['throughput']['output_tokens_per_sec']),fmt(s['slo']['ttft_le_5s']['rate'],100)+'%',fmt(s['slo']['tpot_le_100ms']['rate'],100)+'%',fmt(s['slo']['ttft_le_5s_and_tpot_le_100ms']['rate'],100)+'%',fmt(s['cached_tokens']['token_weighted_ratio'],100)+'%'])
    lines+=table(['系统','完成','失败','未发送','总时长(s)','输入tok/s','输出tok/s','TTFT≤5s','TPOT≤100ms','联合SLO','响应token加权cache命中'],rows)
    lines+=['','cache来自成功响应`meta_info.cached_tokens`，按输入token加权；缺失不是零。PD字段语义需结合服务端prefill日志核对。吞吐中的输入包含命中部分，不等于实际prefill算力。','','## 延迟分位','']
    rows=[]
    for l,s in runs.items():
        for metric in ['ttft','tpot','e2e']:
            d=s['latencies_sec'][metric];k=1000 if metric=='tpot' else 1
            rows.append([l,metric.upper()+(' (ms)' if k==1000 else ' (s)'),d['count']]+[fmt(d[v],k) for v in ['mean','p50','p90','p95','p99','max']])
    lines+=table(['系统','指标','样本数','mean','p50','p90','p95','p99','max'],rows)
    lines+=['','全部2/5/10s与50/100/200ms阈值及联合口径保存在[summary.csv](results/summary.csv)和各系统`*_summary.json`。','','## 实际配置与命令','',
    'Router通过`/root/venvs/sgl/bin/pip install sglang-router==0.1.9`安装，与代码仓库声明版本一致；[安装记录](logs/p3/setup/install-router.log)。DP使用安装wheel的Rust router/cache_aware；PD使用同仓库mini_lb的PD模式/random策略（没有固定random seed），传输为Mooncake TCP（MC_FORCE_TCP=1）。请求timeout=3600s、并发上限4096。该wheel cache_aware默认参数为cache_threshold=0.5、balance_abs_threshold=32、balance_rel_threshold=1.0001、eviction_interval=60s。', '',
    '所有实例的完整命令、环境和`get_server_info`保存在对应正式目录的`commands.json`、`commands.sh`、`run.json`；服务端启动日志也记录全部解析参数。下列代码块与commands.sh是各进程的精确命令清单；实际并发启动和清理由launch/run_experiment脚本管理，不应逐行以前台方式执行清单。run.json的status=PASS表示生命周期完成，不代表正式请求全部成功；请求失败数和formal_returncode单独保留。']
    lines+=['','PD传输包为mooncake-transfer-engine==0.3.13.post1；prefill侧chunked-prefill-size=4096，decode侧采用默认值，实际解析为2048。未安装或使用NIXL。MuxWise与PP分别使用[pdmux_4090.yml](configs/pdmux_4090.yml)和[pdmux_pp_4090.yml](configs/pdmux_pp_4090.yml)。']
    configs=[]; manifests={}
    for l,s in runs.items():
        m=read(ROOT/'results'/f'{l}.formal-started.json');d=pathlib.Path(m['log_dir']);meta=read(d/'run.json');jobs=read(d/'commands.json');manifests[l]=(d,meta,jobs)
        for job in jobs:
            if job['name']=='router':continue
            cmd=job['argv'];log=(d/(job['name']+'.log')).read_text(errors='replace')
            caps=re.findall(r'max_total_num_tokens=(\d+)',log)
            configs.append([l,job['name'],job['env'].get('CUDA_VISIBLE_DEVICES'),cmd[cmd.index('--mem-fraction-static')+1],','.join(dict.fromkeys(caps)), '关闭（代码强制chunk cache）' if l=='sglang_pd' and job['name'] in ('worker2','worker3') else '开启',job['env'].get('MC_FORCE_TCP',job['env'].get('UCX_TLS','—'))])
    lines+=['']+table(['系统','实例','GPU','mem fraction','max_total_num_tokens','radix','传输环境'],configs)
    for l,(d,meta,jobs) in manifests.items():
        rel=d.relative_to(ROOT)
        lines+=['',f'### {l}','',f'日志：[{rel}]({rel})；启动就绪{fmt(meta.get("startup_s"))}秒。','', '```bash',(d/'commands.sh').read_text().strip(),'```','', '正式客户端：','```bash',shlex.join(read(d/'formal-command.json')),'```']
    lines+=['','## 显存选择、启动修复与预检','', '[完整选择审计](logs/p3/SELECTION.md)。只使用启动、31663-token预热和≤1分钟冒烟选择显存，不使用正式负载调参。MuxWise-PP按P1激活OOM证据固定0.65，P2全部特性保持推荐值。', '',
    'PD decode端在`server_args.py::_handle_disaggregation`中强制关闭radix（使用chunk cache），prefill保留radix，属于既有实现限制。PD正式前20条短请求与普通TP1输出比对，证据见其正式日志目录的`correctness20.json`和`correctness-comparison.json`。', '']
    failed_launches=[]
    for label in LABELS:
        for path in sorted((ROOT/'logs/p3'/label).glob('*/run.json')):
            run=read(path)
            if run.get('status')!='FAIL':continue
            causes=[]
            for log in path.parent.glob('worker*.log'):
                match=re.search(r'torch.OutOfMemoryError: (.+)',log.read_text(errors='replace'))
                if match:causes.append(match.group(1).split('Including non-PyTorch')[0])
            cause=causes[0] if causes else run.get('error','')
            failed_launches.append([label,path.parent.name,'未进入正式' if not (path.parent/'formal-command.json').exists() else '已进入正式',cause[:380].replace('|','/')])
    if failed_launches:
        lines+=table(['系统','启动目录','阶段','失败原因'],failed_launches)+['']
    for l,(d,meta,jobs) in manifests.items():
        smoke=read(d/'smoke'/f'{l}_smoke_summary.json')
        long=read(d/'long-warmup.json')
        lines.append(f'- {l}：正式启动后的长请求预热{len(long)}条通过；冒烟{smoke["completed_requests"]}/{smoke["planned_requests"]}成功、{fmt(smoke["duration_sec"])}秒。')
    lines+=['','## 服务端cache日志口径','']
    cache_rows=[]
    for l,su in runs.items():
        d,_,jobs=manifests[l];new=hit=batches=0
        start=su['run_started_unix'];end=start+su['duration_sec']
        for job in jobs:
            if job['name']=='router':continue
            for line in (d/(job['name']+'.log')).read_text(errors='replace').splitlines():
                if l=='muxwise_pp':
                    if 'PP0]' not in line:continue
                    match=re.search(r'admit batch=\d+ requests=\d+ tokens=(\d+) cached=(\d+)',line)
                else:match=re.search(r'#new-token: (\d+), #cached-token: (\d+)',line)
                if not match:continue
                try:stamp=datetime.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').timestamp()
                except ValueError:continue
                if start<=stamp<=end+1:
                    new+=int(match[1]);hit+=int(match[2]);batches+=1
        cache_rows.append([l,batches,new,hit,fmt(hit/(new+hit) if new+hit else None,100)+'%'])
    lines+=table(['系统','prefill日志批次','新token累计','命中token累计','命中/(新+命中)'],cache_rows)
    lines+=['','仅取正式回放时间窗，PP只取stage0以免四级重复计数。日志按秒记录，边界最多混入少量健康/预热token；retract可能造成重算计数，因此这是服务端计算路径口径，和成功响应口径不强制相等。PD应结合此prefill侧命中率判断缓存，decode的响应字段可能为0。','','## 失败、资源与客户端检查','']
    for l,s in runs.items():
        records=[json.loads(t) for t in (ROOT/'results'/f'{l}_requests.jsonl').read_text().splitlines()]
        errors=collections.Counter(r.get('error') for r in records if not r['success'])
        lines.append(f'- {l}：send_lag p99={fmt(s["latencies_sec"]["send_lag"]["p99"],1000)}ms，最大={fmt(s["latencies_sec"]["send_lag"]["max"],1000)}ms；有cache字段的成功请求{s["cached_tokens"]["available_requests"]}条；平均请求命中比例{fmt(s["cached_tokens"]["mean_ratio"],100)}%。')
        if errors:
            for err,n in errors.most_common(8):lines.append(f'  - {n}条：`{str(err)[:700].replace(chr(10)," ")}`')
        d,meta,_=manifests[l]
        samples=collections.defaultdict(list);formal_samples=collections.defaultdict(list)
        for line in (d/'gpu.csv').read_text().splitlines()[1:]:
            fields=line.split(',')
            if len(fields)>=6:
                try:
                    val=(float(fields[2].split()[0]),float(fields[4].split()[0]))
                    samples[int(fields[1])].append(val)
                    stamp=datetime.datetime.strptime(fields[0].strip(),'%Y/%m/%d %H:%M:%S.%f').timestamp()
                    if s['run_started_unix']<=stamp<=s['run_started_unix']+s['duration_sec']:
                        formal_samples[int(fields[1])].append(val)
                except ValueError:pass
        lines.append('  - 5秒采样最大显存(MiB)：'+', '.join(f'GPU{i}={max(v[0] for v in vs):.0f}' for i,vs in sorted(samples.items()))+'。采样包含启动/预热/正式/清理，不是瞬时峰值。')
        lines.append('  - 正式窗口平均GPU利用率 / 最大显存MiB：'+', '.join(f'GPU{i}={sum(v[1] for v in vs)/len(vs):.1f}% / {max(v[0] for v in vs):.0f}' for i,vs in sorted(formal_samples.items()))+'。')
    if 'muxwise_pp' in manifests:
        d,_,_=manifests['muxwise_pp'];log=(d/'pp4.log').read_text(errors='replace');entries=[]
        s=runs['muxwise_pp'];start=s['run_started_unix'];end=start+s['duration_sec']
        for line in log.splitlines():
            if 'MuxWise-PP summary stages=' not in line:continue
            try:
                stamp=datetime.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').timestamp()
                if not start<=stamp<=end+1:continue
                stages=ast.literal_eval(line.split('stages=',1)[1].split(' mbs=',1)[0]);entries.append((stages,line))
            except (ValueError,SyntaxError):pass
        migration_events=[]
        for line in log.splitlines():
            if 'PP0]' not in line or 'MuxWise-PP migrate step=' not in line:continue
            stamp=datetime.datetime.strptime(line[1:20],'%Y-%m-%d %H:%M:%S').timestamp()
            if start<=stamp<=end+1:
                migration_events.append({'line':line,'rids':ast.literal_eval(line.split(' rids=',1)[1])})
        save={'migration_events':migration_events,'count':len(entries),'last_stages':entries[-1][0] if entries else {},'last_line':entries[-1][1] if entries else ''}
        (d/'formal-scheduler-analysis.json').write_text(json.dumps(save,indent=2))
        lines+=['','## MuxWise-PP 调度观察','',f'正式时间窗口内摘要{len(entries)}条；[机器可读分析]({d.relative_to(ROOT)}/formal-scheduler-analysis.json)。分区快照频数是约10秒采样，不能视为精确驻留时长；RLS MAPE为近128个样本的在线诊断，不是端到端SLO预测误差。']
        lines+=['','正式窗口stage0迁移指令数：'+str(len(migration_events))+'；迁移请求计数（含同一请求重复迁移）：'+str(sum(len(e['rids']) for e in migration_events))+'。', '', '分区编号对应prefill/decode SM数：0=128/0，1=96/32，2=80/48，3=64/64，4=0/128（见configs/pdmux_pp_4090.yml）。']
        if entries:
            rows=[]
            for stage,data in sorted(entries[-1][0].items()):
                counts=collections.Counter(e[0][stage]['partition'] for e in entries if stage in e[0])
                rows.append([stage,dict(sorted(counts.items())),data['samples'],fmt(data['mape'][0],100)+'%',fmt(data['mape'][1],100)+'%',data['by_partition']])
            lines+=['']+table(['stage','分区快照次数','decode/prefill样本','末次decode MAPE','末次prefill MAPE','各分区D/P样本累计'],rows)
            error_rows=[]
            for stage in sorted(entries[-1][0]):
                ds=[e[0][stage]['mape'][0] for e in entries if stage in e[0] and e[0][stage]['mape'][0] is not None]
                ps=[e[0][stage]['mape'][1] for e in entries if stage in e[0] and e[0][stage]['mape'][1] is not None]
                error_rows.append([stage,fmt(statistics.median(ds),100),fmt(max(ds),100),fmt(statistics.median(ps),100),fmt(max(ps),100)])
            lines+=['','各级RLS MAPE快照的中位数/最大值（%，快照自身包含近128样本滚动窗口，不是独立误差样本）：','']+table(['stage','decode 中位','decode 最大','prefill 中位','prefill 最大'],error_rows)
            waiting=[int(re.search(r'waiting=(\d+)/',e[1]).group(1)) for e in entries]
            inf=sum(' required=inf ' in e[1] for e in entries)
            lines+=['',f'摘要中的最大waiting请求数={max(waiting)}；required=inf的摘要{inf}/{len(entries)}。这里inf是过期catch-up时间预算触发的设计分支，不是模型数值崩溃。']
            migrations=[int(re.search(r'migrations=(\d+)',e[1]).group(1)) for e in entries]
            lines+=['',f'正式摘要中的累计迁移请求数范围：{min(migrations)}–{max(migrations)}。累计RLS样本包含启动、预热与冒烟，不能把累计数全部归因于正式回放。']
    notes=ROOT/'logs/p3/OBSERVATIONS.md'
    if notes.exists():lines+=['','## 客观分析与异常','',notes.read_text().strip()]
    lines+=['','## 图与原始数据','', '[汇总Markdown](results/summary.md) · [完整CSV](results/summary.csv) · [TTFT CDF](results/ttft_cdf.png) · [TPOT CDF](results/tpot_cdf.png) · [TTFT时间线](results/ttft_over_time.png) · [全部SLO阈值](results/slo_attainment.png)','',
    '每系统只有一轮正式测量，无置信区间；本次结果仅描述该硬件、代码、trace与配置。不同架构的模型副本数、KV容量、缓存驻留、计算/通信与排队共同影响结果，不能单独归因于某个调度特性。','','## 复现入口','', '```bash\nsource /root/workspace/muxpp_work/env.sh\n# 每条命令含启动→健康→长预热→20条冒烟→清cache→正式→清理\nscripts/run_experiment.sh chunked\nscripts/run_experiment.sh muxwise\nscripts/run_experiment.sh sglang_pd\nscripts/run_experiment.sh muxwise_pp\n```','', '当前正式标记已存在时脚本主动拒绝重跑。独立服务入口是`scripts/launch_<label>.sh`；Ctrl-C清理本次服务。预检入口为`python scripts/p3_runner.py preflight <label> --tag <唯一目录名>`，不会写正式结果。', '']
    audit=ROOT/'logs/p3/final-audit.json'
    if audit.exists():
        lines+=['', '最终只读审计：[final-audit.json](logs/p3/final-audit.json)，逐条核对trace、1551条原始结果、汇总、配置、正式锁与运行顺序，并确认GPU与服务进程已清理。', '']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    summary=ROOT/'results/summary.md'
    note='P3b配置与异常说明：sglang_pd发生Mooncake TCP批量KV传输故障，仅28/1551条成功，其延迟只代表成功子集；MuxWise因长prompt显存约束使用Triton并关闭CUDA graph。完整配置、cache口径与单次测量限制见[RESULTS.md](../RESULTS.md)。'
    if summary.exists() and note not in summary.read_text():
        content=summary.read_text();title,body=content.split('\n',1)
        summary.write_text(title+'\n\n'+note+'\n'+body)
    print(ROOT/'RESULTS.md')
if __name__=='__main__':main()
