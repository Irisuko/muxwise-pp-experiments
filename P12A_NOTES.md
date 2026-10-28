# P12a：PP2×TP2 v5 退步排查（开发中）

本任务仅使用 `bench/dev_toolagent_1200_240_x2.5.jsonl`（1620条）和 `bench/dev_big_toolagent_1200_80_x7.5.jsonl`（543条）；不运行正式trace。起点 `1dcbb285e`。当前尚未完成A/B和修复验证，本文件是进行中的记录，不是验收报告。

## 实验方法与当前状态

8B v4固定worktree `/root/workspace/sglang-p12a-v4`，提交`f6833f24e`；v5固定worktree `/root/workspace/sglang-p12a-v5`，提交`1dcbb285e`。v5采用P11A选定参数。PP2×TP2的8B/24B均未开启token额度/refill，仍为inflight3。每档独立flush、31663-token/8输出长预热再flush，客户端warmup2、force-close、drain300、time-scale1。前300/150条与全量分别保存，不将历史结果当本机基线。源码、命令、请求事件、NCCL、lane summary、GPU监控均在`logs/p12a/`。

主SLO=5s/100ms；宽松=10s/100ms。吞吐包含排空。逐请求分解采用父批最后完成门槛，CPP各stage重叠部分不重复相加：stage0完整父批span + 下游依次暴露的queue/tail。独立CUDA计时及READY/接收槽分布另列。分析脚本`p12a_analyze.py`不生成请求。

## 硬件事实（非根因结论）

GPU UUID与P11b一致，全部NUMA0。PCIe当前为GPU0/1/2 Gen4×16、GPU3 Gen4×8（最大宽度报告×16）。v4 NCCL 2.27.3日志的TP/PP数据通道为`SHM/direct/direct`，没有P2P。NCCL线程affinity为NUMA0，抽查共享内存映射页在N0。原始记录见`logs/p12a/hardware/`及各run NCCL日志。

已完成的v4 8B前300：300完成/0失败/0未完成；主/宽松SLO62.667%/95%；TTFT p50/p90/p95/p99=4.470/8.565/9.342/10.943s；TPOT=49.594/68.309/76.805/103.920ms；输出344.853tok/s；GPU均值80.75%。300/300请求分解匹配，平均准入1.509s、stage0队列0.592s、stage0 prefill0.706s、stage1暴露队列0.608s、stage1prefill0.936s。当前证据不足以定位v5根因。

## 监控中断及复核

v4前300测量完成后，01:12:32 UTC的flush_cache阶段，旧监控先读设备显存再读进程显存，GPU1差额229.0625MiB触发自动停止。记录仅含自有PID31942，全部计算PID和利用率PID均为本任务；日志同秒四rank释放缓存。服务清理后35次每秒采样均四卡约1MiB/0%，没有外部PID。原报警保留为`alarm-memory-sampling-race.json`，复核为`alarm-review.json`。

P12a监控改为在进程显存读取前后采样设备显存，记录两次读数和是否变化，按两次设备值较小者计算释放期间的未归属显存下界；外部PID/利用率仍立即停止，阈值没有提高。未将此报警宣称为已确认外部占用。完成的300条不重跑，v4全量以新目录、新服务续跑，因此v4两档RLS不是同一进程连续学习；该差异必须保留。所有STOP事件均保留原始证据。

## 待完成

v4/v5全量与PP4对照、根因消融、最多四轮修复、24B及其余拓扑回归、正确性/取消/压力/原路径、代码提交与v5.1接入。当前没有v5.1结果，也没有正式重跑建议。

## 已完成8B PP2×TP2 A/B

| 版本/规模 | 完成/失败/未完成 | 主/宽松SLO | TTFT p50/p90/p95/p99(s) | TPOT p50/p90/p95/p99(ms) | 输出tok/s | GPU均值 |
| --- | --- | --- | --- | --- | ---: | ---: |
| v4 前300 | 300/0/0 | 62.667/95.000% | 4.470/8.565/9.342/10.943 | 49.594/68.309/76.805/103.920 | 344.85 | 80.75% |
| v4 全量 | 1620/0/0 | 31.728/64.074% | 7.696/17.006/18.159/21.418 | 50.281/68.477/77.452/108.109 | 438.28 | 91.18% |
| v5 前300 | 300/0/0 | 78.667/98.333% | 3.786/7.405/8.940/10.746 | 49.377/74.290/75.586/87.846 | 345.77 | 85.64% |
| v5 全量 | 1620/0/0 | 22.346/43.333% | 11.413/22.761/25.495/29.135 | 50.159/68.982/77.456/94.191 | 433.02 | 93.42% |

全量逐请求匹配均1620/1620。平均准入等待v4→v5为5.376→8.929s；stage0队列0.491→1.406s。对齐约610s summary窗口，stage0/1 prefill CUDA累计376.20/495.51→396.31/534.47s；末级CPU prefill_launch45.99→47.45s，control58.06→56.11s；stage0 READY94.80→118.11s；末级upstream51.23→26.21s。并行lane/CUDA时间不可直接相加。

两版新prefill token总量6,736,772→6,734,212，近似相同。v5的481父批中383批分块，O投影padding额外1,426,478行（相当于有效token行21.18%），每stage每rank送入allreduce的额外payload约174.13GiB（16层×4096宽×BF16；这是接口payload，不是物理总线双向字节）。短测量额外201,206行/24.56GiB。v5 CUDA层片数并非必然更多：短测量329→304；终结/READY生命周期次数82→173。

当前有数据支持：短trace掩盖稳态/突发区段积压；current hardware上v4也较历史明显变差；v5增加prefill GPU/通信工作、进一步挤压慢stage。尚待独立通信探针和修复回放验证因果。

## R1候选（性能尚未验证）

本地提交`638aa7941`：只在CPP投影padding且TP>1分支，保留原padded QKV/O GEMM形状，O投影先skip_all_reduce，取有效行视图，再执行原TP归约。无新增collective，所有rank按原统一chunk计划取相同行数。TP1分支、非CPP、FCFS、KV/未来输出预留、stage0唯一决策、READY/控制重放均保持原行为。冻结worktree `/root/workspace/sglang-p12a-r1`。

CPU验证：MLP/投影37项、PP/TP/准入/校准/Chunked预算111项通过；Black与diff检查通过。新增测试验证GEMM仍用padded行、TP恰好归约一次且只传真实行、reduce_results=False/普通decode不改变。GPU正确性与性能尚待执行，不将CPU结果当作GPU通过。

## 通信探针

纯通信、无模型、无trace，使用真实TP[0,1]/[2,3]与独立prefill communicator，关闭custom allreduce；BF16，full/112SM/64SM流，各形状3次预热、5组×5次计时，按组内较慢rank的中位值报告。初次脚本误将PP作为第二位置参数（实际是EP），在初始化world size校验处退出，未进入计时；保留`diag-shm-direct`。显式关键字修复后为`diag-shm-direct-v2`，没有隐藏失败。

| transport/SM | rows | TP0+1 ms | TP2+3 ms |
| --- | ---: | ---: | ---: |
| SHM direct/112 | 512 | 0.299 | 0.562 |
| SHM direct/112 | 4096 | 2.238 | 4.329 |
| SHM direct/112 | 8192 | 4.465 | 8.604 |
| SHM direct/112 | 16384 | 8.893 | 17.105 |
| SHM CE/direct/112 | 8192 | 3.597 | 6.641 |
| SHM CE/direct/112 | 16384 | 7.543 | 13.250 |

两TP组在大张量上接近2倍差距，改变SM分区不消除差距；与GPU3×8而其余×16的链路事实一致。Gloo MIN int32同步中位约67–71µs、p95约166–169µs。不能拿独立探针代替负载下每次同步的实测，但结合服务lane CPU时间仅小幅增加，额外块级CPU共识不符合当前主导退步量级。

另测NCCL 2.27.3已有`NCCL_SHM_USE_CUDA_MEMCPY=1`，日志确认CE/direct；大payload变快，但小32行由约0.05ms上升到约0.13ms，1行由约0.023ms升到约0.045ms。因此没有直接设成默认，先测试只裁剪无效归约行的最小修复。该变量及默认发送端copy语义来自[NVIDIA对应版本源码](https://raw.githubusercontent.com/NVIDIA/nccl/v2.27.3-1/src/transport/shm.cc)。

两种transport下，四rank的随机BF16张量先完整8192行归约，再分别归约前1/32/511/2048/4096/8191行，全部有效输出exact。该测试验证真实TP2裁剪通信结果，不替代模型生成正确性。记录在各`rank*-crop-equivalence.json`。

PP4 v5开发全量：1620/0/0，主/宽松84.259%/99.259%，TTFT2.961/5.517/6.314/8.126s，TPOT48.947/64.058/72.244/82.615ms，输出437.51tok/s，GPU93.81%。

## 24B同机补充基线（已完成）

24B v4使用P11a的`4c10135d2`安全基线（包含大模型显存与数值修复），而用户指定的8B v4严格使用`f6833f24e`。不将这两个提交混称同一个binary。24B v4前150主/宽松24.0/53.333%，全量543主/宽松11.602/32.228%，TTFT p50 11.612s；v5前150为28.667/60.667%，全量19.705/45.488%，TTFT p50 10.825s。均无失败或未完成。完整四分位、吞吐和GPU均值见`logs/p12a/comparison.md`。

24B全量末级prefill CUDA419.48→433.34s，CPU launch33.99→32.79s，control42.74→45.21s。GPU负载中再次读到GPU3 Gen4×8，其余Gen4×16，记录`hardware/loaded-big-v5-pcie.csv`。注意CUDA span包含设备上的通信及资源等待，不能当成纯GEMM耗时。

已准备v5.1三个新label的入口、启动命令渲染和未来正式单次运行保护；仅做静态/help/commands检查，没有调用experiment执行。最终selected配置必须在候选通过GPU验证后生成，当前不代表修复已验收。

## R1开发验证：短集未达验收（全量尚在进行）

8B PP2×TP2前300：300/0/0，主/宽松70.333/97.667%，TTFT3.729/7.168/8.679/10.525s，TPOT49.814/74.441/81.164/95.151ms。相对v5短集，p50/p90略降但主SLO78.667→70.333%，不能只凭中位数或设备时间宣布成功。末级prefill CUDA84.346→80.518s，stage0为62.824→60.735s，证明省去了工作，但本次短集5秒内TTFT请求数236→213（其中2条TPOT超100ms），分布在门槛附近发生变化。将继续看全量，并保留该不利结果。

R1的8B全量保留完整测量；结束清理并打印END后再停止后续R1队列，避免为未通过短集SLO的候选继续执行全部回归。交接身份/原因保存在`r1-handoff-identities.json`、`r1-handoff.json`。这是主动的轮次切换，不是GPU异常或请求失败。R2保持相同代码和所有调度参数，只对两个PP2×TP2配置增加`NCCL_SHM_USE_CUDA_MEMCPY=1`；PP4/PP2×DP2不加该环境变量。R2重新跑完整验收矩阵；参考实例与legacy路径同样不加该变量。当前R2尚未产生性能结论。

R1 8B全量最终结果：1620/0/0，主/宽松34.630/61.111%，TTFT7.688/18.021/19.551/22.292s，TPOT49.837/69.038/76.486/91.754ms。v5→R1末级prefill CUDA534.47→506.41s（-28.06s），平均准入8.929→5.774s，inflight阻塞489.90→440.59s，stage0 READY118.11→99.75s。只裁剪无效通信便同时降低设备服务时间、反压和准入等待，形成因果消融证据；没有调整额度或一致性协议。R1全量恢复到接近同机v4，但短集主SLO未通过，故不选为最终版本。

## R2失败候选与R3选择

R2只增加SHM copy engine；日志确认`SHM/CE/direct`。02:48:29 UTC末级开始CUDA graph初始化后不再前进，stage0在02:48:34 UTC完成全部8个分区（服务日志显示UTC+8的10:48:29/34）。四分钟后stage0卡在futex、GPU0/1为0%，stage1两个进程及GPU2/3为100%。没有READY、没有发送任何trace请求，主动终止runner，常规close清理全部进程与显存。`run.json`为FAIL，完整日志、`startup-stall.json`和`termination-reason.json`保留。py-spy读取被系统ptrace权限拒绝，未获取Python栈；因此只确认初始化停滞，不声称已经证明具体CUDA/NCCL死锁环。独立通信探针通过不能替代模型启动/graph验证，该环境变量不进入最终版本。

R3回到原direct SHM：8B将投影padding设为0（P11a R3/R5已有8B去padding配置），24B仍保留8192投影padding及R1裁剪。所有CPP尺寸、FCFS、KV预留、inflight3、TP控制协议不变。先重新生成8B普通PP2×TP2参照，执行serial20/cold20/12288跨块冷请求与取消/压力，只有数值预检通过才开始R3性能矩阵。R3尚无结论；最多四轮约束继续适用。

## R3 8B结果与R4最后一轮预设

R3数值/资源预检：serial20=20/20、cold20=20/20、12288冷请求64/64；取消16条全部abort，恢复16输出；200条time-scale0.02压力200/0/0，各rank free_kv+evictable=390804、free_reqs=128、接收槽stage0=0/stage1=2。

R3前300主/宽松75.333/99.000%，TTFT3.857/6.675/7.948/9.547s，TPOT48.806/71.275/78.324/87.528ms；全量主/宽松36.914/64.506%，TTFT7.163/17.506/19.176/22.126s，TPOT50.022/68.525/75.425/86.826ms，输出438.62tok/s。两档均零失败/零未完成。相对v5改善全量，但短集主SLO仍低3.333个百分点，未据此宣称全面恢复。

全量stage0/1 prefill CUDA378.70/500.35s，按层数近似线性估计，18/14层可将两级变为约426/438s。24B基线355/433s对应22/18层约391/390s。R4因此使用已有`SGLANG_PP_LAYER_PARTITION`，只改变两种PP2×TP2；PP4/DP2不改。这只是待验证估算，额外层会压缩快级KV容量，保持原静态显存比例与KV/output预留，不通过放松安全门槛抵消。R4后不再调参；普通参照实例采用相同层分配。R3两模型TP2完整结束再交接，交接记录为`r3-handoff*.json`。
