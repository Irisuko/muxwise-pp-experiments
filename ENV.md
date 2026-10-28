# MuxWise / SGLang 环境（P0）

验证日期：2026-10-20。**环境搭建成功，普通单卡、PDMux 单卡、普通 PP=2 三项冒烟测试全部通过。**

## 激活与硬件

```bash
source /root/workspace/muxpp_work/env.sh
which python  # /root/venvs/sgl/bin/python
```

- 独立 venv：`/root/venvs/sgl`，Python **3.12.3**，`include-system-site-packages = false`。
- 4 × RTX 4090，每卡 24GB、128 SM、compute capability 8.9；无 NVLink。
- 驱动 **580.159.03**，`nvidia-smi` 中 CUDA 13.0 是驱动支持版本。
- Torch wheel 自带 CUDA runtime **12.8**；系统 nvcc **12.4.131**，`CUDA_HOME=/usr/local/cuda`（指向 cuda-12.4）。此次 FlashInfer JIT 编译及全部测试正常，无需安装另一套 toolkit。
- 仓库：`/root/workspace/sglang`，测试提交 `eeac5148f456428a05823c330e1d40afb0ef2438`。
- **本任务未修改仓库源码，也未重编 sgl-kernel。** 测试结束时 `git status --short`、`git diff` 均为空，见 [git-status-final.txt](logs/git-status-final.txt)、[source-diff.patch](logs/source-diff.patch)。

`env.sh` 设置 editable 包的 `PYTHONPATH`、`TORCH_CUDA_ARCH_LIST=8.9`、`MAX_JOBS=16`、`OMP_NUM_THREADS=8`，并将 FlashInfer/Torch extension/Triton 缓存放到本目录的 `cache/` 下。没有固定 `CUDA_VISIBLE_DEVICES`，每次启动按需设置。为使用本地模型，设置了 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`；需要联网下载模型时可取消这两个变量。

## 最终版本

| 包 | 最终版本 |
|---|---|
| sglang | 0.5.3rc0，editable `/root/workspace/sglang/python` |
| torch | 2.8.0+cu128（包元数据 / pip freeze 为 2.8.0） |
| torchaudio / torchvision | 2.8.0 / 0.23.0 |
| sgl-kernel | 0.3.12，官方 Linux x86_64 wheel，包含 spatial |
| flashinfer-python | 0.4.0rc1 |
| mooncake-transfer-engine | 0.3.13.post1 |
| transformers / tokenizers | 4.56.1 / 0.22.2 |
| torchao / torch-memory-saver | 0.9.0 / 0.0.8 |
| xgrammar / llguidance | 0.1.24 / 0.7.30 |
| nvidia-cutlass-dsl / cuda-python | 4.2.0 / 12.9.10 |
| numpy / datasets / pyarrow | 2.5.4 / 4.0.0 / 26.0.0 |
| fsspec | 2025.3.0 |

全部已安装包的精确版本见 [requirements.lock.txt](requirements.lock.txt) 和 [packages-final.json](logs/packages-final.json)。核心依赖遵守仓库 `python/pyproject.toml` 的版本约束。安装了 `[all]` 后仅移除可选的 `decord`，保留核心 SRT 依赖和 test extra。

最终 `python -m pip check` 输出 **No broken requirements found.**，见 [pip-check-final.log](logs/pip-check-final.log)。Torch、SGLang、sgl-kernel、FlashInfer、datasets 和 `from mooncake.engine import TransferEngine` 均导入成功，见 [final-imports.log](logs/final-imports.log)。Mooncake 本次按要求验证导入，未运行分离式 P/D 跨进程 KV 传输测试。

## 安装过程与问题处理

本任务所有安装、卸载均指定 `/root/venvs/sgl` 的 Python 或 uv；没有向 base conda 安装包。创建环境使用：

```bash
/root/miniconda3/bin/python -m venv /root/venvs/sgl
/root/venvs/sgl/bin/python -m pip install --no-cache-dir -U pip uv
/root/venvs/sgl/bin/uv pip install --python /root/venvs/sgl/bin/python --no-cache \
  torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 sgl-kernel==0.3.12
```

Torch 等核心 wheel 通过 uv 默认 PyPI 源直接安装；后续包使用可用的阿里云镜像。最终完成 SGLang / FlashInfer 的关键命令为：

```bash
/root/venvs/sgl/bin/python -m pip install --no-cache-dir wheel ninja packaging requests
/root/venvs/sgl/bin/uv pip install --python /root/venvs/sgl/bin/python \
  --index-url http://mirrors.aliyun.com/pypi/simple \
  --no-build-isolation-package flashinfer-python \
  -e '/root/workspace/sglang/python[all]' mooncake-transfer-engine
/root/venvs/sgl/bin/uv pip install --python /root/venvs/sgl/bin/python \
  --index-url http://mirrors.aliyun.com/pypi/simple datasets==4.0.0
/root/venvs/sgl/bin/python -m pip uninstall -y decord
```

实际日志：[install-core.log](logs/install-core.log)、[install-all-no-isolation.log](logs/install-all-no-isolation.log)、[fix-datasets.log](logs/fix-datasets.log)。上述为实际安装过程；重建时应使用最终锁文件，避免未约束的依赖再次漂移。

1. **FlashInfer 构建隔离下载多余 Torch。** [0.4.0rc1 的 PyPI 发布](https://pypi.org/project/flashinfer-python/0.4.0rc1/)提供 sdist；其构建后端请求未锁版本的 Torch，隔离构建开始下载另一套大型 CUDA 依赖。中止该安装后，对 FlashInfer 单独禁用构建隔离，复用已经安装的 Torch 2.8.0，构建成功。未安装独立 cubin/jit-cache 包；本次用源码包附带的头文件和系统 nvcc 完成运行时 JIT。
2. **datasets / PyArrow 运行时不兼容。** 初次解析得到 datasets 2.14.4 + PyArrow 26.0.0，导入报 `PyExtensionType` 不存在。固定 datasets 4.0.0，解析器同时将 fsspec 调整到 2025.3.0；导入和依赖检查均通过。
3. **可选 decord 的 wheel 标签不一致。** decord 0.6.0 可以导入，但包内部 WHEEL 标记为 `cp36-cp36m-manylinux2010_x86_64`，导致 Python 3.12 下 `pip check` 报不兼容。按照任务允许的可选依赖例外将其移除；Llama 文本推理不依赖它。
4. **仓库目录遮蔽 editable 包。** 在 `/root/workspace` 中直接导入时，仓库根目录 `sglang/` 可被 Python 当作 namespace package。`env.sh` 将真正的 `/root/workspace/sglang/python` 加入 `PYTHONPATH`，保证 `sglang.__file__` 和 `sglang.__version__` 正确。
5. **已有警告。** fork 对 Torch > 2.6.x 发出 PDMux 潜在性能下降警告。本任务遵守 pyproject 的 Torch 2.8.0 约束，功能验证通过；未评估相对 Torch 2.6 的性能变化。另有 pynvml 弃用和 PP 的非 writable buffer 警告，未造成此次请求失败。

构建后的 FlashInfer wheel 已保留在 [artifacts/](artifacts/)，SHA256 在 [SHA256SUMS](artifacts/SHA256SUMS)。临时 uv 下载缓存已清理，运行时 JIT 缓存保留。以相同平台重建已建好 venv 时可使用：

```bash
python -m pip install --find-links /root/workspace/muxpp_work/artifacts \
  -r /root/workspace/muxpp_work/requirements.lock.txt
```

**Base 环境审计：** 本任务开始与结束的 base 包快照有差异，但不能将其说成“base 完全未变”。同机并行的 P3a 任务在 `logs/p3a_codex.log` 中明确执行了 `/root/miniconda3/bin/python -m pip install aiohttp tokenizers`，引入相关依赖变化。本任务没有回滚该并行任务的改动。核对时 base Torch 仍为 2.5.1+cu124，Jupyter PID 813 和 TensorBoard PID 816 持续运行。证据见 [base-before.txt](logs/base-before.txt)、[base-after.txt](logs/base-after.txt)、[base-diff.txt](logs/base-diff.txt)。结束快照使用 `env -u PYTHONPATH`，避免 editable 项目元数据干扰。

## spatial 验证

```bash
source /root/workspace/muxpp_work/env.sh
CUDA_VISIBLE_DEVICES=0 python -c \
  'from sgl_kernel import spatial; print(spatial.get_sm_available(0)); print(spatial.create_greenctx_stream_by_value(96,32,0))'
CUDA_VISIBLE_DEVICES=0 python /root/workspace/muxpp_work/scripts/verify_spatial.py
```

结果：SM 数 **128**；96/32、80/48、64/64 三种分区的底层算子报告实际 SM 数均与请求相同；每对 ExternalStream 都执行张量运算、同步并校验数值成功。官方 [sgl-kernel 0.3.12 wheel](https://pypi.org/project/sgl-kernel/0.3.12/) 已含 `spatial.py` 和 `spatial_ops.abi3.so`。见 [spatial-wheel.log](logs/spatial-wheel.log)、[spatial-verify.log](logs/spatial-verify.log)。

## 冒烟测试命令

以下脚本会启动服务、等待 `/health`、向 `/generate` 发请求，最后清理本次启动的进程组及子进程。**逐项顺序执行**，共同使用端口 30000；端口已占用时脚本拒绝启动。

```bash
source /root/workspace/muxpp_work/env.sh
python /root/workspace/muxpp_work/scripts/smoke.py normal > /root/workspace/muxpp_work/logs/normal-client.log 2>&1
python /root/workspace/muxpp_work/scripts/smoke.py pdmux > /root/workspace/muxpp_work/logs/pdmux-client.log 2>&1
python /root/workspace/muxpp_work/scripts/smoke.py pp > /root/workspace/muxpp_work/logs/pp-client.log 2>&1
```

等价服务启动命令如下，手工运行时须在请求完成后关闭服务：

```bash
source /root/workspace/muxpp_work/env.sh
MODEL=/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct

# 普通单卡
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL" --tp 1 --port 30000 --disable-overlap-schedule

# PDMux 单卡
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path "$MODEL" --tp 1 --port 30000 --disable-overlap-schedule \
  --enable-pdmux --pdmux-config-path /root/workspace/muxpp_work/configs/pdmux_4090.yml \
  --chunked-prefill-size -1 --mem-fraction-static 0.8 --log-level debug

# 普通 PP=2，两卡
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path "$MODEL" --tp 1 --pp-size 2 --port 30000 --disable-overlap-schedule
```

简单手工请求示例（完整聊天模板、并发和 token 计数见 `scripts/smoke.py`）：

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"What is 2 + 2?","sampling_params":{"temperature":0,"max_new_tokens":32}}'
```

## 实测结果与边界

| 测试 | 结果 | 服务就绪用时 | 证据 |
|---|---|---:|---|
| GPU0 普通模式 | PASS；问 2+2，返回 `4` | 61.067s，含首次 JIT | [normal-result.json](logs/normal-result.json)、[服务日志](logs/normal-server.log) |
| GPU0 PDMux | PASS；基本请求 + 6 个并发混合请求全部返回 | 25.035s | [pdmux-result.json](logs/pdmux-result.json)、[服务日志](logs/pdmux-server.log) |
| GPU0,1 普通 PP=2 | PASS；两个 scheduler PP0/PP1 正常运行，返回 `4` | 23.033s | [pp-result.json](logs/pp-result.json)、[服务日志](logs/pp-server.log) |

PDMux 配置 [pdmux_4090.yml](configs/pdmux_4090.yml) 为 5 组：默认 128/0、96/32（decode batch >=1）、80/48（>=8）、64/64（>=16）、默认 0/128。所有非零分区满足 sm_89 最少 4 SM、步长 2 的限制，手工分区总和均为 128。

并发测试先启动 512-token 长输出请求，再延迟 150ms 同时提交长、中、短 prompt。长 prompt 服务端确认为 **5998 tokens**，返回 28 tokens，内容正确概括苹果树和梨树的重复日志；其他请求包含 1244-token 中等 prompt 和三个短 prompt。日志明确出现 `Adjusting stream groups: 1, prefill sm: 96, decode sm: 32`，证明执行了分区调度。80/48、64/64 通过了独立 green stream 运算检查和服务图捕获；本次 6 路并发未触发它们的 8/16 decode batch 阈值。

这些用时只用于证明可运行，不作为性能基准，首次编译和缓存状态不同。本任务分别验证 PDMux（PP=1）与普通 PP=2；当前 fork 的参数检查仍不支持 `--enable-pdmux` 与 `--pp-size 2` 组合，组合功能属于后续源码工作。

**测试后清理：** 三个服务都收到测试脚本发出的终止信号并退出；结果文件中的 `server_exit_code=-9` 是清理服务进程产生，不是请求失败。最终 GPU compute process 列表为空，30000 无监听，见 [nvidia-smi-final.txt](logs/nvidia-smi-final.txt)。全部请求断言先通过，随后才关闭服务。
