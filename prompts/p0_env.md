# 任务 P0：搭建可运行 MuxWise(SGLang 0.5.3rc0) 的环境（4×RTX 4090）

背景：/root/workspace/sglang 是一个 SGLang 0.5.3rc0 的 fork，增加了 PD-Multiplexing（MuxWise，`--enable-pdmux`，依赖 `sgl_kernel.spatial` 的 green context stream）。机器是 4×RTX 4090 (sm_89, 128 SMs, 24GB, 无 NVLink)，驱动 580 / CUDA 13.0 驱动，系统 nvcc 12.4。当前 python 是 /root/miniconda3 (3.12, torch 2.5.1)。pip 源是阿里云镜像（可用）。GitHub 可访问。

目标：
1. 新建一个独立的 Python 3.12 虚拟环境 /root/venvs/sgl（用 uv 或 python -m venv 都行），不要破坏 base conda 环境（jupyter/tensorboard 在用）。
2. 在里面安装：torch==2.8.0 (cu128 或 cu126 wheel 均可，优先能直接 pip 装上的)、sgl-kernel==0.3.12、flashinfer_python==0.4.0rc1（及其 cubin/jit-cache 包如需要），然后 `pip install -e /root/workspace/sglang/python[all]`（如 [all] 里某些可选依赖装不上可以去掉，但核心 srt 依赖必须有）。注意 pyproject 里的版本约束，以它为准。
3. 关键验证：`python -c "from sgl_kernel import spatial; print(spatial.get_sm_available(0))"` 必须能跑，并能用 `spatial.create_greenctx_stream_by_value(96, 32, 0)` 创建 green ctx stream。若 PyPI 的 sgl-kernel 0.3.12 wheel 不含 spatial 或在 sm_89 上不可用，则从 /root/workspace/sglang/sgl-kernel 源码编译（可只编 sm_89 加速：设置 TORCH_CUDA_ARCH_LIST/ CMake 参数只编 89；可能需要安装匹配 torch 的 CUDA toolkit，可用 pip 的 nvidia-cuda-nvcc 或 conda 安装 cuda-toolkit 12.8）。48 核 CPU 可用 MAX_JOBS 控制。
4. 安装 PD 分离所需的传输后端：`mooncake-transfer-engine`（pip）。验证 import 成功。
5. 冒烟测试（用 GPU0）：
   - 模型：/root/autodl-pub-RTX4090-hdd-1/models/llama3.1-8B-Instruct
   - 普通启动：`python -m sglang.launch_server --model-path <模型> --tp 1 --port 30000 --disable-overlap-schedule`，发一个 /generate 请求确认输出正常，然后关闭。
   - pdmux 启动：写一个适配 4090(128 SM, sm_89: 每份最少4 SM、步长2) 的 pdmux 配置 /root/workspace/muxpp_work/configs/pdmux_4090.yml（sm_group_num 5~8，manual_divisions 例如 [96,32,1],[80,48,8],[64,64,16] 之类，按 128 SM 合理设计），`--enable-pdmux --pdmux-config-path ... --chunked-prefill-size -1 --disable-overlap-schedule --mem-fraction-static 0.8`，发几个并发请求（混合长短 prompt，含一个 ~6000 token 的长 prompt）确认能正常返回，然后关闭。
   - PP 启动：`--pp-size 2`（GPU0,1）普通模式，确认可用后关闭。
6. 产出：/root/workspace/muxpp_work/ENV.md，记录：激活方式、各包最终版本、遇到的问题及解决方法、冒烟测试命令与结果。同时写 /root/workspace/muxpp_work/env.sh（source 后即进入环境并设置必要环境变量）。

约束：
- 不要修改 /root/workspace/sglang 下的源代码（编译产物除外）；若必须改才能跑通，在 ENV.md 中明确记录 diff。
- 所有服务进程测试完要杀掉，不要留下占用 GPU 的进程。
- 长时间命令请把日志写到 /root/workspace/muxpp_work/logs/。
- 完成后最后一条消息给出简明总结（成功/失败、关键版本、遗留问题）。
