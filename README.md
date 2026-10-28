# MuxWise-PP experiments

This repo holds the experiment materials for adding pipeline parallelism to MuxWise (PD-Multiplexing on SGLang 0.5.3) and porting FastPP's scheduling ideas. It covers the design docs, phase notes, launch/benchmark scripts, configs and aggregated results. The code itself lives in [Irisuko/muxwise-pp-sglang](https://github.com/Irisuko/muxwise-pp-sglang) (branch `muxwise-pp`).

- Results report: [RESULTS.md](RESULTS.md)
- Design: [DESIGN.md](DESIGN.md); per-phase implementation notes: `P*_NOTES.md`
- Hardware: 4×RTX 4090 (no P2P). Models: Llama3.1-8B-Instruct, Mistral-Small-24B-Instruct-2501

## Code version tags (in the code repo)

| tag | commit | meaning |
|---|---|---|
| muxpp-p1 | 88d3eab9e | P1: base MuxWise-PP |
| muxpp-v1-p3b | bd849d628 | P3b formal version (P2 FastPP port) |
| muxpp-p2-0 | c74f56ff1 | P2-0: pipelined prefill + LRU replay |
| muxpp-v3 | f91e8a18d | v3: async decode ring + MLP chunking |
| muxpp-tp-p8a | d8dfbfaba | TP support (PP2×TP2) |
| muxpp-v4 | f6833f24e | v4: prefill-first partition choice |
| big-model-p10b | 4c10135d2 | Mistral-24B support and memory fixes |
| muxpp-v5 | 1dcbb285e | v5: chunked pipeline prefill + token-based inflight control |
| pd-r2-isolated | 19261937a | SGLang-PD r2 (branch p7a-preserved-p5a) |
| pd-p5a-frozen | 2374302bb | P5a PD transfer evaluation |

Experiment reports reference these commit hashes directly; the hashes are unchanged.

## Not stored in git

| content | where to get it |
|---|---|
| Full logs `logs/`, `archive/`, per-request `results/*_requests.jsonl` | Attachments on this repo's GitHub Release (`*.tar.zst.part-*`) |
| Traces `bench/*.jsonl` | Regenerate with `bench/build_trace.py` (see below) |
| FlashInfer/Triton caches, wheels | Rebuilt automatically |
| Models | Download yourself (see below) |

## Restoring on a new server

```bash
# 1. Code and experiment materials
mkdir -p /root/workspace && cd /root/workspace
git clone https://github.com/Irisuko/muxwise-pp-sglang.git sglang
cd sglang && git checkout muxwise-pp && cd ..
git clone https://github.com/Irisuko/muxwise-pp-experiments.git muxpp_work

# 2. (Optional) Full logs
cd muxpp_work
gh release download raw-logs-<date> -p '*.part-*'
cat raw-logs.tar.zst.part-* | zstd -d | tar -x

# 3. Environment: Python 3.12 venv; versions are in ENV.md / requirements.lock.txt
python3.12 -m venv /root/venvs/sgl
/root/venvs/sgl/bin/pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 sgl-kernel==0.3.12
/root/venvs/sgl/bin/pip install --no-build-isolation -e '/root/workspace/sglang/python[all]' mooncake-transfer-engine
# If anything breaks, align with requirements.lock.txt (flashinfer-python 0.4.0rc1 needs --no-build-isolation)
# env.sh hardcodes paths (/root/venvs/sgl, /root/workspace/...); update them if the paths change
source env.sh

# 4. Models
#   Llama3.1-8B-Instruct, Mistral-Small-24B-Instruct-2501 (ModelScope: mistralai/Mistral-Small-24B-Instruct-2501)
#   Launch scripts read the model path from scripts/*.sh and configs; edit them to match the new machine

# 5. Traces: get the Mooncake FAST'25 traces
git clone --depth 1 https://github.com/kvcache-ai/Mooncake /root/ext/mooncake
T=/root/ext/mooncake/FAST25-release/traces
L=/path/to/llama3.1-8B-Instruct; M=/path/to/Mistral-Small-24B-Instruct-2501
python bench/build_trace.py --source $T/toolagent_trace.jsonl --start-sec 600  --duration-sec 240 --stretch 2.5 --max-total-len 32768 --tokenizer $L --out bench/toolagent_600_240_x2.5.jsonl
python bench/build_trace.py --source $T/toolagent_trace.jsonl --start-sec 1200 --duration-sec 240 --stretch 2.5 --max-total-len 32768 --tokenizer $L --out bench/dev_toolagent_1200_240_x2.5.jsonl
python bench/build_trace.py --source $T/conversation_trace.jsonl --start-sec 600  --duration-sec 240 --stretch 2.5 --max-total-len 32768 --tokenizer $L --out bench/conversation_600_240_x2.5.jsonl
python bench/build_trace.py --source $T/conversation_trace.jsonl --start-sec 1200 --duration-sec 240 --stretch 2.5 --max-total-len 32768 --tokenizer $L --out bench/dev_conversation_1200_240_x2.5.jsonl
python bench/build_trace.py --source $T/toolagent_trace.jsonl --start-sec 600  --duration-sec 80 --stretch 7.5 --max-total-len 32768 --tokenizer $M --out bench/big_toolagent_600_80_x7.5.jsonl
python bench/build_trace.py --source $T/toolagent_trace.jsonl --start-sec 1200 --duration-sec 80 --stretch 7.5 --max-total-len 32768 --tokenizer $M --out bench/dev_big_toolagent_1200_80_x7.5.jsonl
# Compare the request counts against bench/*.stats.json to confirm the regenerated traces match
```

Scripts reference absolute paths of the original machine (`/root/workspace/...`, `/dev/shm/models/...`, `/root/autodl-pvt-A40-hdd-1/...`). Using the same directory layout avoids most edits.
