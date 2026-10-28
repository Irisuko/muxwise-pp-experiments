"""Opt-in, diagnostic-only logits capture; never on performance PYTHONPATH."""
import os

if os.environ.get('P11A_LOGITS_PROBE'):
    from collections import defaultdict
    from pathlib import Path
    import torch
    from sglang.srt.model_executor.model_runner import ModelRunner

    _root = Path(os.environ['P11A_LOGITS_PROBE'])
    _counts = defaultdict(int)
    _original = ModelRunner.sample

    def _sample(self, logits_output, forward_batch):
        marker = _root / 'active-case'
        try:
            case = marker.read_text().strip()
        except FileNotFoundError:
            return _original(self, logits_output, forward_batch)
        raw = logits_output.next_token_logits.detach().float().cpu()
        ids = _original(self, logits_output, forward_batch)
        index = _counts[case]
        _counts[case] += 1
        torch.save(dict(logits=raw, ids=ids.detach().cpu(),
                        seq_lens=forward_batch.seq_lens.detach().cpu(),
                        mode=str(forward_batch.forward_mode)),
                   _root / f'{case}-pp{self.pp_rank}-tp{self.tp_rank}-{index:04d}.pt')
        return ids

    ModelRunner.sample = _sample

    if os.environ.get('P11A_LAYER_PROBE'):
        import json
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        from sglang.srt.models.llama import LlamaDecoderLayer

        _layers = set(map(int, os.environ['P11A_LAYER_PROBE'].split(',')))
        _layer_forward = LlamaDecoderLayer.forward

        def _forward(self, positions, hidden_states, forward_batch, residual):
            layer = self.self_attn.attn.layer_id
            try:
                case = (_root / 'active-case').read_text().strip()
            except FileNotFoundError:
                case = None
            if case is None or layer not in _layers:
                return _layer_forward(self, positions, hidden_states, forward_batch, residual)
            pos = positions.detach().cpu()
            total = json.loads((_root / 'case-lengths.json').read_text())[case]
            wanted = {0, 1, 100} | set(range(max(0, total - 32), total))
            for boundary in range(1024, total, 1024):
                wanted.update(range(boundary - 2, boundary + 3))
            indices = [i for i, value in enumerate(pos.tolist()) if value in wanted]
            selected = pos[indices]
            rank = get_tensor_model_parallel_rank()

            def pack(value):
                if isinstance(value, torch.Tensor):
                    return dict(shape=list(value.shape), value=value[indices].detach().cpu())
                if isinstance(value, (list, tuple)):
                    return [pack(x) for x in value]
                return None

            def dump(name, value):
                torch.save(dict(positions=selected, data=pack(value)),
                           _root / f'{case}-layer{layer}-tp{rank}-start{int(pos[0])}-{name}.pt')

            dump('input', (hidden_states, residual))
            modules = dict(input_norm=self.input_layernorm,
                           qkv=self.self_attn.qkv_proj, rope=self.self_attn.rotary_emb,
                           attention=self.self_attn.attn, o_proj=self.self_attn.o_proj,
                           post_norm=self.post_attention_layernorm, mlp=self.mlp)
            hooks = [m.register_forward_hook(lambda module, args, result, name=name: dump(name, result))
                     for name, m in modules.items()]
            try:
                result = _layer_forward(self, positions, hidden_states, forward_batch, residual)
                dump('output', result)
                return result
            finally:
                for hook in hooks:
                    hook.remove()

        LlamaDecoderLayer.forward = _forward

    if os.environ.get('P11A_FULL_ACCUM'):
        from sglang.srt.models.llama import LlamaAttention
        _attn_forward = LlamaAttention.forward

        def _full_accum_forward(self, positions, hidden_states, forward_batch):
            old = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            change = forward_batch.forward_mode.is_split_prefill()
            if change:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            try:
                return _attn_forward(self, positions, hidden_states, forward_batch)
            finally:
                if change:
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old

        LlamaAttention.forward = _full_accum_forward
