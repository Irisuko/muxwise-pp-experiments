import json

import torch
from sgl_kernel import spatial

torch.cuda.init()
assert spatial.get_sm_available(0) == 128
partitions = [[96, 32], [80, 48], [64, 64]]
results = []
for a, b in partitions:
    raw = torch.ops.sgl_kernel.create_greenctx_stream_by_value(a, b, 0)
    assert list(raw[2:]) == [a, b], raw
    sa, sb = spatial.create_greenctx_stream_by_value(a, b, 0)
    tensors = []
    for stream in [sa, sb]:
        with torch.cuda.stream(stream):
            x = torch.arange(1024, device='cuda', dtype=torch.float32)
            tensors.append((x * 2).sum())
    torch.cuda.synchronize()
    assert all(t.item() == 1047552 for t in tensors)
    results.append({'requested_sms': [a, b], 'actual_sms': raw[2:],
                    'streams': [sa.cuda_stream, sb.cuda_stream], 'tensor_check': 'PASS'})
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0), 'sms': spatial.get_sm_available(0),
                  'partitions': results}, indent=2))
