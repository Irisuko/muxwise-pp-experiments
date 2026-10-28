#!/usr/bin/env python3
"""Compare logits only while both serial generations have the same prefix."""
import argparse
import json
from pathlib import Path
import torch


def top(row):
    values, indices = row.topk(5)
    return list(zip(indices.tolist(), values.tolist()))


def compare(reference, candidate, case):
    def files(root):
        return sorted((root / 'logits').glob(f'cold-{case}-pp*-tp0-*.pt'))
    refs, actuals = files(reference), files(candidate)
    assert refs and len(refs) == len(actuals), (case, len(refs), len(actuals))
    rows = []
    for position, (left, right) in enumerate(zip(refs, actuals)):
        a = torch.load(left, map_location='cpu', weights_only=True)
        b = torch.load(right, map_location='cpu', weights_only=True)
        assert torch.equal(a['seq_lens'], b['seq_lens'])
        assert a['logits'].shape == b['logits'].shape and len(a['logits']) == 1
        u, v = a['logits'][0], b['logits'][0]
        i, j = int(a['ids'][0]), int(b['ids'][0])
        d = (u - v).abs()
        rows.append(dict(position=position, reference_id=i, candidate_id=j,
                         max_abs=float(d.max()), mean_abs=float(d.mean()),
                         reference_top5=top(u), candidate_top5=top(v),
                         reference_choice_margin=float(u[i] - u[j]),
                         candidate_choice_margin=float(v[j] - v[i])))
        if i != j:
            break
    return dict(case=case, compared_same_prefix_positions=len(rows),
                first_logit_difference=next((x['position'] for x in rows if x['max_abs']), None),
                first_token_difference=next((x['position'] for x in rows if x['reference_id'] != x['candidate_id']), None),
                rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--cases', type=int, nargs='+', required=True)
    x = p.parse_args()
    report = dict(reference=str(x.reference), candidate=str(x.candidate),
                  cases=[compare(x.reference, x.candidate, i) for i in x.cases])
    (x.candidate / 'numeric-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    for c in report['cases']:
        print('case', c['case'], 'first differing logits', c['first_logit_difference'],
              'first differing token', c['first_token_difference'], 'last compared', c['rows'][-1])


if __name__ == '__main__':
    main()
