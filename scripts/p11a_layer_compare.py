#!/usr/bin/env python3
"""Align sampled layer rows by absolute position across pipeline chunks."""
import argparse
import json
from pathlib import Path
import re
import torch


def leaves(value, prefix=''):
    if isinstance(value, dict):
        yield prefix, value['value']
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, prefix + '/' + str(i))


def read(root, case):
    values = {}
    for path in (root / 'logits').glob(f'cold-{case}-layer*-tp*-start*-*.pt'):
        m = re.fullmatch(r'cold-\d+-layer(\d+)-tp(\d+)-start\d+-(.*)\.pt', path.name)
        layer, tp, name = m.groups()
        x = torch.load(path, map_location='cpu', weights_only=True)
        for component, tensor in leaves(x['data']):
            key = (int(layer), int(tp), name + component)
            for pos, row in zip(x['positions'].tolist(), tensor):
                assert (key, pos) not in values
                values[key, pos] = row.float()
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--case', type=int, required=True)
    x = p.parse_args()
    a, b = read(x.reference, x.case), read(x.candidate, x.case)
    assert a.keys() == b.keys(), (len(a), len(b), list(a.keys()-b.keys())[:3], list(b.keys()-a.keys())[:3])
    stats = {}
    for (key, pos), value in a.items():
        d = (value - b[key, pos]).abs()
        s = stats.setdefault(key, dict(layer=key[0], tp=key[1], component=key[2], rows=0,
                                      different_rows=0, max_abs=0.0, first_differing_position=None))
        s['rows'] += 1
        s['max_abs'] = max(s['max_abs'], float(d.max()))
        if torch.any(d):
            s['different_rows'] += 1
            old = s['first_differing_position']
            s['first_differing_position'] = pos if old is None else min(old, pos)
    result = [stats[k] for k in sorted(stats)]
    (x.candidate / f'layer-comparison-{x.case}.json').write_text(json.dumps(result, indent=2)+'\n')
    for s in result:
        print(s)


if __name__ == '__main__':
    main()
