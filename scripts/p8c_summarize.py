#!/usr/bin/env python3
"""CPU-only P8c reporting; reuse frozen metrics without changing benchmark/source files."""
import ast
import collections
import csv
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
sys.path.insert(0, str(ROOT / 'bench'))
import summarize as summary
import numpy as np

NEW = 'muxpp_pp2tp2'
TZ = dt.timezone(dt.timedelta(hours=8))
MANIFEST = {}


def read(path):
    value = path.read_bytes()
    MANIFEST[str(path.relative_to(ROOT))] = hashlib.sha256(value).hexdigest()
    return value.decode(errors='replace')


def load(path):
    return json.loads(read(path))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def dist(values):
    values = list(values)
    return {'n': len(values), **({
        'p50': float(np.percentile(values, 50)),
        'p99': float(np.percentile(values, 99)),
        'mean': float(np.mean(values)), 'max': float(max(values)),
    } if values else {})}


def csvwrite(path, rows):
    if rows:
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)


def parse_logs(s, records):
    label = s['label']
    folder = Path(load(OUT / f'{label}.formal-started.json')['log_dir'])
    run = load(folder / 'run.json')
    origin, duration = s['run_started_unix'], s['duration_sec']
    batches, windows, idle, errors = [], {}, {}, []
    for worker in run['server_info']:
        path = folder / f'{worker}.log'
        if not path.exists():
            continue
        snapshots = []
        for lineno, line in enumerate(read(path).splitlines(), 1):
            if re.search(r'Traceback|RuntimeError|AssertionError|OutOfMemoryError|out of memory|Scheduler.*exception', line):
                errors.append({'file': str(path.relative_to(ROOT)), 'line': lineno, 'text': line})
            m = re.match(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)([^\]]*)\]', line)
            if not m:
                continue
            t = dt.datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ).timestamp() - origin
            if 'idle self-check PASS' in line:
                idle[m[2].strip()] = {'file': str(path.relative_to(ROOT)), 'line': lineno, 'text': line}
            # One logical admission stream. Other TP/PP ranks only replay the decision.
            if any(int(v) for v in re.findall(r'(?:TP|PP)(\d+)', m[2])):
                continue
            if 'MuxWise-PP summary stages=' in line and 0 <= t and t + 1 <= duration:
                stages = ast.literal_eval(line.split('summary stages=', 1)[1].split(' mbs=', 1)[0])
                snapshots.append((t, lineno, stages))
            if not -1 < t <= duration:
                continue
            admit = re.search(r'MuxWise-PP admit batch=(\d+) requests=(\d+) tokens=(\d+) cached=(\d+) inflight=(\d+)', line)
            prefill = re.search(r'Prefill batch\. #new-seq: (\d+), #new-token: (-?\d+), #cached-token: (\d+)', line)
            if admit:
                batch, n, new, cached, inflight = map(int, admit.groups())
            elif prefill:
                n, new, cached = map(int, prefill.groups())
                batch, inflight = None, None
            else:
                continue
            # Same formal trace lower bound as P8b; excludes client warmups and health probes.
            if new + cached < 892:
                continue
            batches.append(dict(label=label, worker=worker, file=str(path.relative_to(ROOT)),
                                line=lineno, t=t, n=n, new=new, cached=cached,
                                batch=batch, inflight=inflight))
        if len(snapshots) >= 2:
            ta, la, a = snapshots[0]
            tb, lb, b = snapshots[-1]
            stages = {}
            for st in a.keys() & b.keys():
                x, y = a[st]['timing'], b[st]['timing']
                z = {'elapsed': y['elapsed'] - x['elapsed']}
                z['cuda'] = {k: v - x['cuda'].get(k, 0) for k, v in y['cuda'].items()}
                z['wall'] = {side: {k: v - x['wall'].get(side, {}).get(k, 0) for k, v in vals.items()}
                             for side, vals in y['wall'].items()}
                z['slices'] = {k: [v[i] - x['slices'].get(k, [0, 0])[i] for i in range(2)]
                               for k, v in y['slices'].items()}
                if a[st].get('admission') and b[st].get('admission'):
                    z['admission_wall'] = {k: v - a[st]['admission']['wall'].get(k, 0)
                                           for k, v in b[st]['admission']['wall'].items()}
                    z['admission_fraction'] = {k: v / z['elapsed'] for k, v in z['admission_wall'].items()}
                stages[str(st)] = z
            windows[worker] = dict(start_t=ta, end_t=tb, first_line=la, last_line=lb,
                                   file=str(path.relative_to(ROOT)), stages=stages)
    # Bidirectionally unique exact token/cache signature plus overlapping start-time interval.
    signatures = collections.defaultdict(list)
    for r in records:
        if r['success']:
            signatures[(r['input_len'], r['cached_tokens'])].append(r)
    candidates, reverse = {}, collections.defaultdict(list)
    for i, b in enumerate(batches):
        if b['n'] != 1:
            continue
        candidates[i] = [r for r in signatures[(b['new'] + b['cached'], b['cached'])]
                         if max(b['t'], r['sent_at']) <= min(b['t'] + 1, r['first_token_at'])]
        for r in candidates[i]:
            reverse[r['idx']].append(i)
    matches = []
    for i, candidates_i in candidates.items():
        if len(candidates_i) != 1 or len(reverse[candidates_i[0]['idx']]) != 1:
            continue
        r, b = candidates_i[0], batches[i]
        lo, hi = max(b['t'], r['sent_at']), min(b['t'] + 1, r['first_token_at'])
        matches.append({**b, 'idx': r['idx'], 'input': r['input_len'], 'ttft': r['ttft'],
                        'start_lo': lo, 'start_hi': hi,
                        'queue_est': (lo + hi) / 2 - r['sent_at'],
                        'service_lo': r['first_token_at'] - hi,
                        'service_hi': r['first_token_at'] - lo,
                        'service_est': r['first_token_at'] - (lo + hi) / 2})
    counts = dict(requests=sum(b['n'] for b in batches),
                  input_tokens=sum(b['new'] + b['cached'] for b in batches),
                  cached_tokens=sum(b['cached'] for b in batches))
    expected = dict(requests=s['completed'], input_tokens=s['input_tokens'], cached_tokens=s['cached_tokens']['total'])
    return dict(windows=windows, matches=matches, idle=idle, errors=errors,
                counts=counts, expected=expected, conserved=counts == expected), batches


def main():
    # Extend only this reporting process. Frozen benchmark files stay byte-identical.
    summary.MATRIX[NEW] = ('MuxWise-PP', 'TP2PP2')
    paths = [OUT / f'{label}{"_cut300" if label in summary.HISTORICAL else ""}_summary.json'
             for label in summary.MATRIX]
    for path in paths:
        read(path)
        read(OUT / f'{load(path)["label"]}_requests.jsonl')
    runs = summary.prepare_matrix(summary.load_runs([str(p) for p in paths]), OUT)
    # Preserve the established heatmap column spelling internally; display PP2 x TP2 below.
    best = summary.select_best(runs)
    for prefix, selected in [('matrix_', runs), ('best_', best)]:
        summary.write_matrix_tables(selected, OUT, prefix)
        path = OUT / f'{prefix}summary.md'
        value = path.read_text().replace('完整配置矩阵（15行）', '完整配置矩阵（16行）')
        value = value.replace('MuxWise-PP仅TP1，不支持TP2DP2、TP4、TP2PP2；PP1DP4归原版MuxWise。',
                              'MuxWise-PP支持TP，已测PP2×TP2；TP2DP2、TP4这类无PP组合不适用，PP1归原版MuxWise。')
        value = value.replace('| MuxWise-PP | TP2PP2 |', '| MuxWise-PP | PP2×TP2 |')
        value += '\nPP2×TP2与矩阵其它配置运行于不同日期，GPU物理映射可能不同；详见主报告及logs/p8c/runtime-gpu-mapping.json。\n'
        path.write_text(value)
        # Consistent display names in exported machine-readable tables.
        csv_path = OUT / f'{prefix}summary.csv'
        with csv_path.open() as f:
            reader = csv.DictReader(f); fields = reader.fieldnames; rows = list(reader)
        for row in rows:
            if row['label'] == NEW:
                row['matrix.configuration'] = 'PP2×TP2'
        with csv_path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        exported = json.loads(json.dumps([s for s, _ in selected]))
        for s in exported:
            if s['label'] == NEW:
                s['matrix']['configuration'] = 'PP2×TP2'
        save(OUT / f'{prefix}summary.json', exported)

    def save_figure(fig, output, name):
        if name == 'matrix_slo_heatmap.png':
            ax = fig.axes[0]
            for item in ax.texts:
                if item.get_position()[1] == 3 and item.get_text() == 'Unsupported':
                    item.set_text('N/A (PP1)')
            ax.set_xticklabels(['PP2 x TP2' if x.get_text() == 'TP2PP2' else x.get_text()
                                for x in ax.get_xticklabels()], rotation=25, ha='right')
        if name == 'best_main_slo.png':
            ax = fig.axes[0]
            ax.set_xticks(ax.get_xticks(), [x.get_text().replace('TP2PP2', 'PP2 x TP2') for x in ax.get_xticklabels()])
        fig.text(.5, .005, 'One run per configuration; PP2 x TP2 measured on a different date/GPU mapping.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(output / name, dpi=160)
        fig.savefig(output / Path(name).with_suffix('.svg'))
        summary.plt.close(fig)

    summary.save_figure = save_figure
    summary.plot_runs(best, OUT, 'best_')
    summary.plot_matrix(runs, best, OUT)
    indexed = {s['label']: (s, rs) for s, rs in runs}
    labels = [NEW, 'muxpp_pp2dp2', 'muxwise_pp_v3', 'muxwise_tp2dp2', 'muxwise_tp4']
    analyses, all_matches, all_batches = {}, [], []
    for label in labels:
        s, rs = indexed[label]
        result, batches = parse_logs(s, rs)
        all_matches.extend(result['matches']); all_batches.extend(batches)
        itl = [v for r in rs if r['success'] for v in r['itl']]
        result['itl'] = dist(itl)
        result['itl_gt_1s'] = sum(v > 1 for v in itl)
        result['output_buckets'] = {}
        for name, low, high in [('2-8', 2, 8), ('9-64', 9, 64), ('65+', 65, float('inf'))]:
            rr = [r for r in rs if r['success'] and low <= r['output_tokens'] <= high]
            result['output_buckets'][name] = {
                'tpot': dist(r['tpot'] for r in rr),
                'decode_seconds': dist(r['e2e'] - r['ttft'] for r in rr),
                'decode_total_seconds': sum(r['e2e'] - r['ttft'] for r in rr)}
        result['slo_misses'] = {
            'ttft_gt_5': sum(r['ttft'] is not None and r['ttft'] > 5 for r in rs),
            'tpot_gt_100ms': sum(r['tpot'] is not None and r['tpot'] > .1 for r in rs),
            'both': sum(r['ttft'] is not None and r['ttft'] > 5 and r['tpot'] is not None and r['tpot'] > .1 for r in rs),
            'undefined_tpot': sum(r['tpot'] is None for r in rs)}
        analyses[label] = result
    # Same trace idx and exact cache, with single-request admission on both sides.
    new_matches = {m['idx']: m for m in analyses[NEW]['matches']}
    pairs, paired_summary = [], {}
    for old in labels[1:]:
        paired = []
        for before in analyses[old]['matches']:
            after = new_matches.get(before['idx'])
            if after and (before['input'], before['cached']) == (after['input'], after['cached']):
                paired.append({'before_label': old, 'idx': before['idx'], 'new_tokens': before['new'],
                               'cached_tokens': before['cached'],
                               **{f'before_{k}': before[k] for k in ['service_est', 'service_lo', 'service_hi']},
                               **{f'after_{k}': after[k] for k in ['service_est', 'service_lo', 'service_hi']}})
        pairs.extend(paired)
        paired_summary[old] = {}
        for name, subset in [('all', paired), ('uncached_ge8192', [p for p in paired if p['new_tokens'] >= 8192])]:
            if not subset:
                paired_summary[old][name] = {'n': 0}
                continue
            sums = {k: sum(p[k] for p in subset) for k in subset[0] if k.startswith(('before_service', 'after_service'))}
            paired_summary[old][name] = {
                'n': len(subset), 'before_mean': sums['before_service_est'] / len(subset),
                'after_mean': sums['after_service_est'] / len(subset),
                'speed_ratio_before_over_after': sums['before_service_est'] / sums['after_service_est'],
                'quantization_ratio_lo': sums['before_service_lo'] / sums['after_service_hi'],
                'quantization_ratio_hi': sums['before_service_hi'] / sums['after_service_lo'] if sums['after_service_lo'] else None}
    csvwrite(OUT / 'p8c_prefill_matches.csv', all_matches)
    csvwrite(OUT / 'p8c_prefill_pairs.csv', pairs)
    csvwrite(OUT / 'p8c_admissions.csv', all_batches)
    save(OUT / 'p8c_prediction_metrics.json', {'runs': analyses, 'matched_service': paired_summary})
    assert all(a['conserved'] for a in analyses.values()), 'Admission/client totals do not match'
    assert len(runs) == 16 and len(best) == 4
    for rel, sha in MANIFEST.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == sha, rel
    save(ROOT / 'logs/p8c/report-input-sha256.json', MANIFEST)
    print(json.dumps({'matrix_rows': len(runs), 'best': [s['label'] for s, _ in best],
                      'conservation': {l: a['conserved'] for l, a in analyses.items()},
                      'new_summary': indexed[NEW][0]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
