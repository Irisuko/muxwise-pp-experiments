#!/usr/bin/env python3
"""Check whether the historical random-ID oracle is stable in ordinary TP1."""
import json
import random
import signal
from pathlib import Path

import p3_runner as p3


def main():
    c = p3.Cluster('chunked', p3.ROOT/'logs/p4a/normal-diagnostic')
    def term(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, term)
    try:
        c.start()
        rng = random.Random(20020)
        cases = [[128000]+[rng.randrange(100,120000) for _ in range(63+i*3)] for i in range(20)]
        reference = json.loads((p3.ROOT/'logs/p3/normal-correctness20.json').read_text())
        comparisons = {}
        for worker in range(4):
            url = f'http://127.0.0.1:{31000+worker}'
            p3.post(url+'/flush_cache', timeout=30).raise_for_status()
            results = [p3.generate(url, ids, 16) for ids in cases]
            p3.save(c.path/f'worker{worker}-random20.json', results)
            comparisons[str(worker)] = [i for i,(a,b) in enumerate(zip(results,reference)) if a['response']['text'] != b['response']['text']]
            print('ordinary TP1', worker, 'mismatch indices', comparisons[str(worker)], flush=True)
        p3.save(c.path/'comparison.json', comparisons)
        c.meta['status'] = 'PASS'
    except BaseException as exc:
        c.meta.update(status='FAIL', error=repr(exc))
        raise
    finally:
        c.close()


if __name__ == '__main__':
    main()
