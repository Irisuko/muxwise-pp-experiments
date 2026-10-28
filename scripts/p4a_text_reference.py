#!/usr/bin/env python3
"""Freeze 20 contextual short prompts before testing any PD output."""
import random
import signal

from transformers import AutoTokenizer
import p3_runner as p3


def main():
    tok = AutoTokenizer.from_pretrained(p3.MODEL)
    rng = random.Random(20261020)
    nouns = ['river', 'lantern', 'garden', 'window', 'harbor', 'forest', 'mountain', 'bridge', 'library', 'island']
    adjectives = ['amber', 'blue', 'quiet', 'bright', 'silver', 'green', 'ancient', 'small', 'warm', 'golden']
    cases = []
    for i in range(20):
        records = ['The '+rng.choice(adjectives)+' '+rng.choice(nouns)+' holds '+str(rng.randrange(11,90))+' carefully numbered objects.' for _ in range(4)]
        prompt = '\n'.join(f'Record {j+1}: {s}' for j,s in enumerate(records))
        prompt += f'\nCopy only the full sentence from Record {i%4+1}, exactly ten times, one copy per line. Do not add an introduction or explanation.'
        ids = tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=True, add_generation_prompt=True)
        cases.append({'id':i,'prompt':prompt,'input_ids':ids,'max_new_tokens':32})
    path = p3.ROOT/'logs/p4a/text20-cases.json'
    with path.open('x') as f:
        import json
        json.dump(cases,f,indent=2)
    c = p3.Cluster('chunked',p3.ROOT/'logs/p4a/normal-text20')
    def term(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,term)
    try:
        c.start()
        reference = None
        comparisons = {}
        for worker in range(4):
            url = f'http://127.0.0.1:{31000+worker}'
            p3.post(url+'/flush_cache',timeout=30).raise_for_status()
            results = [p3.generate(url,case['input_ids'],case['max_new_tokens']) for case in cases]
            p3.save(c.path/f'worker{worker}-text20.json',results)
            if reference is None:
                reference = results
            comparisons[str(worker)] = [a['response']['output_ids']==b['response']['output_ids'] and a['response']['text']==b['response']['text'] for a,b in zip(results,reference)]
            print('ordinary TP1',worker,'text20 matches',sum(comparisons[str(worker)]),flush=True)
        p3.save(c.path/'comparison.json',comparisons)
        assert all(all(v) for v in comparisons.values()),'Natural text oracle is not stable across ordinary TP1 instances'
        c.meta['status']='PASS'
    except BaseException as exc:
        c.meta.update(status='FAIL',error=repr(exc))
        raise
    finally:
        c.close()


if __name__=='__main__':
    main()
