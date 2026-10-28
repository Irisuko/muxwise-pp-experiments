#!/usr/bin/env python3
"""Move only explicitly superseded v3 evidence, verifying every file hash."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

import p9b_report as r


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    decisions=r.load(r.AUDIT/'replacement-decisions.json')
    manifest_path=r.ARCHIVE/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        for row in manifest['files']:
            assert sha(r.ROOT/row['destination'])==row['sha256'],row
        print('Existing archive verified; no moves repeated')
        return
    r.ARCHIVE.mkdir(parents=True,exist_ok=True)
    files=[]
    def transfer(source,destination,move):
        assert source.is_file() and not destination.exists(),(source,destination)
        digest=sha(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if move:shutil.move(str(source),str(destination))
        else:shutil.copy2(source,destination)
        assert sha(destination)==digest,destination
        files.append(dict(source=str(source.relative_to(r.ROOT)),destination=str(destination.relative_to(r.ROOT)),
                          sha256=digest,operation='move' if move else 'copy'))
    # Preserve the previous presentation before refreshing aggregate reports/figures.
    aggregates=[]
    for pattern in ['matrix_*','best_*','final_*','summary.*','ttft_*','tpot_*','e2e_*','slo_*']:
        aggregates.extend(p for p in r.OUT.glob(pattern) if p.is_file())
    for p in dict.fromkeys(aggregates):
        transfer(p,r.ARCHIVE/'previous_report/results'/p.name,False)
    transfer(r.AUDIT/'RESULTS-before-P9b.md',r.ARCHIVE/'previous_report/RESULTS.md',False)
    replaced=[]
    for old,new in r.PAIRS.items():
        if not decisions[new]['replace']:continue
        assert (r.OUT/f'{new}_summary.json').exists()
        marker=r.load(r.OUT/f'{old}.formal-started.json')
        folder=Path(marker['log_dir'])
        targets=list(r.OUT.glob(old+'_*.json*'))+[r.OUT/f'{old}.formal-started.json']
        targets+=sorted(p for p in folder.rglob('*') if p.is_file())
        for p in targets:transfer(p,r.ARCHIVE/p.relative_to(r.ROOT),True)
        # Remove only emptied evidence directories, never unrelated paths.
        for p in sorted((p for p in folder.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
            p.rmdir()
        folder.rmdir()
        replaced.append(dict(old=old,new=new,reason=decisions[new]['reason']))
    r.save(manifest_path,dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                              replaced=replaced,files=files))
    (r.ARCHIVE/'README.md').write_text('# P9b替代版本归档\n\n'
        '原始请求、summary、正式锁与正式服务日志按原相对路径保留；文件内容及SHA256不变。'
        '旧锁中的绝对log_dir保持历史原文，当前读取器会将不存在的原路径映射到本目录。'
        'previous_report保存更新前的报告/表/图快照。\n\n'
        '替换原因与逐文件校验见[manifest.json](manifest.json)。\n')
    print(json.dumps({'replaced':replaced,'files':len(files)},ensure_ascii=False))


if __name__=='__main__':main()
