"""Pinned public weights, verified HTTP ranges and publisher hashes; no inference or private data."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.request


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def download(args):
    if not re.fullmatch(r'Qwen/Qwen3-VL-(2B|4B)-Instruct',args.repo):raise ValueError('Unapproved model publisher/repository')
    if not re.fullmatch(r'[a-f0-9]{40}',args.revision):raise ValueError('Pin an exact publisher commit')
    target=Path(args.target);target.mkdir(parents=True,exist_ok=True)
    runtime=Path(args.runtime);runtime.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{args.repo}/revision/{args.revision}?blobs=true',timeout=45) as response:
        meta=json.load(response)
    assert meta['sha']==args.revision
    entries=[e for e in meta['siblings'] if e['rfilename']!='.gitattributes' and
             (Path(e['rfilename']).suffix in {'.json','.txt','.safetensors','.md'} or e['rfilename'].startswith('LICENSE'))]
    jobs=[];fds={};lock=threading.Lock();progress={};done=set();started=time.monotonic()
    journal=runtime/'ranges-complete.json'
    if journal.exists():done=set(json.loads(journal.read_text()))
    small=[];total=0
    for entry in entries:
        name=entry['rfilename']
        if Path(name).name!=name:raise ValueError('Unexpected nested model file')
        size=entry.get('lfs',{}).get('size',entry.get('size',0));total+=size
        if 'lfs' not in entry:small.append(entry);continue
        final=target/name
        if final.exists() and final.stat().st_size==size and sha256(final)==entry['lfs']['sha256']:
            progress[name]=size;continue
        part=runtime/(name+'.partial');part.touch(exist_ok=True);fds[name]=os.open(part,os.O_RDWR)
        for start in range(0,size,32*1024*1024):
            end=min(size,start+32*1024*1024)-1;key=name+':'+str(start)
            if key in done and part.stat().st_size>=end+1:progress[key]=end-start+1
            else:jobs.append((name,start,end,size,key))
    def chunk(job):
        name,start,end,size,key=job
        for attempt in range(6):
            try:
                url=f'https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}?download=true&part={start}&attempt={attempt}'
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                        raise ValueError('Publisher returned a mismatched range')
                    offset=start
                    while offset<=end:
                        block=response.read(min(1024*1024,end-offset+1))
                        if not block:raise IOError('Incomplete public model range')
                        written=0
                        while written<len(block):written+=os.pwrite(fds[name],block[written:],offset+written)
                        offset+=len(block)
                        with lock:progress[key]=offset-start
                with lock:done.add(key)
                return
            except Exception:
                if attempt==5:raise
                time.sleep(2*(attempt+1))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(chunk,job) for job in jobs]
            while not all(f.done() for f in futures):
                with lock:
                    report={'downloaded_bytes':sum(progress.values()),'total_bytes':total,
                        'completed_ranges':len(done),'elapsed_sec':round(time.monotonic()-started)}
                    journal.write_text(json.dumps(sorted(done)))
                (runtime/'progress.json').write_text(json.dumps(report));print(json.dumps(report),flush=True)
                time.sleep(15)
            for future in futures:future.result()
    finally:
        for fd in fds.values():os.fsync(fd);os.close(fd)
        journal.write_text(json.dumps(sorted(done)))
    for entry in entries:
        name=entry['rfilename'];final=target/name
        if name in fds:
            part=runtime/(name+'.partial')
            if sha256(part)!=entry['lfs']['sha256']:raise ValueError('Publisher weight hash mismatch')
            part.replace(final)
        elif entry in small:
            with urllib.request.urlopen(f'https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}',timeout=45) as response:
                content=response.read()
            if hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()!=entry['blobId']:
                raise ValueError('Publisher configuration blob mismatch')
            final.write_bytes(content)
    files=[{'file':e['rfilename'],'bytes':(target/e['rfilename']).stat().st_size,
            'sha256':sha256(target/e['rfilename'])} for e in entries]
    result={'complete':True,'repo':args.repo,'revision':args.revision,'publisher_hashes_verified':True,
        'files':files,'elapsed_sec':round(time.monotonic()-started)}
    (target/'setup-manifest.json').write_text(json.dumps(result,indent=2))
    Path(args.report).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--target',required=True);p.add_argument('--runtime',required=True);p.add_argument('--report',required=True)
    p.add_argument('--workers',type=int,choices=range(1,33),default=8)
    download(p.parse_args())
