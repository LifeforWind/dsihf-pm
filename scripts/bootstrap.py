#!/usr/bin/env python3
"""Fetch immutable upstreams and materialize only the required benchmark files.

No model calls. No result files are extracted. Existing dirty checkouts are refused.
Use --source-root PATH to reuse a local research workspace without network access.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def git(path, *args, binary=False):
    return subprocess.check_output(['git', '-C', str(path), *args], text=not binary)

def put(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError('Refusing to overwrite different materialized content: '+str(path))
    path.write_bytes(content)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    configs = json.loads((ROOT/'configs/upstreams.json').read_text())
    for name, config in configs.items():
        dest = ROOT/config['destination']
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if args.source_root:
                source = args.source_root.resolve()/'benchmarks'/name
                subprocess.run(['git','clone','--shared','--no-checkout',str(source),str(dest)], check=True)
            else:
                subprocess.run(['git','init',str(dest)], check=True, stdout=subprocess.DEVNULL)
                subprocess.run(['git','-C',str(dest),'remote','add','origin',config['url']], check=True)
                subprocess.run(['git','-C',str(dest),'fetch','--depth','1','origin',config['commit']], check=True)
            subprocess.run(['git','-C',str(dest),'sparse-checkout','set','--no-cone',
                            *['/'+p for p in config['sparse']]], check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--detach',config['commit']],check=True)
        if git(dest,'rev-parse','HEAD').strip()!=config['commit'] or git(dest,'status','--porcelain').strip():
            raise RuntimeError('Existing upstream differs from locked clean revision: '+name)
        print(name, config['commit'])
    upstream = ROOT/configs['PMBench']['destination']
    names = git(upstream,'ls-tree','-r','--name-only','HEAD','sim').splitlines()
    for name in names:
        if name.endswith('.py'):
            put(ROOT/'benchmarks/PMBench'/name,git(upstream,'show','HEAD:'+name,binary=True))
    name='data/synthetic_week_v9.json'
    put(ROOT/'benchmarks/PMBench'/name,git(upstream,'show','HEAD:'+name,binary=True))
    # This is the author's decision parser bundled inside their published archive.
    # Extract exactly one file; no author trajectories, scores, or archive paths.
    archive=git(ROOT/configs['PIS']['destination'],'show','HEAD:data.zip',binary=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        content=z.read('PMBench/sim/pm_bench.py')
    expected='a8917efc2a065981dc13fbec5ffc46966e0eb5b60a00b12766e4e8389b83dbcb'
    if hashlib.sha256(content).hexdigest()!=expected:
        raise RuntimeError('Author decision parser checksum mismatch')
    put(ROOT/'.deps/author_pm_bench.py',content)
    print('Upstreams ready. No model API calls or result files generated.')

if __name__=='__main__':main()
