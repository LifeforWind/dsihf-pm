#!/usr/bin/env python3
"""Check tracked release contents without reading or printing credential values."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    names=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z'],text=True).split('\0')
    files=[ROOT/n for n in names if n]
    if not files:raise SystemExit('Stage the release files before checking')
    problems=[];size=0
    forbidden={'runs','results','reports','outputs','cache','.cache','.deps','.venv','__pycache__'}
    for path in files:
        name=path.relative_to(ROOT)
        if forbidden.intersection(name.parts) or path.suffix in {'.jsonl','.log','.sqlite','.db','.zip','.safetensors','.pt','.pth'}:
            problems.append(str(name)+': generated/runtime artifact')
        if path.name.startswith('.env') and path.name!='.env.example':
            problems.append(str(name)+': credential file')
        content=path.read_bytes();size+=len(content)
        if len(content)>2_000_000:problems.append(str(name)+': oversized code-release file')
        if re.search(rb'^-----BEGIN (?:OPENSSH )?PRIVATE KEY-----',content,re.M):
            problems.append(str(name)+': private key material')
        if re.search(rb'\bsk-(?:or-v1-)?[A-Za-z0-9_-]{24,}',content):
            problems.append(str(name)+': token-shaped literal')
        if path.suffix=='.py':ast.parse(content,filename=str(name))
    inventory=json.loads((ROOT/'docs/source-provenance.json').read_text())
    import hashlib
    for row in inventory['sources']:
        path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['release_sha256']:
            problems.append(row['path']+': stale release provenance')
    if problems:raise SystemExit('\n'.join(problems))
    print(json.dumps(dict(status='passed',tracked_files=len(files),bytes=size,credentials_or_results_detected=0)))

if __name__=='__main__':main()
