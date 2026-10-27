"""Shared I/O for the released author PIS adapter; no policy implementation."""
import hashlib, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'sim'))
import pm_bench as pm
from openmodel_config import MODEL
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def rows(path):
    path=Path(path)
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
