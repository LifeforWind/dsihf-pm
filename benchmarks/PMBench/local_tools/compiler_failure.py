"""A terminal extraction failure is not a successful empty model answer."""
from copy import deepcopy
import hashlib,json

POLICY='reject-truncated-intention-batch-v1'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def failure_cache_entry(request,records):
    """Preserve the exact incomplete response without parsing any proposals."""
    assert request['stage']=='compile' and records
    last=records[-1];wire=last['request'];response=last['response']
    assert last['stage']=='compile' and last['kind']=='llm'
    assert wire['model']==response['model']==request['model']
    assert wire['max_tokens']==request['max_tokens']
    assert len(response['choices'])==1 and response['choices'][0]['finish_reason']=='length'
    assert isinstance(response['choices'][0]['message']['content'],str)
    failure=dict(policy=POLICY,reason='output_truncated',finish_reason='length',
                 max_tokens=request['max_tokens'],raw_record_sha256=digest(last))
    return dict(request=deepcopy(request),terminal_failure=failure,records=deepcopy(records))

def validate_failure_cache(entry):
    assert 'result' not in entry, 'A failed extraction cannot be a successful result'
    expected=failure_cache_entry(entry['request'],entry['records'])
    assert entry['terminal_failure']==expected['terminal_failure']
    return expected

class TruncatedCompilation(RuntimeError):
    def __init__(self,key,entry):
        self.entry=validate_failure_cache(entry)
        assert key==digest(self.entry['request'])
        self.key=key
        super().__init__('Truncated intention batch rejected; original response retained')
