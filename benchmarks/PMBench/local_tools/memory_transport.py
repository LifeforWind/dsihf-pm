"""User-relay embeddings and fixed local chat profile; exact prefix replay.

No semantic retries, fallback providers, or swallowed network failures.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
from openmodel_config import MODEL
EMBED_MODEL = 'text-embedding-3-small'


class ExperimentStop(BaseException):
    """Bypass upstream handlers which would silently degrade memory."""


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def rows(path):
    path = Path(path)
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def relay_config():
    from dotenv import dotenv_values
    values = dotenv_values(WORKSPACE / 'a-mem/.env')
    base = values['OPENAI_BASE_URL'].rstrip('/')
    parsed = urlsplit(base)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or 'openrouter' in parsed.hostname.lower()):
        raise ExperimentStop('Configured user HTTPS relay required')
    os.environ.pop('OPENROUTER_API_KEY', None)
    os.environ.pop('OPENAI_API_BASE', None)
    os.environ.update(OPENAI_BASE_URL=base, OPENAI_API_KEY=values['OPENAI_API_KEY'],
                      MEM0_TELEMETRY='False', POSTHOG_DISABLED='1')
    return base, values['OPENAI_API_KEY']


class Transport:
    def __init__(self, out, cache, offline=False):
        from openai import OpenAI
        import httpx
        from embedding_redirect import EmbeddingRedirectTransport
        self.out, self.cache = Path(out), Path(cache)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.embedding_base, key = relay_config()
        from qwen_wire import BASE, KEY
        self.base = BASE
        self.offline = offline
        self.client = None if offline else OpenAI(base_url=self.base, api_key=KEY, timeout=600, max_retries=0)
        self.embedding_router = None if offline else EmbeddingRedirectTransport()
        self.embedding_client = None if offline else OpenAI(base_url=self.embedding_base, api_key=key, timeout=60, max_retries=0,
            http_client=httpx.Client(transport=self.embedding_router,follow_redirects=False))
        self.occurrences = Counter()
        self.stage = 'unspecified'
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.embeddings = SimpleNamespace(create=self.embed)
        self.physical = self.logical = self.request_chars = 0
        self.trace_path = self.cache.parent / 'request_tape.jsonl'
        self.trace = rows(self.trace_path)

    @contextmanager
    def role(self, stage):
        previous = self.stage
        self.stage = stage
        try:
            yield
        finally:
            self.stage = previous

    def create(self, **kwargs):
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(self.request('chat', kwargs))

    def embed(self, **kwargs):
        from openai.types import CreateEmbeddingResponse
        return CreateEmbeddingResponse.model_validate(self.request('embedding', kwargs))

    def request(self, kind, request):
        from qwen_wire import wire
        try:
            request = wire(request) if kind == 'chat' else deepcopy(request)
        except AssertionError as exc:
            raise ExperimentStop('Qwen request adaptation refused') from exc
        base = self.base if kind == 'chat' else self.embedding_base
        expected = MODEL if kind == 'chat' else EMBED_MODEL
        if request.get('model') != expected:
            raise ExperimentStop('Unexpected model request')
        legacy_hit = False
        if kind == 'embedding':
            request.setdefault('dimensions', 1536)
            request.setdefault('encoding_format', 'float')
            if request['dimensions'] != 1536 or request['encoding_format'] != 'float':
                raise ExperimentStop('Embedding shape/encoding changed')
        initial = digest([kind, self.stage, request, base])
        self.occurrences[initial] += 1
        key = digest([initial, self.occurrences[initial]])
        marker = dict(kind=kind, stage=self.stage, cache_key=key)
        if self.logical < len(self.trace):
            if self.trace[self.logical] != marker:
                raise ExperimentStop('Recovery request prefix differs before API access')
        else:
            if self.offline:
                raise ExperimentStop('Offline request sequence tape exhausted')
            append(self.trace_path, marker)
            self.trace.append(marker)
        path = self.cache / (key + '.json')
        hit = path.exists()
        if hit:
            record = json.loads(path.read_text())
            # Historical length-ended responses retain their original rejection flag.
            # Deliver the exact response to the native parser; never regenerate it.
            historical_length = (kind == 'chat' and record.get('accepted') is False
                and record.get('response', {}).get('model', '').split('/')[-1] == MODEL
                and bool(record['response'].get('choices'))
                and all(c.get('finish_reason') == 'length' for c in record['response']['choices']))
            if (record['request'] != request or record['api_base_url'] != base
                    or record['kind'] != kind or not (record['accepted'] or historical_length)):
                raise ExperimentStop('Exact cache provenance mismatch')
        else:
            if legacy_hit:
                raise ExperimentStop('Legacy route must never call the network')
            if self.offline:
                raise ExperimentStop('Offline exact request cache miss: ' + key)
            # These are failure/cost protections, never an observation budget.
            prior = list(self.cache.glob('*.json'))
            if len(prior) >= 2500 or self.request_chars + len(json.dumps(request)) > 40000000:
                raise ExperimentStop('Per-trajectory API/input protection reached')
            stop = os.environ.get('MEMORY_BASELINES_STOP')
            for attempt in range(4):
                if stop and Path(stop).exists():
                    raise ExperimentStop('Study circuit breaker is open')
                start = time.monotonic()
                try:
                    fn = self.client.chat.completions.create if kind == 'chat' else self.embedding_client.embeddings.create
                    response = fn(**request)
                except Exception as exc:
                    status = getattr(exc, 'status_code', None)
                    append(self.out / 'api_errors.jsonl', dict(kind=kind, stage=self.stage,
                        cache_key=key, request=request, attempt=attempt + 1, status_code=status,
                        error_type=type(exc).__name__, seconds=time.monotonic() - start))
                    if status in (401, 402, 403) and stop:
                        write(stop, dict(reason='relay authorization/quota', status_code=status))
                    retryable = status in (408, 429, 500, 502, 503, 504, 520, 522, 524) or type(exc).__name__ in ('APIConnectionError', 'APITimeoutError')
                    if not retryable or attempt == 3:
                        raise ExperimentStop('Relay failure: ' + type(exc).__name__) from exc
                    time.sleep(min(2 ** attempt, 4))
                    continue
                data = response.model_dump(mode='json')
                accepted = data.get('model', '').split('/')[-1] == expected
                # Native PMA/Mem0/action parsers own finish/content handling.
                if kind == 'embedding':
                    import math
                    items = data.get('data', [])
                    count = 1 if isinstance(request['input'], str) else len(request['input'])
                    accepted = accepted and len(items) == count and sorted(x['index'] for x in items) == list(range(count))
                    accepted = accepted and all(len(x['embedding']) == 1536 and all(math.isfinite(v) for v in x['embedding']) for x in items)
                record = dict(kind=kind, stage=self.stage, request=request, response=data,
                    cache_key=key, accepted=accepted, api_base_url=base,
                    credential_source=('server-local endpoint; no credential' if kind == 'chat' else 'a-mem/.env:OPENAI_API_KEY'), attempt=attempt + 1,
                    seconds=time.monotonic() - start)
                if kind=='embedding' and getattr(self,'embedding_router',None) is not None:
                    record['http_route']=deepcopy(self.embedding_router.last_route)
                append(self.out / 'model_calls.jsonl', record)
                self.physical += 1
                if not accepted:
                    raise ExperimentStop('Model identity or embedding shape mismatch')
                write(path, record)
                break
        self.logical += 1
        self.request_chars += len(json.dumps(request))
        append(self.out / 'requests.jsonl', dict(kind=kind, stage=self.stage, cache_key=key, cache_hit=hit))
        write(self.out / 'usage.json', dict(logical_calls=self.logical, physical_responses=self.physical,
            request_chars=self.request_chars, billing='Provider-returned usage only; missing costs are unknown'))
        return deepcopy(record['response'])


class NativeTape:
    """Record actual random IDs and wall-clock values, replay without resampling."""
    def __init__(self, path, offline=False):
        self.path, self.offline = Path(path), offline
        self.values = rows(self.path)
        self.position = 0
        self.lock = threading.Lock()

    def get(self, label, generate):
        with self.lock:
            if self.position == len(self.values):
                if self.offline:
                    raise ExperimentStop('Native replay tape exhausted: ' + label)
                item = dict(label=label, value=generate())
                append(self.path, item)
                self.values.append(item)
            item = self.values[self.position]
            if item['label'] != label:
                raise ExperimentStop('Native replay tape order mismatch')
            self.position += 1
            return item['value']

    def uuid_module(self, label):
        return SimpleNamespace(uuid4=lambda: uuid.UUID(self.get(label + '.uuid4', lambda: uuid.uuid4().hex)))

    def datetime_class(self, label):
        tape = self
        class TapedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                value = tape.get(label + '.now.' + str(tz), lambda: datetime.now(tz).isoformat())
                return datetime.fromisoformat(value)
        return TapedDateTime
