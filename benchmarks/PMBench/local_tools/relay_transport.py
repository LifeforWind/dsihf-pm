"""GPT-only user relay, with immutable legacy cache provenance and no fallback."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
from qwen_wire import MODEL as EXPECTED, BASE, KEY, wire as qwen_wire
from run_official_baselines import (AuditedTransport, MODEL, PROVIDER, LIMITS,
                                    StudyStop, append, digest, usage, write)


def validate_base(url):
    if url.rstrip('/') != BASE:
        raise StudyStop('Expected server-local Qwen endpoint; no fallback')
    return BASE


def configure():
    os.environ.update(OPENAI_API_KEY=KEY, OPENAI_BASE_URL=BASE, PMBENCH_MODEL=MODEL,
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ANONYMIZED_TELEMETRY='False')
    os.environ.pop('OPENROUTER_API_KEY', None)
    if os.environ.get('MATCHED_OFFLINE') == '1':
        import socket
        def denied(*args, **kwargs):
            raise StudyStop('Offline audit forbids network connections')
        socket.socket.connect = denied


def make_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                  base_url=validate_base(os.environ['OPENAI_BASE_URL']),
                  timeout=600, max_retries=0)


def canonical_request(original):
    wire = deepcopy(original)
    wire.setdefault('extra_body', {})['provider'] = deepcopy(PROVIDER)
    return wire


def relay_request(original):
    wire = deepcopy(original)
    wire['model'] = EXPECTED
    wire.get('extra_body', {}).pop('provider', None)
    if wire.get('extra_body') == {}:
        wire.pop('extra_body')
    try:
        return qwen_wire(wire)
    except AssertionError as exc:
        raise StudyStop('Qwen request adaptation refused') from exc


class RelayTransport(AuditedTransport):
    def __init__(self, client, out, cache, budget=None):
        out, cache = Path(out), Path(cache)
        out.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        super().__init__(client, out, cache)
        self.budget = budget
        self.base = validate_base(str(client.base_url))

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        from openai.types.chat import ChatCompletion
        if kwargs['model'].split('/')[-1] != EXPECTED:
            raise StudyStop('Unexpected requested model')
        original = deepcopy(kwargs)
        original['model'] = MODEL
        if self.budget is not None:
            original = self.budget.request(original)
        canonical = canonical_request(original)
        wire = relay_request(original)
        request_key = digest(canonical)
        with self.lock:
            self.calls += 1
            ordinal = self.calls
            occurrence = self.occurrences.get(request_key, 0) + 1
            self.occurrences[request_key] = occurrence
            if any(self.physical[k] >= v for k, v in LIMITS.items()):
                raise StudyStop('Protection budget reached; condition incomplete')
        key = digest([canonical, occurrence])
        path = self.cache / (key + '.json')
        hit = path.exists()
        if hit:
            record = json.loads(path.read_text())
            assert record['official_request'] == original
            assert record.get('canonical_request', record['wire_request']) == canonical
            assert record['identity_accepted']
            response = ChatCompletion.model_validate(record['response'])
            with self.lock:
                self.hits += 1
        else:
            if os.environ.get('MATCHED_OFFLINE') == '1':
                raise StudyStop('Offline exact request cache miss; network prohibited')
            stop = os.environ.get('RELAY_STOP_FILE')
            if stop and Path(stop).exists():
                raise StudyStop('Study circuit breaker prevents further calls')
            for attempt in range(4):
                if stop and Path(stop).exists():
                    raise StudyStop('Study circuit breaker prevents further calls')
                started = time.monotonic()
                record = dict(call=ordinal, attempt=attempt+1, cache_key=key, occurrence=occurrence,
                              official_request=original, canonical_request=canonical,
                              wire_request=wire, api_base_url=self.base, route='local-qwen')
                try:
                    response = self.client.chat.completions.create(**wire)
                except Exception as exc:
                    status = getattr(exc, 'status_code', None)
                    record.update(error_type=type(exc).__name__, status_code=status,
                                  error_body=getattr(exc, 'body', None), seconds=time.monotonic()-started)
                    with self.lock:
                        append(self.out/'api_errors.jsonl', record)
                    if status in (401, 402, 403) and stop:
                        # One account/routing failure must not drain the whole queue.
                        Path(stop).write_text(json.dumps(dict(status_code=status, route='local-qwen')))
                    retryable = status in (429,500,502,503,504,520,522,524) or type(exc).__name__ in (
                        'APIConnectionError','APITimeoutError')
                    if not retryable or attempt == 3:
                        raise StudyStop('Recorded relay API failure: '+type(exc).__name__) from exc
                    time.sleep(min(2**attempt, 4))
                    continue
                data = response.model_dump(mode='json')
                accepted = data['model'].split('/')[-1] == EXPECTED
                record.update(response=data, identity_accepted=accepted, seconds=time.monotonic()-started)
                with self.lock:
                    append(self.out/'model_calls.jsonl', record)
                    for k, v in usage([record]).items():
                        self.physical[k] += v
                if not accepted:
                    raise StudyStop("Wrong Qwen model identity; no resampling")
                write(path, record)
                break
            else:
                raise StudyStop('No accepted relay response')
        if response.model.split('/')[-1] != EXPECTED:
            raise StudyStop('Cached model identity mismatch')
        with self.lock:
            append(self.out/'requests.jsonl', dict(call=ordinal, cache_key=key, occurrence=occurrence,
                   cache_hit=hit, official_request=original, canonical_request=canonical,
                   wire_request=record['wire_request'], route=record.get('route','legacy-openrouter-cache')))
            for k,v in usage([record]).items():
                self.used[k] += v
            write(self.out/'usage.json', dict(logical=self.used, physical=self.physical,
                                            cache_hits=self.hits, calls=self.calls))
        return response.model_copy(update={'model': EXPECTED})
