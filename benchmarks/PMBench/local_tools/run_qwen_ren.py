"""Explicitly authorized Qwen compiler + original Jev and REN policy."""
import json, os, sys, urllib.request
from pathlib import Path
from unittest.mock import patch
from dotenv import dotenv_values
from openai.resources.chat.completions import Completions
import run_ren_confirmation as frozen
from qwen_transport import BASE_URL, MODEL, TRANSPORT, check_route, map_disconnect
from openmodel_config import KEY, PROFILE
from run_recommendation import WORKSPACE, append, write_json


def main():
    os.environ['OPENAI_API_KEY'] = KEY
    os.environ['OPENAI_BASE_URL'] = BASE_URL
    os.environ['OPENROUTER_API_KEY'] = dotenv_values(WORKSPACE/'a-mem/.env')['OPENROUTER_API_KEY']
    assert '--model' not in sys.argv and '--base-url' not in sys.argv
    sys.argv += ['--model', MODEL, '--base-url', BASE_URL]
    out = Path(sys.argv[sys.argv.index('--out')+1])
    create, urlopen = Completions.create, map_disconnect(urllib.request.urlopen)
    original_main = frozen.base.main
    def configured_main(**kwargs):
        kwargs['llm_timeout'] = 600
        kwargs['extra_sources'] += [Path(__file__), Path(__file__).with_name('qwen_transport.py')]
        kwargs['extra_manifest'].update(thinking='disabled', transport=TRANSPORT,
            endpoint_authorization='Explicit release profile: '+PROFILE,
            provider_documented_version=MODEL+'; deployment configured by user',
            speculative_decoding=json.loads(os.environ.get('PMBENCH_SPECULATIVE_CONFIG','null')), execution_host='user-configured', model_precision='deployment-defined', compiler_timeout_seconds=600,
            json_schema_validation='Identical schema constrained by native strict response_format and supplied in system instruction; local validation; no semantic retry')
        return original_main(**kwargs)
    def before(url, payload):
        stop = os.environ.get('QWEN_STOP_FILE')
        if stop and Path(stop).exists():
            raise RuntimeError('Experiment stop file present')
        route = check_route(url, payload)
        append(out/'route_events.jsonl', dict(url=url, model=payload['model'], route=route,
            thinking=payload.get('extra_body', {}).get('chat_template_kwargs')))
    def guarded_create(resource, *args, **kwargs):
        assert resource._client.api_key == KEY, 'Unexpected credential source'
        before(str(resource._client.base_url).rstrip('/')+'/chat/completions', kwargs)
        return create(resource, *args, **kwargs)
    def guarded_urlopen(request, *args, **kwargs):
        before(request.full_url, json.loads(request.data))
        return urlopen(request, *args, **kwargs)
    try:
        with patch.object(Completions,'create',guarded_create), patch.object(urllib.request,'urlopen',guarded_urlopen), patch.object(frozen.base,'main',configured_main):
            frozen.main()
    except Exception as exc:
        name = type(exc).__name__
        transport = name in ('APIConnectionError','APITimeoutError','RateLimitError','InternalServerError','URLError','TimeoutError','RemoteDisconnected') or str(exc).startswith(
            ('Decisions transport failed','Decisions HTTP 429','Decisions HTTP 500','Decisions HTTP 502','Decisions HTTP 503','Decisions HTTP 504'))
        if out.exists():
            write_json(out/'failure_kind.json',dict(error_type=name,retryable_transport=transport))
        # Do not print SDK exceptions, which can contain request details.
        print('Run failed: '+name, file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
