"""Provider-only adaptation: same schemas, explicit non-thinking, no fallback."""
from copy import deepcopy
import json
import jsonschema

from openmodel_config import MODEL, BASE as BASE_URL, TEMPLATE, PROFILE
TRANSPORT = 'openmodel-native-json-schema-v2'

def payload_for(stage, messages, schema, max_tokens):
    assert stage.startswith('compile'), 'Only compiler calls may use Qwen'
    wire = deepcopy(messages)
    # Enforce the same schema at decoding, as the native mini compiler does.
    # Keep the existing instruction verbatim; never repair or resample responses.
    instruction = 'Return only a JSON object conforming exactly to this JSON Schema:\n' + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    # Qwen's template accepts only one leading system message. Preserve each
    # original system string verbatim, joined after the schema instruction.
    leading = []
    while wire and wire[0]['role'] == 'system':
        leading.append(wire.pop(0)['content'])
    assert not any(m['role'] == 'system' for m in wire), 'Non-leading system message'
    wire.insert(0, dict(role='system', content='\n\n'.join([instruction, *leading])))
    return dict(model=MODEL, messages=wire, temperature=0, max_tokens=max_tokens,
                response_format={'type': 'json_schema', 'json_schema': {
                    'name': 'typed_result', 'strict': True, 'schema': deepcopy(schema)}},
                **({'extra_body': {'chat_template_kwargs': deepcopy(TEMPLATE)}} if PROFILE != 'relay' else {}))

def validate_response(data, schema):
    assert data['model'] == MODEL, 'Unexpected Qwen response identity'
    choice = data['choices'][0]
    assert not choice['message'].get('reasoning_content'), 'Thinking output received'
    assert not choice['message'].get('reasoning'), 'Thinking output received'
    assert '<think>' not in (choice['message'].get('content') or ''), 'Thinking text received'
    details = data.get('usage', {}).get('completion_tokens_details') or {}
    assert not details.get('reasoning_tokens'), 'Reasoning tokens received'
    if choice['finish_reason'] == 'stop':
        jsonschema.validate(json.loads(choice['message']['content']), schema)

def check_route(url, payload):
    if payload.get('model') == 'typesafe/jev-1.13':
        assert url == 'https://openrouter.ai/api/alpha/decisions'
        return 'jev-openrouter'
    assert url == BASE_URL + '/chat/completions'
    assert payload['model'] == MODEL
    assert payload.get('extra_body', {}) == ({'chat_template_kwargs': deepcopy(TEMPLATE)} if PROFILE != 'relay' else {})
    assert 'reasoning_effort' not in payload
    fmt = payload['response_format']
    assert fmt['type'] == 'json_schema'
    assert fmt['json_schema']['name'] == 'typed_result'
    assert fmt['json_schema']['strict'] is True
    assert isinstance(fmt['json_schema']['schema'], dict)
    return 'user-authorized-qwen-gpu0'
