"""Transport-only Qwen adaptation; no method or prompt rewriting."""
from copy import deepcopy
from openmodel_config import MODEL
from openmodel_config import BASE, TEMPLATE, KEY, PROFILE
def wire(request):
    request = deepcopy(request)
    assert request['model'].split('/')[-1] == MODEL
    request['model'] = MODEL
    extra = request.setdefault('extra_body', {})
    extra.pop('provider', None)
    if PROFILE != 'relay':
        extra['chat_template_kwargs'] = deepcopy(TEMPLATE)
    elif not extra:
        request.pop('extra_body', None)
    if PROFILE == 'relay':
        return request
    messages = request['messages']
    systems = []
    while messages and messages[0]['role'] == 'system':
        systems.append(messages.pop(0)['content'])
    assert not any(m['role'] == 'system' for m in messages), 'Nonleading system requires review'
    if systems:
        messages.insert(0, {'role': 'system', 'content': '\n\n'.join(systems)})
    return request
