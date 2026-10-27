"""Shared route guard retained for regression coverage."""
from openmodel_config import MODEL as EXPECTED

def check_route(url, payload, relay):
    if payload.get('model','').startswith('typesafe/jev-'):
        assert url=='https://openrouter.ai/api/alpha/decisions', 'Unexpected Jev endpoint'
        return 'jev-openrouter'
    assert url==relay+'/chat/completions', 'Non-Jev request must use configured relay'
    assert payload['model']==EXPECTED, 'Unexpected non-Jev model'
    return 'user-relay'
