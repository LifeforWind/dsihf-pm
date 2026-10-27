"""Portable, explicit chat profiles. Relay credentials live in a-mem/.env."""
import os
from pathlib import Path
from urllib.parse import urlsplit
from dotenv import dotenv_values
WORKSPACE=Path(__file__).resolve().parents[3]
VALUES=dotenv_values(WORKSPACE/'a-mem/.env')
PROFILES={
    'qwen4b':dict(model='Qwen3.5-4B',repo='Qwen/Qwen3.5-4B',port=8002,template={'enable_thinking':False}),
    'gemma26b':dict(model='gemma-4-26B-A4B-it',repo='google/gemma-4-26B-A4B-it',port=8003,template={'enable_thinking':False}),
}
PROFILE=os.environ.get('PMBENCH_PROFILE','relay')
if PROFILE=='relay':
    MODEL=os.environ.get('PMBENCH_MODEL','gpt-4o-mini-2024-07-18')
    BASE=(VALUES.get('OPENAI_BASE_URL') or 'https://relay.example.invalid/v1').rstrip('/')
    KEY=VALUES.get('OPENAI_API_KEY') or 'missing-configure-a-mem-env'
    TEMPLATE={};CONFIG=dict(model=MODEL,template=TEMPLATE)
    parsed=urlsplit(BASE)
    if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or 'openrouter' in parsed.hostname.lower():
        raise RuntimeError('Non-Jev models require the configured HTTPS relay; no fallback')
else:
    if PROFILE not in PROFILES:raise RuntimeError('Unknown explicit model profile')
    CONFIG=PROFILES[PROFILE];MODEL=CONFIG['model'];TEMPLATE=CONFIG['template']
    BASE=f"http://127.0.0.1:{CONFIG['port']}/v1";KEY='local-no-secret'
