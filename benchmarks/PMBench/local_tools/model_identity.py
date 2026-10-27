"""Retry the same request only when the returned model identity is wrong."""
import json
from pathlib import Path
import time


def identity_checked(create, expected_model, out, phase, retries=2):
    def call(**kwargs):
        for attempt in range(retries+1):
            start = time.monotonic()
            response = create(**kwargs)
            if response.model == expected_model:
                return response
            event = dict(phase=phase, attempt=attempt+1, request=kwargs,
                         response=response.model_dump(), seconds=time.monotonic()-start,
                         reason='returned_model_mismatch', applied_to_state=False)
            with (Path(out)/'rejected_model_responses.jsonl').open('a') as f:
                f.write(json.dumps(event)+'\n')
        raise RuntimeError('Expected model unavailable after identical-request retries')
    return call
