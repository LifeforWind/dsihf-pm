"""Verify that each Jev cache response originated in its physical-call log."""
from run_recommendation import digest
JEV="typesafe/jev-1.13-20260917"

def check_judge(cache,physical):
    request=cache['request'];records=cache['model_records'];assert records
    assert all(digest(r) in physical for r in records),'Unlogged cache provenance'
    last=records[-1];response=last['response']
    assert request['judge']=='jev' and request['jev_version']==JEV
    assert last['kind']=='jev' and last['stage']==request['stage']
    assert last['request']==dict(model='typesafe/jev-1.13',state=request['state'],questions=request['questions'])
    assert response['model']==JEV and response['provider']=='TypeSafe'
    assert response['answers']==cache['answers']
