#!/usr/bin/env python3
"""REN with source-grounded interpretation; demand lifecycle and rotation unchanged."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import sys
from unittest.mock import patch
import run_recommendation as base
from run_ren_grounded import GroundedModels,GroundedRENPolicy
from run_ren_natural import NEEDS_SOURCE,SCENE
from ren_public_frames import normalize,frame_questions
from ren_source_compiler import SourceCompiler

CONTRACT_RULES='''
Decompose ONLY the trigger field, not the day/action/source-history context.
The intention's day and activation window are already enforced by the lifecycle.
Never add a second condition solely to repeat today's day or time-window metadata.
Physical locations, counters, racks, carts, signs and objects seen in a scene use
__visible_scene__; a thematically related process channel does not reveal their
physical availability. Use a hidden source only for the particular state it can
report, not because its name shares the action's topic. Preserve genuine ANDs.
'''

def situated_questions(state,questions):
    questions=frame_questions(state,questions)
    for key,q in questions.items():
        n=int(key.split('_')[-1]);item=state['evidence_items'][n]
        if key.startswith('positive_'):
            q['instructions'] += (' All qualifications matter: availability of one option does not '
                'establish a better/earlier/different option, and a related object does not establish '
                'this object. If a required property is unspecified, do not fill it in.')
            if item['source']==SCENE:
                q['instructions'] += (' This is the user\'s current situated scene. An activity framing '
                    'the current scene indicates the user is engaged in that activity; a described '
                    'nearby counter or location normally establishes being there and noticing it. '
                    'Do not require a separate literal declaration of starting or arriving. '
                    'This does not apply to remotely described scenes, plans, memories, hypotheticals '
                    'or merely mentioned destinations. Use the intention to resolve the specific '
                    'location meant by generic words such as counter, not as evidence of presence.')
    return questions

class SituatedModels(GroundedModels):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.frames={}

    def chat(self,stage,messages,schema,max_tokens):
        if stage=='compile_condition_contracts':
            messages=deepcopy(messages);messages[0]['content']+=CONTRACT_RULES
        return super().chat(stage,messages,schema,max_tokens)

    def evaluate(self,stage,state,questions):
        if stage=='verify_condition_contracts':
            questions=deepcopy(questions)
            for q in questions.values():q['instructions']+=CONTRACT_RULES
        if stage!='interpret_grounded_evidence' or not questions:
            return super().evaluate(stage,state,questions)
        framed=deepcopy(state)
        for row in framed['evidence_items']:
            if row['kind']!='event':continue
            token=(row['source'],row['observed_text'])
            if token not in self.frames:
                self.frames[token]=normalize(self,*token)
                base.append(self.out/'public_frames.jsonl',dict(source=token[0],text=token[1],**self.frames[token]))
            row['normalized_facts']=self.frames[token]['facts']
        revised=situated_questions(framed,questions)
        answers=super().evaluate('interpret_situated_evidence',framed,revised)
        base.append(self.out/'situated_evidence.jsonl',dict(state=framed,questions=revised,answers=answers))
        return answers

def main():
    argv=sys.argv[1:];index=argv.index('--candidate-mode');mode=argv[index+1];del argv[index:index+2]
    args=base.parse_args(argv)
    sources=[Path(__file__),NEEDS_SOURCE]+[Path(__file__).with_name(n) for n in
        ('run_ren_grounded.py','run_ren_natural.py','ren_public_frames.py','ren_source_compiler.py','residual_policy.py','compiler_failure.py')]
    with patch.object(base,'Compiler',SourceCompiler):
        base.main(args=args,policy_factory=partial(GroundedRENPolicy,candidate_mode=mode),model_factory=SituatedModels,
            extra_sources=sources,extra_manifest=dict(ren_adapter='situated-v1',candidate_mode=mode,
                shared_condition_decision=True,natural_language_entry=True,structured_oracle_contracts=False,
                source_span_repair='unique case-insensitive match restored to exact observed span',
                condition_scope='only trigger; date already enforced',observation_frames='quote-grounded public text only',
                compiler_failure_policy='reject-truncated-compiler-batch-v1',
                model_identity_retries=2,evidence_validity='current public step only; inferred by adapter',
                receipt_interface_available=False,completion_feedback_exercised=False))

if __name__=='__main__':main()
