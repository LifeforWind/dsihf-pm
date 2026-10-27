"""Conservative syntax repair before unchanged lifecycle semantic verification."""
from copy import deepcopy
import hashlib,json,re
from run_intentions import IntentionStore,COMPILER_PROMPT,PROPOSALS_SCHEMA,append,write_json


def source_span(quote,text):
    if quote and quote in text:return quote
    if not quote.strip():return None
    matches=list(re.finditer(re.escape(quote),text,re.IGNORECASE))
    if len(matches)!=1:return None
    return text[matches[0].start():matches[0].end()]


def repair_proposal(proposal,text):
    proposal=deepcopy(proposal);changes=[]
    old=proposal['source_quote'];actual=source_span(old,text)
    if actual is not None and actual!=old:
        proposal['source_quote']=actual
        changes.append([proposal['id'],'source_quote',old,actual,'unique case-insensitive span restored to exact observed bytes'])
    spec=proposal['spec']
    clock=re.fullmatch(r'The clock reaches ([0-2][0-9]:[0-5][0-9])[.!]?',spec['trigger'],re.I)
    if (clock and int(clock[1][:2])<24 and spec['kind']=='event' and actual is not None
            and re.findall(r'\b\d{2}:\d{2}\b',actual)==[clock[1]]):
        spec.update(kind='clock',at=clock[1],channel='clock',comparator=None,threshold=None)
        changes.append([proposal['id'],'kind','event','clock','explicit absolute-clock predicate and quoted time agree'])
    return proposal,changes


class SourceCompiler:
    def __init__(self, models, cache):
        self.models, self.cache, self.draft = models, cache, IntentionStore()
        cache.mkdir(parents=True, exist_ok=True)
        self.hits = self.misses = 0

    def compile(self, evidence_id, text, day, step, channels):
        call_start = len(getattr(self.models, 'records', []))
        self.draft.observe(evidence_id, text, day, step)
        state = dict(observation=text, today=day, channels=channels,
                     existing=[dict(id=i.id, status=i.status, **i.spec()) for i in self.draft.intents.values()])
        messages = [dict(role='system', content=COMPILER_PROMPT), dict(role='user', content=json.dumps(state))]
        key = hashlib.sha256(json.dumps([self.models.model, messages, PROPOSALS_SCHEMA], sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + '.json')
        cache_hit = path.exists()
        if cache_hit:
            cached = json.loads(path.read_text())
            proposals = deepcopy(cached['result']['proposals'])
            self.hits += 1
        else:
            result = self.models.chat('compile', messages, PROPOSALS_SCHEMA, 5500)
            proposals = deepcopy(result['proposals'])
            cached = dict(result=result, evidence_id=evidence_id, request=messages)
            self.misses += 1
        trial = deepcopy(self.draft)
        normalizations, rejected, valid = [], [], []
        aliases = {}
        for i, proposal in enumerate(proposals):
            if proposal['op'] == 'add':
                old = proposal['id']
                proposal['id'] = f'{evidence_id}_intent_{i}'
                aliases.setdefault(old, []).append(proposal['id'])
                normalizations.append([old, proposal['id'], 'allocate immutable ID'])
        for proposal in proposals:
            try:
                proposal, repairs = repair_proposal(proposal, text)
                normalizations.extend(repairs)
                for field, value in proposal['spec'].items():
                    if field not in ('action', 'trigger', 'day', 'kind') and value == 'null':
                        proposal['spec'][field] = None
                        normalizations.append([proposal['id'], field, 'string-null to JSON null'])
                parent = proposal['spec']['parent_id']
                if parent in aliases and parent not in trial.intents:
                    if len(aliases[parent]) != 1:
                        raise ValueError('Ambiguous dependency on a new intention')
                    proposal['spec']['parent_id'] = aliases[parent][0]
                trial.validate(proposal, channels)
                trial.apply(proposal, evidence_id)
                valid.append(proposal)
            except (ValueError, KeyError, TypeError) as exc:
                # Unsupported extraction is a measured compiler error, not permission
                # to invent a source, use hidden labels, or discard the entire episode.
                rejected.append(dict(proposal=proposal, reason=str(exc)))
        self.draft = trial
        if not path.exists():
            cached['model_records'] = getattr(self.models, 'records', [])[call_start:]
            write_json(path, cached)
        append(self.models.out / 'compilation.jsonl', dict(evidence_id=evidence_id, cache_key=key,
               proposals=valid, rejected=rejected, cache_hit=cache_hit, normalizations=normalizations))
        return valid
