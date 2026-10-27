"""Conservative syntax repair before unchanged lifecycle semantic verification."""
from copy import deepcopy
import hashlib,json,re
from run_intentions import IntentionStore,COMPILER_PROMPT,PROPOSALS_SCHEMA,append,write_json
from compiler_failure import TruncatedCompilation, validate_failure_cache, digest

ELLIPSIS = re.compile(r'\.{3,}|\u2026+')


def anchored_span(groups,text):
    """Return a span only when all proposed anchor layouts identify one place."""
    spans = set()
    boundaries = [0, *[m.end() for m in re.finditer(r'[.!?](?=\s|$)|[;\r\n]',text)], len(text)]
    for anchors in groups:
        if len(anchors)<2 or any(not part or not re.search(r'\w',part) for part in anchors):
            continue
        # Lookahead retains overlapping occurrences, which also count as ambiguity.
        patterns = [re.compile(r'(?=('+re.escape(part)+r'))',re.IGNORECASE) for part in anchors]
        for start,end in zip(boundaries,boundaries[1:]):
            block = text[start:end]
            for first in patterns[0].finditer(block):
                position = first.end(1)
                for pattern in patterns[1:-1]:
                    match = pattern.search(block,position)
                    if match is None:
                        break
                    position = match.end(1)
                else:
                    for last in patterns[-1].finditer(block,position):
                        spans.add((start+first.start(1),start+last.end(1)))
                        if len(spans)>1:
                            return None
    if not spans:
        return None
    start,end = spans.pop()
    return text[start:end]


def ellipsis_span(quote,text):
    """Restore one unique, contiguous observed span from literal quote anchors.

    Gaps stay within one source line and sentence/clause. This only restores
    quotation syntax; action/trigger/scope still undergo semantic verification.
    """
    anchors = [part.strip() for part in ELLIPSIS.split(quote)]
    if len(anchors)<2 or any(not part or not re.search(r'\w',part) for part in anchors):
        return None
    return anchored_span([anchors],text)


def omitted_span(quote,text):
    """Restore one implicit gap, keeping a literal prefix of at least two words.

    Every quoted character stays in order in two literal anchors. No words are
    substituted or reordered, and all possible gap positions must agree.
    """
    words = list(re.finditer(r'\S+',quote))
    if len(words)<4:
        return None
    groups = [[quote[:words[i-1].end()].strip(),quote[words[i].start():].strip()]
              for i in range(2,len(words))]
    return anchored_span(groups,text)


def source_span(quote,text):
    if quote and quote in text:return quote
    if not quote.strip():return None
    matches=list(re.finditer(re.escape(quote),text,re.IGNORECASE))
    if len(matches)==1:return text[matches[0].start():matches[0].end()]
    if matches:return None
    return ellipsis_span(quote,text) if ELLIPSIS.search(quote) else omitted_span(quote,text)


def repair_proposal(proposal,text):
    proposal=deepcopy(proposal);changes=[]
    old=proposal['source_quote'];actual=source_span(old,text)
    if actual is not None and actual!=old:
        proposal['source_quote']=actual
        reason = ('unique case-insensitive span restored to exact observed bytes'
                  if re.fullmatch(re.escape(old),actual,re.IGNORECASE) else
                  'unique same-line ellipsis expansion restored to exact observed bytes'
                  if ELLIPSIS.search(old) else
                  'unique same-line omitted span restored to exact observed bytes')
        changes.append([proposal['id'],'source_quote',old,actual,reason])
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

    def reject_truncated(self,failure,previous,evidence_id,key,cache_hit):
        self.draft=previous
        event=dict(evidence_id=evidence_id,operation='reject_entire_truncated_compile_preserve_store',
                   chat_cache_key=failure.key,**failure.entry['terminal_failure'],
                   preserved_store_sha256=digest(previous.dump()),parsed_proposals=0)
        append(self.models.out/'compiler_failures.jsonl',event)
        append(self.models.out/'compilation.jsonl',dict(evidence_id=evidence_id,cache_key=key,
               proposals=[],rejected=[],cache_hit=cache_hit,normalizations=[],failure=event))
        return []

    def compile(self, evidence_id, text, day, step, channels):
        call_start = len(getattr(self.models, 'records', []))
        previous=deepcopy(self.draft)
        self.draft.observe(evidence_id, text, day, step)
        state = dict(observation=text, today=day, channels=channels,
                     existing=[dict(id=i.id, status=i.status, **i.spec()) for i in self.draft.intents.values()])
        messages = [dict(role='system', content=COMPILER_PROMPT), dict(role='user', content=json.dumps(state))]
        key = hashlib.sha256(json.dumps([self.models.model, messages, PROPOSALS_SCHEMA], sort_keys=True).encode()).hexdigest()
        path = self.cache / (key + '.json')
        cache_hit = path.exists()
        if cache_hit:
            cached = json.loads(path.read_text())
            assert cached['request']==messages
            if 'terminal_failure' in cached:
                assert 'result' not in cached
                entry=dict(request=cached['failure_chat_request'],terminal_failure=cached['terminal_failure'],records=cached['model_records'])
                validate_failure_cache(entry)
                assert entry['request']['messages']==messages and entry['request']['model']==self.models.model
                assert entry['request']['schema']==PROPOSALS_SCHEMA and entry['request']['max_tokens']==5500
                failure=TruncatedCompilation(cached['failure_chat_key'],entry)
                self.hits+=1
                return self.reject_truncated(failure,previous,evidence_id,key,True)
            proposals = deepcopy(cached['result']['proposals'])
            self.hits += 1
        else:
            try:
                result = self.models.chat('compile', messages, PROPOSALS_SCHEMA, 5500)
            except TruncatedCompilation as failure:
                self.misses+=1
                entry=failure.entry
                cached=dict(request=messages,evidence_id=evidence_id,
                            terminal_failure=entry['terminal_failure'],failure_chat_key=failure.key,
                            failure_chat_request=entry['request'],model_records=entry['records'])
                write_json(path,cached)
                return self.reject_truncated(failure,previous,evidence_id,key,False)
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
