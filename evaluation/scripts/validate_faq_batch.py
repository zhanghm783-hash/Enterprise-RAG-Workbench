"""校验固定FAQ回答；仅将引用中的省略号展开回同一源文的真实连续片段。"""
from __future__ import annotations

import json
import re
from generate_grounded_answers import ROOT,digest,write,parse_grounded,visible_markdown


def expand_excerpt(quote,source):
    visible=visible_markdown(source)
    if visible_markdown(quote) in visible:return quote,False
    parts=[visible_markdown(x) for x in re.split(r'(?:…+|\.{3,})',quote)]
    if len(parts)<2 or any(len(x)<8 for x in parts):raise ValueError('Quotation cannot be recovered from literal segments')
    start=None;cursor=0
    for part in parts:
        position=visible.find(part,cursor)
        if position<0:raise ValueError('Quoted segment missing/out of order')
        if start is None:start=position
        cursor=position+len(part)
    recovered=visible[start:cursor]
    if len(recovered)>280:raise ValueError('Recovered quote too long')
    return recovered,True


def main():
    source=ROOT/'evaluation/artifacts/answer_quality_faq_v1';dest=ROOT/'evaluation/artifacts/answer_quality_faq_v1_validated'
    if (dest/'ready.json').exists():
        ready=json.loads((dest/'ready.json').read_text(encoding='utf-8'));assert digest(dest/'answers.jsonl')==ready['answers_sha256']
        print('FAQ validated answers already ready; no model calls.');return
    rows=[json.loads(x) for x in (source/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=48 or len({r['sample_id'] for r in rows})!=48:raise ValueError('Incomplete raw batch')
    recovered=[];result=[]
    for original in rows:
        row=dict(original)
        if row['status_code']!=200:raise ValueError('Generation API failure requires separate repair')
        if row['scheme']=='grounded_final':
            obj=json.loads(row['raw_response'].strip())
            for statement in obj['statements']:
                quote,changed=expand_excerpt(statement['quote'],row['context_contents'][statement['source']-1])
                if changed:
                    recovered.append({'sample_id':row['sample_id'],'original_quote':statement['quote'],'expanded_literal_quote':quote})
                    statement['quote']=quote
            obj,answer=parse_grounded(json.dumps(obj,ensure_ascii=False),row,allow_visible_markdown=True)
            row.update(status='ok',response=answer,evidence_statements=obj['statements'],insufficient_evidence=obj['insufficient_evidence'],missing=obj.get('missing',''))
            row.pop('error_type',None);row.pop('error_detail',None)
        if row['status']!='ok':raise ValueError('Unrecovered answer failure')
        result.append(row)
    dest.mkdir(exist_ok=False)
    (dest/'input_jobs.json').write_bytes((source/'input_jobs.json').read_bytes())
    with (dest/'answers.jsonl').open('w',encoding='utf-8') as stream:
        for row in result:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(status='complete',errors=0,parent_raw_answers_sha256=digest(source/'answers.jsonl'),
                    citation_repairs=recovered,quote_repair_policy='Only restore omitted literal source text, same source and order, no answer wording changes',repair_model_calls=0)
    write(dest/'manifest.json',manifest)
    write(dest/'ready.json',{'status':'ready','answers':48,'questions':24,'schemes':['vector','grounded_final'],
                           'answers_sha256':digest(dest/'answers.jsonl'),'manifest_sha256':digest(dest/'manifest.json')})
    print(json.dumps({'status':'ready','answers':48,'literal_excerpt_expansions':len(recovered),'paid_calls':0},ensure_ascii=False))


if __name__=='__main__':main()
