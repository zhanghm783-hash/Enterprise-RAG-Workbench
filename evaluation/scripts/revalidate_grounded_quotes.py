"""保留原始回答，仅按Markdown显示文本校验引用；不调用模型。"""
from __future__ import annotations

import json
from pathlib import Path

from generate_grounded_answers import ROOT,parse_grounded,digest,write

SOURCE=ROOT/'evaluation/artifacts/answer_quality_v3_grounded_max'
DEST=ROOT/'evaluation/artifacts/answer_quality_v3_grounded_max_visible'


def main():
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] not in ('complete','stopped_with_errors'):raise ValueError('Generation not terminal')
    rows=[json.loads(x) for x in (SOURCE/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=24 or len({r['question_id'] for r in rows})!=24:raise ValueError('Incomplete batch')
    if (DEST/'ready.json').exists():
        ready=json.loads((DEST/'ready.json').read_text(encoding='utf-8'))
        assert digest(DEST/'answers.jsonl')==ready['answers_sha256']
        print('Validated batch already ready; no model calls.');return
    corrected=[];formatted=[]
    for original in rows:
        if original['status_code']!=200 or not original.get('raw_response'):raise ValueError('API/raw response missing')
        row=dict(original);obj,answer=parse_grounded(row['raw_response'],row,allow_visible_markdown=True)
        row.update(status='ok',response=answer,evidence_statements=obj['statements'],
                   insufficient_evidence=obj['insufficient_evidence'],missing=obj.get('missing',''))
        row.pop('error_type',None);row.pop('error_detail',None)
        if any(s['quote_match']!='exact' for s in obj['statements']):formatted.append(row['question_id'])
        corrected.append(row)
    if (DEST/'answers.jsonl').exists():raise ValueError('Partial derived batch exists; preserve before repair')
    DEST.mkdir(parents=True,exist_ok=True)
    (DEST/'input_jobs.json').write_bytes((SOURCE/'input_jobs.json').read_bytes())
    with (DEST/'answers.jsonl').open('w',encoding='utf-8') as stream:
        for row in corrected:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest.update(status='complete',errors=0,refusals=sum(r['insufficient_evidence'] for r in corrected),
        parent_raw_answers_sha256=digest(SOURCE/'answers.jsonl'),parent_folder=str(SOURCE.relative_to(ROOT)),
        quote_validation='Exact text or same visible text after removing Markdown inline links/backticks and collapsing whitespace; no factual word changes',
        visible_format_matches=formatted,revalidation_paid_calls=0)
    write(DEST/'manifest.json',manifest)
    write(DEST/'ready.json',{'status':'ready','answers':24,'questions':24,'schemes':['grounded_final'],
        'answers_sha256':digest(DEST/'answers.jsonl'),'manifest_sha256':digest(DEST/'manifest.json')})
    print(json.dumps({'status':'ready','samples':24,'visible_format_matches':formatted,'paid_calls':0},ensure_ascii=False))


if __name__=='__main__':main()
