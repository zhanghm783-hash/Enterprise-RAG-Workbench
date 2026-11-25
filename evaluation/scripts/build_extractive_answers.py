"""统一抽取式回答对照：直接展示模型已选且校验过的原文，不新增模型调用。"""
from __future__ import annotations

import json
from generate_grounded_answers import ROOT,digest,write


def main():
    source=ROOT/'evaluation/artifacts/answer_quality_v3_grounded_max_visible'
    dest=ROOT/'evaluation/artifacts/answer_quality_v4_extractive'
    ready=json.loads((source/'ready.json').read_text(encoding='utf-8'))
    assert digest(source/'answers.jsonl')==ready['answers_sha256']
    rows=[json.loads(x) for x in (source/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==24 and all(r['status']=='ok' for r in rows)
    result=[]
    for original in rows:
        row=dict(original);row['sample_id']=row['question_id']+':extractive_final';row['scheme']='extractive_final'
        selected=[];seen=set()
        for statement in row['evidence_statements']:
            key=(statement['source'],statement['quote'])
            if key in seen:continue
            seen.add(key)
            selected.append('> '+statement['quote'].replace('\n','\n> ')+f' [{statement["source"]}]')
        row['response']='资料依据：\n\n'+'\n\n'.join(selected) if selected else original['response']
        row['rendering_policy']='Uniform verbatim evidence excerpts for all 24 questions; no free-form factual additions'
        result.append(row)
    if (dest/'answers.jsonl').exists():raise ValueError('Extractive version already exists; preserve frozen output')
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'input_jobs.json').write_bytes((source/'input_jobs.json').read_bytes())
    with (dest/'answers.jsonl').open('w',encoding='utf-8') as stream:
        for row in result:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    manifest.update(schemes=['extractive_final'],parent_answers_sha256=ready['answers_sha256'],
        rendering_policy='Evidence-extractive answers using all previously selected quotations; no original factual synthesis',
        extraction_model_calls=0,scope='Extractive answer fidelity; relevance and completeness require separate assessment')
    write(dest/'manifest.json',manifest)
    write(dest/'ready.json',{'status':'ready','answers':24,'questions':24,'schemes':['extractive_final'],
        'answers_sha256':digest(dest/'answers.jsonl'),'manifest_sha256':digest(dest/'manifest.json')})
    print(json.dumps({'status':'ready','samples':24,'model_calls':0},ensure_ascii=False))


if __name__=='__main__':main()
