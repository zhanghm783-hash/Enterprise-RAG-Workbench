"""同24FAQ回答表达前后相关性复算；保留原向量方案作为额外参照。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'evaluation/artifacts'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def main():
    old=ART/'answer_quality_faq_v1_validated';new=ART/'answer_quality_faq_contextual_v2'
    op=old/'relevancy_qwen37max_v1_zh2_embv4';np=new/'relevancy_qwen37max_v1_zh2_grounded_final_embv4'
    om=json.loads((op/'manifest.json').read_text(encoding='utf-8'));nm=json.loads((np/'manifest.json').read_text(encoding='utf-8'))
    for key in ['metric','ragas_version','judge_model','embedding_model','strictness','prompt_language','localized_prompt_instructions','judge_max_tokens']:
        assert om[key]==nm[key],key
    assert sha(old/'answers.jsonl')==om['answers_sha256'] and sha(new/'answers.jsonl')==nm['answers_sha256']
    oa={r['sample_id']:r for r in read(old/'answers.jsonl')};na={r['sample_id']:r for r in read(new/'answers.jsonl')}
    os={r['sample_id']:r for r in read(op/'scores.jsonl')};ns={r['sample_id']:r for r in read(np/'scores.jsonl')}
    assert len(oa)==len(na)==48 and len(ns)==24 and len(os)==48
    rows=[]
    for qid in sorted({r['question_id'] for r in na.values()}):
        key=qid+':grounded_final';ref=qid+':vector'
        before=oa[key];after=na[key]
        assert before['user_input']==after['user_input'] and before['reference']==after['reference']
        assert before['model']==after['model']=='qwen3.7-max'
        assert before['source_chunk_ids']==after['source_chunk_ids']
        assert oa[ref]['response']==na[ref]['response'] and after['status']=='ok'
        assert os[key]['status']==ns[key]['status']==os[ref]['status']=='ok'
        rows.append({'question_id':qid,'before':os[key]['score'],'after':ns[key]['score'],'original_vector_reference':os[ref]['score']})
    assert len(rows)==24
    before=mean(r['before'] for r in rows);after=mean(r['after'] for r in rows)
    report={'status':'passed','samples':24,'comparison':'same FAQ tasks: short grounded answer v1 → contextual grounded answer v2',
        'before':before,'after':after,'delta':after-before,'relative_gain':after/before-1,
        'original_vector_reference':mean(r['original_vector_reference'] for r in rows),
        'change':'restore real section titles and require object/trigger/conclusion in answer expression',
        'judge_model':om['judge_model'],'embedding_model':om['embedding_model'],'language':'zh2',
        'strictness':3,'errors':0,'target_value_met':after>=.85,'positive_version_delta':after>before,
        'scope':'development FAQ cohort; expression optimization, not large improvement over raw vector or production accuracy',
        'previous_semantic_v1_scores_not_comparable':True,'rows':rows,
        'source_sha256':{'before_answers':sha(old/'answers.jsonl'),'after_answers':sha(new/'answers.jsonl'),
                         'before_scores':sha(op/'scores.jsonl'),'after_scores':sha(np/'scores.jsonl')},'auditor_model_calls':0}
    (new/'comparison_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','source_sha256']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
