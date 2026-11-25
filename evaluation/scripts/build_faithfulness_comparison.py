"""离线合并技术性补评，核对24题忠实度前后比较及引用，不调用模型。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from generate_grounded_answers import ROOT,visible_markdown,write


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'evaluation/artifacts/answer_quality_v1'
    raw=base/'faithfulness_qwen37max_v1_zh2_vector'
    final=ROOT/'evaluation/artifacts/answer_quality_v4_extractive'
    graded=final/'faithfulness_qwen37max_v1_zh2'
    before={r['question_id']:r for r in lines(raw/'scores.jsonl')};repairs=[]
    raw_manifest=read(raw/'manifest.json');new_manifest=read(graded/'manifest.json')
    assert len(before)==24
    for qid,row in before.items():
        if row['status']=='ok':continue
        assert row['error_type']=='LLMDidNotFinishException',qid
        child=base/('faithfulness_qwen37max_v1_zh2_vector_ids'+qid+'_max8192')
        repaired=lines(child/'scores.jsonl');assert len(repaired)==1
        assert repaired[0]['question_id']==qid and repaired[0]['status']=='ok'
        cm=read(child/'manifest.json')
        assert cm['localized_prompt_instructions']==raw_manifest['localized_prompt_instructions']
        assert cm['answers_sha256']==raw_manifest['answers_sha256']
        before[qid]=repaired[0];repairs.append({'question_id':qid,'path':str(child.relative_to(ROOT)),
            'original_error':row['error_type'],'change':'judge output capacity 3072 -> 8192 only'})
    after={r['question_id']:r for r in lines(graded/'scores.jsonl')}
    assert len(after)==24 and set(before)==set(after)
    assert raw_manifest['localized_prompt_instructions']==new_manifest['localized_prompt_instructions']
    assert raw_manifest['judge_model']==new_manifest['judge_model']=='qwen3.7-max'
    assert raw_manifest['ragas_version']==new_manifest['ragas_version']=='0.4.4'
    assert all(r['status']=='ok' and math.isfinite(r['score']) for r in list(before.values())+list(after.values()))
    original_answers={r['question_id']:r for r in lines(base/'answers.jsonl') if r['scheme']=='vector'}
    final_answers={r['question_id']:r for r in lines(final/'answers.jsonl')}
    assert len(original_answers)==len(final_answers)==24
    quote_count=0;refusals=[]
    for qid,row in final_answers.items():
        assert original_answers[qid]['user_input']==row['user_input']
        if row['insufficient_evidence']:refusals.append(qid)
        for statement in row['evidence_statements']:
            body=row['context_contents'][statement['source']-1]
            assert visible_markdown(statement['quote']) in visible_markdown(body)
            quote_count+=1
    before_mean=math.fsum(r['score'] for r in before.values())/24
    after_mean=math.fsum(r['score'] for r in after.values())/24
    conservative=math.fsum(r['score'] for qid,r in after.items() if qid not in refusals)/24
    report={'status':'passed','questions':24,'before':before_mean,'after':after_mean,
            'delta':after_mean-before_mean,'metric':'Ragas Faithfulness, fixed Chinese prompt zh2',
            'before_scheme':'vector retrieval + free-form qwen-plus answers',
            'after_scheme':'query rewrite + hybrid retrieval + rerank + qwen3.7-max evidence selection + verbatim quotation rendering',
            'score_errors':0,'undefined':0,'verified_quotations':quote_count,'refusals':refusals,
            'after_with_refusals_zero':conservative,'nonrefusal_fraction':(24-len(refusals))/24,
            'target':0.90,'target_met':after_mean>=0.90 and conservative>=0.90 and len(refusals)<=2,
            'baseline_technical_repairs':repairs,
            'source_sha256':{'baseline_raw_scores':sha(raw/'scores.jsonl'),'improved_scores':sha(graded/'scores.jsonl'),
                             'improved_answers':sha(final/'answers.jsonl')},
            'paired_rows':[{'question_id':qid,'before':before[qid]['score'],'after':after[qid]['score']} for qid in sorted(before)],
            'scope':'24 previously exposed semantic development tasks; extractive-answer fidelity, not complete-answer or production accuracy',
            'human_semantic_review':'pending','application_integration':'not yet implemented',
            'offline_auditor_model_calls':0,'prior_default_English_prompt_scores_mixed_in':False}
    write(final/'comparison_report.json',report)
    write(final/'baseline_repaired_scores.json',list(before.values()))
    print(json.dumps({k:v for k,v in report.items() if k!='paired_rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
