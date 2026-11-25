"""可交接的纯离线复算：只读评分原文件，另写审计结果，无模型调用。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SCHEMES=('bm25','vector','final_chain')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--metric',choices=['relevancy','faithfulness','both'],default='both')
    args=parser.parse_args()
    folder=ROOT/'evaluation/artifacts/answer_quality_v1'
    ready=read(folder/'ready.json')
    assert hashlib.sha256((folder/'answers.jsonl').read_bytes()).hexdigest()==ready['answers_sha256']
    answers={r['sample_id']:r for r in map(json.loads,(folder/'answers.jsonl').read_text(encoding='utf-8').splitlines())}
    metrics=('relevancy','faithfulness') if args.metric=='both' else (args.metric,)
    reports={}
    for metric in metrics:
        path=folder/(metric+'_qwen37max_v1')
        manifest=read(path/'manifest.json')
        assert manifest['status'] in ('complete','complete_with_undefined_or_errors'),'Scoring still running'
        assert manifest['judge_model']=='qwen3.7-max'
        assert manifest['answers_sha256']==ready['answers_sha256']
        rows=[json.loads(line) for line in (path/'scores.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==len({r['sample_id'] for r in rows})==72
        assert {r['sample_id'] for r in rows}==set(answers)
        saved=read(path/'summary.json');recomputed={};by_id={r['sample_id']:r for r in rows}
        for scheme in SCHEMES:
            selected=[r for r in rows if r['scheme']==scheme]
            assert len(selected)==24
            for row in selected:
                assert row['question_id']==answers[row['sample_id']]['question_id']
                assert row['scheme']==answers[row['sample_id']]['scheme']
                assert row['metric']==metric
                assert row['status'] in ('ok','undefined','error')
                assert (row['score'] is not None)==(row['status']=='ok')
                if row['score'] is not None:assert math.isfinite(row['score'])
            finite=[r['score'] for r in selected if r['score'] is not None]
            value={'samples':24,'finite_scores':len(finite),
                   'undefined':sum(r['status']=='undefined' for r in selected),
                   'errors':sum(r['status']=='error' for r in selected),
                   'mean_on_finite':math.fsum(finite)/len(finite) if finite else None,
                   'finite_score_fraction':len(finite)/24,
                   'mean_with_missing_as_zero_sensitivity':math.fsum(finite)/24}
            for key,x in value.items():
                y=saved[scheme][key]
                assert (x is None and y is None) or (x is not None and y is not None and math.isclose(x,y,abs_tol=1e-12)),(scheme,key)
            recomputed[scheme]=value
        common=[q for q in sorted({r['question_id'] for r in rows})
                if all(by_id[q+':'+s]['score'] is not None for s in SCHEMES)]
        common_means={s:math.fsum(by_id[q+':'+s]['score'] for q in common)/len(common) if common else None for s in SCHEMES}
        comparisons={}
        for baseline in ('bm25','vector'):
            paired=[q for q in sorted({r['question_id'] for r in rows})
                    if all(by_id[q+':'+s]['score'] is not None for s in (baseline,'final_chain'))]
            differences=[by_id[q+':final_chain']['score']-by_id[q+':'+baseline]['score'] for q in paired]
            comparisons[baseline]={'paired_questions':len(paired),'final_minus_baseline_mean':math.fsum(differences)/len(differences) if differences else None,
                                   'final_higher':sum(d>1e-12 for d in differences),'equal':sum(abs(d)<=1e-12 for d in differences),
                                   'final_lower':sum(d < -1e-12 for d in differences)}
        events=read(path/'api_events.json')
        usage={'judge_input_tokens':sum(e.get('usage',{}).get('input_tokens',0) for e in events if e['kind']=='judge'),
               'judge_output_tokens':sum(e.get('usage',{}).get('output_tokens',0) for e in events if e['kind']=='judge'),
               'embedding_tokens':sum(e.get('usage',{}).get('total_tokens',0) for e in events if e['kind']=='embedding')}
        assert manifest['api_calls']==len(events)
        for k,v in usage.items():assert manifest[k]==v,k
        report={'status':'passed','metric':metric,'source_scores_sha256':hashlib.sha256((path/'scores.jsonl').read_bytes()).hexdigest(),
                'summary':recomputed,'complete_paired_questions':common,'complete_paired_means':common_means,
                'paired_comparison':comparisons,'usage':usage,'api_calls':len(events),
                'sdk_events_with_sample_id':sum(e.get('sample_id') is not None for e in events),
                'limitations':['Offline audit verifies saved score aggregation, not semantic correctness of judge decisions',
                               'SDK executor may lose ContextVar sample IDs; individual scores are keyed, API usage is aggregate'],
                'paid_model_calls_by_auditor':0}
        report['estimated_cost_rmb_from_reported_usage']=(usage['judge_input_tokens']*12+usage['judge_output_tokens']*36+usage['embedding_tokens']*0.7)/1_000_000
        report['pricing']={'qwen3.7-max_input_rmb_per_million':12,'qwen3.7-max_output_rmb_per_million':36,
                           'text-embedding-v1_rmb_per_million':0.7,
                           'sources':['https://help.aliyun.com/zh/model-studio/qwen3-7-max','https://help.aliyun.com/zh/model-studio/model-pricing'],
                           'account_bill_verified':False,'interrupted_inflight_usage_may_be_missing':manifest.get('interrupted_inflight_calls_may_have_unreported_usage',False)}
        (folder/(metric+'_independent_audit.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        reports[metric]=report
        print(json.dumps({'metric':metric,'status':'passed','summary':recomputed},ensure_ascii=False))
    if args.metric=='both':
        report={'status':'passed','metrics':reports,'human_efficiency_measured':False}
        (folder/'independent_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
