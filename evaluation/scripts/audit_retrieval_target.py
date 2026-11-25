"""独立按固定必要依据集合复算前后Top5；无模型调用。"""
from __future__ import annotations

import argparse
import json
from evaluation_datasets import ROOT,load_dataset,digest


def read_rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['semantic','faq'],required=True)
    parser.add_argument('--baseline-run',required=True);parser.add_argument('--improved-run',required=True);args=parser.parse_args()
    data_path,data_sha,data=load_dataset(args.dataset);questions={q['id']:q for q in data['questions']}
    old_path=ROOT/args.baseline_run/'results.jsonl';new_path=ROOT/args.improved_run/'results.jsonl'
    old={r['question_id']:r for r in read_rows(old_path)};new={r['question_id']:r for r in read_rows(new_path)}
    assert len(old)==len(new)==len(questions)==24 and set(old)==set(new)==set(questions)
    totals={'vector':0,'bm25':0,'rrf':0,'improved':0};rows=[]
    for qid,q in questions.items():
        assert old[qid]['question']==new[qid]['question']==q['question']
        assert new[qid]['status']=='ok'
        assert new[qid]['reference_answer']==q['reference_answer']
        scores={}
        for mode,hits in [('vector',old[qid]['results']['vector']['hits']),('bm25',old[qid]['results']['bm25']['hits']),
                          ('rrf',old[qid]['results']['hybrid']['hits']),('improved',new[qid]['hits'])]:
            if mode!='improved':assert old[qid]['results'][{'rrf':'hybrid'}.get(mode,mode)]['status']=='ok'
            ids={h['chunk_id'] for h in hits};assert len(hits)==len(ids)<=5
            complete=all(ids.intersection(gold) for gold in q['evidence_mapping']['required_unit_gold_chunk_ids'].values())
            totals[mode]+=int(complete);scores[mode]=bool(complete)
        rows.append({'question_id':qid,**scores})
    rates={mode:count/24 for mode,count in totals.items()}
    report={'status':'passed','dataset':data['dataset_id'],'dataset_sha256':data_sha,'questions':24,'top_k':5,
            'totals':totals,'rates':rates,'vector_to_improved_percentage_points':100*(rates['improved']-rates['vector']),
            'rrf_to_improved_percentage_points':100*(rates['improved']-rates['rrf']),
            'target':0.90,'target_met':rates['improved']>=0.90,'rows':rows,
            'source_sha256':{'dataset':data_sha,'baseline':digest(old_path),'improved':digest(new_path)},
            'scope':'fixed previously exposed development tasks; full query-rewrite/hybrid/rerank pipeline, not pure RRF or production distribution',
            'auditor_model_calls':0}
    (ROOT/args.improved_run/'comparison_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','source_sha256']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
