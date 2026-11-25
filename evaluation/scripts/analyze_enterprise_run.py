"""从固定映射独立重算企业评测，复查候选排名；无数据库或模型调用。"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging

from enterprise_runtime import ROOT,digest,write_json
from run_retrieval import _fuse_hits


def rank_units(hits,mapping):
    return {unit_id:next((i for i,hit in enumerate(hits,1) if hit['chunk_id'] in ids),None)
            for unit_id,ids in mapping['required_unit_gold_chunk_ids'].items()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();directory=ROOT/args.run_dir
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    dataset_path=ROOT/manifest.get('dataset_path','evaluation/datasets/enterprise_tidb_v2_frozen/questions.json')
    if digest(dataset_path)!=manifest['dataset_sha256']:raise ValueError('Frozen dataset differs from run')
    dataset=json.loads(dataset_path.read_text(encoding='utf-8'))
    questions={q['id']:q for q in dataset['questions'] if q['split']==manifest['split']}
    records=[json.loads(line) for line in (directory/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records)!=len(questions) or {r['question_id'] for r in records}!=set(questions):
        raise ValueError('Question set differs from frozen split')
    totals={mode:0 for mode in ('vector','bm25','hybrid')};details=[];csv_rows=[]
    for record in records:
        q=questions[record['question_id']];mapping=q['evidence_mapping']
        row={'id':q['id'],'question':q['question'],'reference_answer':q['reference_answer']}
        if q['question']!=record['question']:raise ValueError('Question content changed')
        for mode in totals:
            result=record['results'][mode];ranks=rank_units(result['hits'][:5],mapping)
            full=all(rank is not None for rank in ranks.values());any_hit=any(rank is not None for rank in ranks.values())
            if full!=result['evidence_score']['complete_evidence_at_k'] or any_hit!=result['evidence_score']['any_evidence_hit_at_k']:
                raise ValueError('Stored evidence score differs from independent recomputation')
            totals[mode]+=full;row[mode+'_complete']=full;row[mode+'_status']=result['status']
        all_fused=_fuse_hits(query=q['question'],vector_hits=copy.deepcopy(record['candidates']['vector']),
                              bm25_hits=copy.deepcopy(record['candidates']['bm25']),
                              top_k=len(record['candidates']['vector'])+len(record['candidates']['bm25']))
        if [h['chunk_id'] for h in all_fused[:5]]!=[h['chunk_id'] for h in record['results']['hybrid']['hits']]:
            raise ValueError('Replayed fusion differs from saved Top5')
        if not all(row[mode+'_complete'] for mode in totals):
            details.append({'id':q['id'],'question':q['question'],'reference_answer':q['reference_answer'],
                'required_unit_best_rank':{'vector_top20':rank_units(record['candidates']['vector'],mapping),
                                          'bm25_top20':rank_units(record['candidates']['bm25'],mapping),
                                          'full_fusion':rank_units(all_fused,mapping)},
                'complete_top5':{mode:row[mode+'_complete'] for mode in totals}})
        csv_rows.append(row)
    summary=json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    for mode,total in totals.items():
        if summary[mode]['complete_evidence_hits']!=total:raise ValueError('Summary differs')
    report={'question_count':len(records),'scheme_question_results':len(records)*3,'independent_score_check':'passed',
            'fusion_replay_check':'passed','totals':totals,'differences_and_misses':details,
            'dataset_sha256':manifest['dataset_sha256'],'paid_model_calls':0}
    write_json(directory/'independent_analysis.json',report)
    with (directory/'per_question_scores.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    logging.disable(logging.CRITICAL);main()
