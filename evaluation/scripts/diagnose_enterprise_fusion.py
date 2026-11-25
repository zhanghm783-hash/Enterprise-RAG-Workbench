"""已暴露候选的离线诊断和参数重放，不是独立验证，不调用模型。"""
from __future__ import annotations

import itertools
import json

from enterprise_runtime import ROOT,digest,write_json

PATHS={
    'original':ROOT/'evaluation/artifacts/enterprise_test_20261123_172904_689007',
    'semantic':ROOT/'evaluation/artifacts/enterprise_semantic_test_20261123_180731_544685',
}


def complete(record,ids):
    return all(set(ids)&set(gold) for gold in record['evidence_mapping']['required_unit_gold_chunk_ids'].values())


def ranking(record,k,vector_weight,head=0):
    merged={}
    for source,weight in [('vector',vector_weight),('bm25',1)]:
        for rank,hit in enumerate(record['candidates'][source],1):
            entry=merged.setdefault(hit['chunk_id'],{'id':hit['chunk_id'],'f':0.,'b':0.,'v':0.})
            entry['f']+=weight/(k+rank)
            entry['b' if source=='bm25' else 'v']=float(hit['score'])
    ordered=[entry['id'] for entry in sorted(merged.values(),key=lambda item:(item['f'],item['b'],item['v']),reverse=True)]
    reserved=[]
    for index in range(head):
        for source in ['vector','bm25']:
            if len(record['candidates'][source])>index:
                chunk_id=record['candidates'][source][index]['chunk_id']
                if chunk_id not in reserved:reserved.append(chunk_id)
    return (reserved+[chunk_id for chunk_id in ordered if chunk_id not in reserved])[:5]


def main():
    records={name:[json.loads(line) for line in (path/'results.jsonl').read_text(encoding='utf-8').splitlines()]
             for name,path in PATHS.items()}
    baseline={}
    for name,rows in records.items():
        for record in rows:
            if ranking(record,60,1)!=[hit['chunk_id'] for hit in record['results']['hybrid']['hits']]:
                raise ValueError('Baseline replay differs from stored RRF Top5')
        pool=[record for record in rows if complete(record,[hit['chunk_id'] for source in ['vector','bm25']
                                                            for hit in record['candidates'][source]])]
        pool_ids={record['question_id'] for record in pool}
        baseline[name]={'questions':len(rows),'baseline_hits':sum(complete(record,ranking(record,60,1)) for record in rows),
                        'pool20_union_ceiling':len(pool),
                        'candidate_missing_questions':[record['question_id'] for record in rows if record['question_id'] not in pool_ids],
                        'rank_loss_questions':[record['question_id'] for record in pool if not complete(record,ranking(record,60,1))]}
    configs=[]
    for k,weight,head in itertools.product([1,5,10,20,40,60,100],[.25,.5,.75,1,1.5,2,3,4],[0,1,2]):
        scores={name:sum(complete(record,ranking(record,k,weight,head)) for record in rows) for name,rows in records.items()}
        configs.append({'rrf_k':k,'vector_weight':weight,'bm25_weight':1,'reserved_heads_per_route':head,
                        **scores,'total':sum(scores.values())})
    configs.sort(key=lambda item:(item['total'],min(item['original'],item['semantic'])),reverse=True)
    report={'scope':'posthoc development replay of exposed candidates; not new validation',
            'source_results_sha256':{name:digest(path/'results.jsonl') for name,path in PATHS.items()},
            'baseline':baseline,'configurations_evaluated':len(configs),'best_configurations':configs[:15],
            'best_plain_rrf':max([item for item in configs if item['vector_weight']==1 and item['reserved_heads_per_route']==0],key=lambda item:item['total']),
            'best_equal_weight_head_reserve':max([item for item in configs if item['vector_weight']==1],key=lambda item:item['total']),
            'best_semantic_hits_in_grid':max(item['semantic'] for item in configs),
            'interpretation':'Candidate union coverage uses gold labels to diagnose an ideal ranking ceiling; it is not an executable oracle or a measured reranker result. No ranking strategy receives gold labels.',
            'business_code_modified':False,'paid_model_calls':0}
    destination=ROOT/'evaluation/artifacts/enterprise_fusion_diagnosis_20261123'
    destination.mkdir(exist_ok=True)
    write_json(destination/'diagnosis.json',report);write_json(destination/'all_configurations.json',configs)
    print(json.dumps({key:report[key] for key in ['scope','baseline','configurations_evaluated','best_plain_rrf',
                     'best_equal_weight_head_reserve','best_semantic_hits_in_grid','paid_model_calls']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
