"""纯本地独立复算三项优先指标，可交给只执行任务的助手。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def main():
    baseline=ROOT/'evaluation/artifacts/enterprise_semantic_test_20261123_180731_544685/results.jsonl'
    final=ROOT/'evaluation/artifacts/enterprise_rewrite_rerank_20261123_185611_118735/results.jsonl'
    frozen=ROOT/'evaluation/datasets/enterprise_semantic_paraphrase_v1/questions.json'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()=='4f4549d7002d07371f31e474e5869f807183cdde52a41fbea4ebb8f3c54aa10f'
    questions={q['id']:q for q in json.loads(frozen.read_text(encoding='utf-8'))['questions']}
    first=[json.loads(line) for line in baseline.read_text(encoding='utf-8').splitlines()]
    last=[json.loads(line) for line in final.read_text(encoding='utf-8').splitlines()]
    totals={'bm25':0,'vector':0,'final_chain':0}
    for records in [first,last]:assert len(records)==24 and {r['question_id'] for r in records}==set(questions)
    for records,modes in [(first,['bm25','vector']),(last,['final_chain'])]:
        for record in records:
            q=questions[record['question_id']];assert q['question']==record['question']
            for mode in modes:
                hits=record['hits'] if mode=='final_chain' else record['results'][mode]['hits']
                assert len(hits)<=5
                ids={hit['chunk_id'] for hit in hits}
                totals[mode]+=all(ids&set(gold) for gold in q['evidence_mapping']['required_unit_gold_chunk_ids'].values())
    assert totals=={'bm25':8,'vector':11,'final_chain':21}
    output={'status':'passed','questions':24,'top_k':5,'totals':totals,'rates':{key:value/24 for key,value in totals.items()},
            'source_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [baseline,final,frozen]},
            'paid_model_calls':0,'source_files_modified':False}
    target=ROOT/'evaluation/artifacts/deepseek_readonly_audit';target.mkdir(parents=True,exist_ok=True)
    (target/'priority_results_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
