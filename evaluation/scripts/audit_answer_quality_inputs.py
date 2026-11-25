"""离线核对三组回答及实际上下文，不读取密钥或调用模型。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    folder=ROOT/'evaluation/artifacts/answer_quality_v1'
    ready=read_json(folder/'ready.json')
    for filename,key in [('answers.jsonl','answers_sha256'),('manifest.json','manifest_sha256')]:
        assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==ready[key],filename
    answers=[json.loads(line) for line in (folder/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    jobs={row['sample_id']:row for row in read_json(folder/'input_jobs.json')}
    assert len(answers)==len(jobs)==72
    assert len({row['sample_id'] for row in answers})==72
    manifest=read_json(folder/'manifest.json')
    source_paths={
        'baseline':ROOT/'evaluation/artifacts/enterprise_semantic_test_20261123_180731_544685/results.jsonl',
        'improved':ROOT/'evaluation/artifacts/enterprise_rewrite_rerank_20261123_185611_118735/results.jsonl'}
    for name,path in source_paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['source_results_sha256'][name],name
    counts=Counter();questions={};empty=[];truncated=[];lengths=[]
    for row in answers:
        assert row['status']=='ok' and row['status_code']==200,row['sample_id']
        assert row['response'].strip(),row['sample_id']
        job=jobs[row['sample_id']]
        for key,value in job.items():assert row[key]==value,(row['sample_id'],key)
        scheme=row['scheme'];counts[scheme]+=1
        q=questions.setdefault(row['question_id'],{})
        q[scheme]=(row['user_input'],row['reference'])
        bodies=row['context_contents'];blocks=row['retrieved_contexts'];ids=row['source_chunk_ids']
        assert len(bodies)==len(blocks)==len(ids)<=5
        assert len(ids)==len(set(ids))
        for body,block in zip(bodies,blocks):
            assert len(body)<=500 and body in block
            lengths.append(len(body))
        assert '\n\n'.join(blocks)==row['rendered_context']
        if not bodies:empty.append(row['sample_id'])
        if row.get('finish_reason')=='length':truncated.append(row['sample_id'])
    assert counts=={'bm25':24,'vector':24,'final_chain':24}
    for question,schemes in questions.items():
        assert set(schemes)==set(counts),question
        assert len(set(schemes.values()))==1,question
    assert len(questions)==24
    report={'status':'passed','samples':72,'paired_questions':24,'schemes':dict(counts),
            'answers_sha256':ready['answers_sha256'],'manifest_sha256':ready['manifest_sha256'],
            'input_jobs_preserved':True,'paired_questions_and_reference_equal':True,
            'actual_context_body_max_characters':max(lengths),'context_limit':500,
            'empty_context_samples':empty,'generation_length_limit_samples':truncated,
            'paid_calls':0,'semantic_human_review':'pending; structural audit does not establish answer correctness'}
    (folder/'input_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
