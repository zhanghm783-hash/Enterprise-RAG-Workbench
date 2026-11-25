"""汇总已保存的正式检索结果及排名差异；离线执行，不调用模型。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.rag.retriever import _fuse_hits  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(records, mode):
    hits = sum(record['results'][mode]['hit_at_5'] for record in records)
    return {'hits': hits, 'questions': len(records), 'hit_rate_at_5': hits / len(records)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    frozen_path = ROOT / 'evaluation/datasets/tidb_v1_frozen/questions.json'
    dataset = json.loads(frozen_path.read_text(encoding='utf-8'))
    if (manifest['split'] != 'test' or manifest['dataset_sha256'] != digest(frozen_path)
            or dataset['status'] != 'frozen' or manifest['post_run_data_check'] != 'passed'):
        raise ValueError('Formal dataset or data validation mismatch')
    records = [json.loads(line) for line in (run / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    formal = {question['id']: question for question in dataset['questions'] if question['split'] == 'test'}
    if len(records) != 40 or {record['question_id'] for record in records} != set(formal):
        raise ValueError('Formal question IDs incomplete or duplicated')
    modes = manifest['modes']
    for record in records:
        question = formal[record['question_id']]
        expected_gold = {item['chunk_id'] for item in question['baseline_full_evidence_chunks']}
        if record['question'] != question['question'] or set(record['gold_chunk_ids']) != expected_gold:
            raise ValueError('Question or gold labels changed')
        for mode in modes:
            result = record['results'][mode]
            if result['status'] != 'ok' or result['errors']:
                raise ValueError('Execution failure in formal results')
            hit = any(item['chunk_id'] in expected_gold for item in result['hits'][:5])
            if hit != result['hit_at_5']:
                raise ValueError('Saved score mismatch')
    groups = {}
    for field in ('topic', 'kind'):
        groups[field] = {value: {mode: score([record for record in records if record[field] == value], mode)
                                for mode in modes} for value in sorted({record[field] for record in records})}
    comparisons = {}
    for left, right in (('hybrid', 'vector'), ('hybrid', 'bm25'), ('bm25', 'vector')):
        comparisons[f'{left}_vs_{right}'] = {
            'wins': [r['question_id'] for r in records if r['results'][left]['hit_at_5'] and not r['results'][right]['hit_at_5']],
            'losses': [r['question_id'] for r in records if not r['results'][left]['hit_at_5'] and r['results'][right]['hit_at_5']],
            'ties': sum(r['results'][left]['hit_at_5'] == r['results'][right]['hit_at_5'] for r in records),
        }
    diagnostics = []
    for record in records:
        if all(record['results'][mode]['hit_at_5'] for mode in modes):
            continue
        gold = set(record['gold_chunk_ids'])
        complete_fusion = _fuse_hits(query=record['question'], vector_hits=record['candidates']['vector'],
                                     bm25_hits=record['candidates']['bm25'], top_k=40)
        diagnostics.append({
            'question_id': record['question_id'], 'question': record['question'],
            'reference_answer': record['reference_answer'],
            'valid_evidence_ranks': {mode: [{'rank': rank, 'filename': hit['filename'],
                                             'chunk_index': hit['chunk_index'], 'chunk_id': hit['chunk_id']}
                                            for rank, hit in enumerate(hits, 1) if hit['chunk_id'] in gold]
                                    for mode, hits in {'vector': record['candidates']['vector'],
                                                       'bm25': record['candidates']['bm25'],
                                                       'full_fusion': complete_fusion}.items()},
            'top_five': {mode: record['results'][mode]['hits'] for mode in modes},
        })
    tokens = manifest['embedding_reported_total_tokens']
    analysis = {
        'split': 'test', 'dataset_sha256': digest(frozen_path),
        'results_sha256': digest(run / 'results.jsonl'), 'analysis_script_sha256': digest(Path(__file__)),
        'verification': '40 unique questions; fixed labels unchanged; 120 executions ok; scores independently recomputed',
        'scores': {mode: score(records, mode) for mode in modes}, 'groups': groups,
        'paired_comparisons': comparisons,
        'failure_question_ids': {mode: [r['question_id'] for r in records if not r['results'][mode]['hit_at_5']] for mode in modes},
        'embedding_usage': {'sdk_calls': manifest['embedding_sdk_calls'], 'reported_total_tokens': tokens,
                            'estimated_rmb': tokens * 0.7 / 1_000_000,
                            'price_rmb_per_million_input_tokens': 0.7,
                            'price_reference_url': 'https://help.aliyun.com/zh/model-studio/model-pricing',
                            'price_checked_date': '2026-11-23', 'account_bill_verified': False},
        'new_model_calls_for_analysis': 0,
        'limits': 'Three documents, 84 chunks, 40 AI-authored source-reviewed factual questions; not independent document holdout or production traffic; answer generation untested',
    }
    target = run / 'formal_analysis.json'
    target.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (run / 'rank_diagnostics.json').write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (run / 'per_question_scores.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['题号', '主题', '题型', '问题', '参考答案', '向量命中', 'BM25命中', '混合命中',
                         '向量首个正确排名', 'BM25首个正确排名', '混合首个正确排名'])
        for record in records:
            writer.writerow([record['question_id'], record['topic'], record['kind'], record['question'],
                             record['reference_answer'], *[record['results'][mode]['hit_at_5'] for mode in modes],
                             *[record['results'][mode]['first_relevant_rank'] for mode in modes]])
    print(json.dumps({'scores': analysis['scores'], 'topic_scores': groups['topic'],
                      'paired_comparisons': comparisons, 'usage': analysis['embedding_usage'],
                      'diagnostic_ranks': [{key: item[key] for key in ('question_id', 'valid_evidence_ranks')}
                                           for item in diagnostics]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
