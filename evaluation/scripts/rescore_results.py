"""对已保存的开发题结果应用修正后的依据标签；不重新检索，不覆盖原结果。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['split'] != 'dev' or manifest['post_run_data_check'] != 'passed':
        raise ValueError('Only validated development runs can be rescored here')
    original_path = ROOT / 'evaluation/datasets/tidb_v1_draft/questions_draft_pre_dev_review.json'
    updated_path = ROOT / 'evaluation/datasets/tidb_v1_draft/questions_draft.json'
    before = json.loads(original_path.read_text(encoding='utf-8'))
    after = json.loads(updated_path.read_text(encoding='utf-8'))
    if manifest['dataset_sha256'] != sha(original_path):
        raise ValueError('Original dataset does not match run')
    if before['baseline_snapshot_sha256'] != after['baseline_snapshot_sha256']:
        raise ValueError('Baseline changed')
    old_questions = {q['id']: q for q in before['questions']}
    questions = {q['id']: q for q in after['questions']}
    if set(questions) != set(old_questions):
        raise ValueError('Question set changed')
    changed = []
    for qid, question in questions.items():
        for field in ('question', 'reference_answer', 'split', 'topic', 'kind'):
            if question[field] != old_questions[qid][field]:
                raise ValueError('Question or answer changed during label review')
        old_gold = {g['chunk_id'] for g in old_questions[qid]['baseline_full_evidence_chunks']}
        new_gold = {g['chunk_id'] for g in question['baseline_full_evidence_chunks']}
        if old_gold != new_gold:
            if question['split'] != 'dev':
                raise ValueError('Formal gold labels changed during development result review')
            changed.append(qid)
    records = [json.loads(line) for line in (run / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    for record in records:
        question = questions[record['question_id']]
        if record['question'] != question['question']:
            raise ValueError('Recorded question differs')
        gold = {item['chunk_id'] for item in question['baseline_full_evidence_chunks']}
        record['original_auto_gold_chunk_ids'] = record['gold_chunk_ids']
        record['gold_chunk_ids'] = sorted(gold)
        record['evidence_options'] = question['evidence_options']
        for result in record['results'].values():
            result['original_auto_hit_at_5'] = result['hit_at_5']
            relevant = [index for index, hit in enumerate(result['hits'][:5], 1) if hit['chunk_id'] in gold]
            result['hit_at_5'] = bool(relevant)
            result['first_relevant_rank'] = relevant[0] if relevant else None
    summary = {}
    for mode in manifest['modes']:
        entries = [record['results'][mode] for record in records]
        hits = sum(entry['hit_at_5'] for entry in entries)
        summary[mode] = {'questions': len(entries), 'hits': hits,
                         'hit_rate_at_5_including_execution_failures': hits / len(entries),
                         'statuses': dict(Counter(entry['status'] for entry in entries)),
                         'note': 'Development evidence labels reviewed by AI; not a formal test score'}
    output = run / 'label_review_1'
    output.mkdir(exist_ok=False)
    review = {'changed_gold_questions': changed, 'original_results_sha256': sha(run / 'results.jsonl'),
              'original_dataset_sha256': sha(original_path), 'updated_dataset_sha256': sha(updated_path),
              'reviewed_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'reason': after['label_review_note'], 'new_model_calls': 0,
              'question_text_and_reference_answers_unchanged': True,
              'formal_gold_sets_unchanged': True, 'original_retrieval_order_unchanged': True}
    for filename, data in (('review_manifest.json', review), ('summary.json', summary)):
        (output / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'results.jsonl').write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n'
                                              for record in records), encoding='utf-8')
    print(json.dumps({'output': str(output), 'changed_gold_questions': changed, 'summary': summary}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
