"""正式检索前复核原文锚点，固定题目、等价依据及评分契约，不调用模型。"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from build_questions import locate, normalized

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / 'evaluation/datasets/tidb_v1_draft/questions_draft.json'
DESTINATION = ROOT / 'evaluation/datasets/tidb_v1_frozen'

# 原文复核发现的等价答案，加入时尚未查看正式检索结果。
ALTERNATIVES = {
    'E01': ('error-codes.md', ['set @@tidb_skip_isolation_level_check = 1;'],
            '绕过检查的 SQL 本身明确给出了变量名和取值。'),
    'E02': ('error-codes.md', ["SET GLOBAL tidb_gc_life_time = '30m';"],
            '同一文档的 GC 过早清理长事务数据示例也明确给出调大 GC 保留时间的变量。'),
    'M09': ('configure-memory-usage.md',
            ['tidb-server has the risk of OOM because of memory usage exceeds alarm ratio'],
            '真实日志示例也包含题目所问的报警关键词。'),
    'T01': ('transaction-isolation-levels.md', ['TiDB 实现的是论文中的快照隔离级别。'],
            'ANSI 隔离级别对比段也明确说明 TiDB 实现快照隔离。'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DESTINATION.exists():
        raise FileExistsError('Frozen dataset already exists; do not overwrite')
    if any((ROOT / 'evaluation/artifacts').glob('test_*')):
        raise ValueError('Formal results already exist; initial freeze must precede them')
    draft = json.loads(DRAFT.read_text(encoding='utf-8'))
    snapshot_path = ROOT / draft['baseline_snapshot']
    if sha(snapshot_path) != draft['baseline_snapshot_sha256']:
        raise ValueError('Baseline changed')
    baseline = json.loads(snapshot_path.read_text(encoding='utf-8'))
    texts = {}
    for item in draft['source_files']:
        source = ROOT / 'evaluation/corpus/tidb' / item['filename']
        if sha(source) != item['sha256']:
            raise ValueError('Corpus changed')
        texts[item['filename']] = source.read_text(encoding='utf-8')
    frozen = copy.deepcopy(draft)
    changed_sets = []
    review_rows = []
    for question in frozen['questions']:
        if question['split'] != 'test':
            continue
        old_ids = {item['chunk_id'] for item in question['baseline_full_evidence_chunks']}
        note = '已对照原文复核问题、答案及适用条件；现有依据保留。'
        if question['id'] in ALTERNATIVES:
            filename, quotes, note = ALTERNATIVES[question['id']]
            alternatives = [dict(locate(texts[filename], quote, 0, len(texts[filename])), filename=filename)
                            for quote in quotes]
            question['evidence_options'].append(alternatives)
        if question['id'] == 'E06':
            note = '依据须对应获取事务 start tso 的故障上下文；仅提到 PD 的其他错误描述不自动视为完整依据。'
        for option in question['evidence_options']:
            for anchor in option:
                source_text = texts[anchor['filename']]
                if source_text[anchor['start_character']:anchor['end_character']] != anchor['text']:
                    raise ValueError(f"Invalid source anchor: {question['id']}")
        gold = []
        for chunk in baseline['chunks']:
            filename = chunk['metadata_json']['filename']
            if any(all(anchor['filename'] == filename
                       and normalized(anchor['text']) in normalized(chunk['content']) for anchor in option)
                   for option in question['evidence_options']):
                gold.append({'chunk_id': chunk['id'], 'chunk_index': chunk['chunk_index'], 'filename': filename})
        if not gold:
            raise ValueError(f"No reviewed full evidence: {question['id']}")
        question['baseline_full_evidence_chunks'] = gold
        question['review_status'] = 'ai_source_reviewed_before_formal_run'
        question['review_note'] = note
        if old_ids != {item['chunk_id'] for item in gold}:
            changed_sets.append(question['id'])
        review_rows.append([question['id'], question['topic'], question['kind'], question['question'],
                            question['reference_answer'], '\n或者\n'.join('\n'.join(
                                f"[{anchor['filename']}:{anchor['line_start']}] {anchor['text']}" for anchor in option)
                                for option in question['evidence_options']),
                            ', '.join(f"{item['filename']} #{item['chunk_index']}" for item in gold), note])
    formal = [question for question in frozen['questions'] if question['split'] == 'test']
    if len(formal) != 40 or len({question['id'] for question in formal}) != 40:
        raise ValueError('Expected 40 unique formal questions')
    for old, new in zip(draft['questions'], frozen['questions']):
        for field in ('id', 'split', 'question', 'reference_answer', 'topic', 'kind'):
            if old[field] != new[field]:
                raise ValueError('Question or reference answer changed')
    frozen.update(status='frozen', frozen_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  source_draft_sha256=sha(DRAFT), formal_gold_sets_changed_before_run=changed_sets,
                  reviewer='AI source review; user authorized execution and received sample explanations',
                  user_individual_question_review='not_claimed', label_revision='formal_source_review_1')
    frozen['scoring_contract'].update(
        primary_score='Fixed reviewed source-evidence chunk IDs; retain all three mode results',
        label_error_policy='Preserve original score and publish versioned corrections applied to all modes; never silently replace labels',
        formal_exposure='All 40 questions and source labels frozen before first formal retrieval; same-document holdout only',
    )
    DESTINATION.mkdir(parents=True, exist_ok=False)
    target = DESTINATION / 'questions.json'
    target.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (DESTINATION / 'questions.sha256').write_text(sha(target) + '\n', encoding='ascii')
    with (DESTINATION / 'source_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['题号', '主题', '题型', '问题', '参考答案', '原文依据', '有效片段', 'AI 原文复核意见'])
        writer.writerows(review_rows)
    print(json.dumps({'file': str(target), 'sha256': sha(target), 'formal_questions': len(formal),
                      'formal_characters': sum(len(question['question']) for question in formal),
                      'topics': dict(Counter(question['topic'] for question in formal)),
                      'gold_sets_changed_before_formal_run': changed_sets,
                      'new_model_calls': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
