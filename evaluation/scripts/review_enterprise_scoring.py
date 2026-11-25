"""检索前固定跨块评分契约并离线核对可用性，不改写题目草稿。"""
from __future__ import annotations

import csv
import hashlib
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

from enterprise_evidence_scoring import compile_mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.rag.ingest import _split_section  # noqa: E402
from app.rag.loader import load_document  # noqa: E402

EXPECTED_DRAFT = '3f024b87c7ac08c87d6bdc0b56cc0a15b75137a32990ce0a161427fd79299ff9'


def unit(unit_id, description, filename, anchors):
    return {'id': unit_id, 'description': description,
            'options': [{'filename': filename, 'quotes': [anchor['quote'] for anchor in anchors]}]}


def write_once(path: Path, value: dict):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Versioned evidence already exists with different content: {path.name}')
    if not path.exists():
        path.write_bytes(data)


def main():
    source_dir = ROOT / 'evaluation/datasets/enterprise_tidb_v2_draft'
    destination = ROOT / 'evaluation/datasets/enterprise_tidb_v2_review1'
    draft_path = source_dir / 'questions_draft.json'
    draft_bytes = draft_path.read_bytes()
    if hashlib.sha256(draft_bytes).hexdigest() != EXPECTED_DRAFT:
        raise ValueError('Draft differs from recorded source-anchored version')
    dataset = json.loads(draft_bytes)
    rules = []
    for q in dataset['questions']:
        anchors = q['anchors']
        rule = {'id': q['id'], 'split': q['split'], 'required_units': [
            unit('answer', '问题所需的完整解释／处理依据', q['filename'], anchors)], 'optional_units': []}
        if q['id'] == 'EV04':
            rule['required_units'] = [unit('mode_scope', 'Lock View 仅提供悲观锁冲突和等待信息', q['filename'], anchors[:1]),
                                      unit('sql_visibility', '不能恢复含格式和参数的完整 SQL', q['filename'], anchors[1:])]
            rule['review_reason'] = '问题明确问两个预期，范围限制与 SQL 可见性均是必答项；允许前五条分开覆盖。'
        elif q['id'] == 'EV06':
            rule['required_units'] = [unit('ttl_limit', '悲观锁默认 TTL 上限及配置参数', q['filename'], anchors[:2]),
                                      unit('business_action', '优化业务并考虑拆分大事务', q['filename'], anchors[2:])]
            rule['review_reason'] = '问题同时问时间限制和业务调整；仅返回错误名称或参数不足以回答处理办法。'
        elif q['id'] == 'EV08':
            rule['required_units'] = [unit('preexecution_explanation', '子查询可能提前执行，算子耗时不包含该阶段', q['filename'], anchors[:2])]
            rule['optional_units'] = [unit('numeric_example', '说明相同现象的慢日志数值示例', q['filename'], anchors[2:])]
            rule['review_reason'] = '题目已给出数值，询问原因；解释是必要依据，重复给出的示例数值不是必答项。'
        rules.append(rule)
    contract = {
        'schema_version': 1, 'contract_id': 'enterprise_tidb_v2_evidence_v1',
        'status': 'scoring_semantics_fixed_before_retrieval; actual_index_mapping_pending',
        'parent_draft_sha256': EXPECTED_DRAFT, 'counts': dataset['counts'],
        'reviewer': 'AI source review under user authorization; not user independent or external review',
        'questions_and_reference_answers_changed': False,
        'metrics': {
            'primary': 'CompleteEvidence@5：每题的全部必要依据单元均被前五条覆盖才记 1，可由多条合起来覆盖。',
            'secondary': 'AnyEvidenceHit@5：前五条至少覆盖一个必要单元，不能等同完整命中。',
            'diagnostic': 'SingleChunkCompleteHit@5：前五条内存在一条独自覆盖全部必要单元的片段。',
            'aggregation': '分别对开发 8 题、正式 24 题求平均；保留所有题，失败留在分母，未映射金标先阻止评分。',
            'optional': '可选佐证单独记录，不影响主指标或部分命中；各检索方案共用同一标签和 Top K。',
            'comparison_limit': '原 40 题采用单条完整依据的 Hit Rate@5；新主指标允许多块组合，不能将两组成绩当作同口径前后优化。',
        },
        'mapping_policy': '本次原文摘录选项仍需补齐语义等价来源。实际入库后映射真实 ID，正式检索前冻结标签；不依据排名回写答案或删除失败题。',
        'questions': rules,
    }
    corpus = ROOT / 'evaluation/corpus/enterprise_tidb_v2'
    manifest_path = corpus / 'source_manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != dataset['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    manifest = json.loads(manifest_bytes)
    chunks = []
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline scoring review network call')):
        for source in manifest['files']:
            path = corpus / source['local_path']
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != source['sha256']:
                raise ValueError(f"Source changed: {source['local_path']}")
            if source['category'] == '出处与许可':
                continue
            document = load_document(path)
            index = 0
            for section in document.sections:
                for chunk in _split_section(text=section.text, file_type=document.file_type,
                                            section_metadata=section.metadata)[1]:
                    chunks.append({'chunk_id': f"offline:{source['local_path']}:{index}",
                                   'corpus_path': source['local_path'], 'chunk_index': index,
                                   'content': chunk.content})
                    index += 1
    mappings = [compile_mapping(rule, chunks) for rule in rules]
    if any(mapping['missing_required_units'] for mapping in mappings):
        raise ValueError('Some required units have no literal source candidates')
    if draft_path.read_bytes() != draft_bytes:
        raise ValueError('Original draft changed')
    destination.mkdir(parents=True, exist_ok=True)
    write_once(destination / 'scoring_contract_v1.json', contract)
    contract_hash = hashlib.sha256((destination / 'scoring_contract_v1.json').read_bytes()).hexdigest()
    (destination / 'scoring_contract_v1.sha256').write_text(contract_hash+'\n', encoding='ascii')
    report = {
        'scope': 'offline source mapping only; not ranked retrieval, final gold labels or database IDs',
        'parent_draft_sha256': EXPECTED_DRAFT, 'scoring_contract_sha256': contract_hash,
        'source_integrity_checks': len(manifest['files']), 'question_count': len(rules), 'offline_chunks': len(chunks),
        'required_units': sum(len(rule['required_units']) for rule in rules),
        'optional_units': sum(len(rule['optional_units']) for rule in rules),
        'all_required_units_have_literal_candidates': True,
        'single_chunk_full_evidence_questions': sum(bool(mapping['single_chunk_full_evidence_ids']) for mapping in mappings),
        'multiple_chunks_required_by_current_literal_mapping': [mapping['id'] for mapping in mappings if not mapping['single_chunk_full_evidence_ids']],
        'semantic_equivalent_source_review': 'pending', 'actual_index_mapping': 'pending',
        'retrieval_calls': 0, 'paid_model_calls': 0, 'mappings': mappings,
    }
    write_once(destination / 'offline_mapping_review.json', report)
    by_id = {rule['id']: rule for rule in rules}
    with (destination / 'question_evidence_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'split', 'question', 'reference_answer', 'required_units', 'optional_units', 'review_reason'])
        for q in dataset['questions']:
            rule = by_id[q['id']]
            writer.writerow([q['id'], q['split'], q['question'], q['reference_answer'],
                             '; '.join(unit['description'] for unit in rule['required_units']),
                             '; '.join(unit['description'] for unit in rule['optional_units']), rule.get('review_reason', '')])
    print(json.dumps({key:value for key,value in report.items() if key != 'mappings'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
