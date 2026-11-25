"""原文题集的离线分块可用性检查，不生成正式金标，不执行检索。"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import socket
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.rag.ingest import _split_section  # noqa: E402
from app.rag.loader import load_document  # noqa: E402


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def main():
    directory = ROOT / 'evaluation/datasets/enterprise_tidb_v2_draft'
    dataset_path = directory / 'questions_draft.json'
    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    with (directory / 'questions_review.csv').open(encoding='utf-8-sig', newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    if [q['id'] for q in csv_rows] != [q['id'] for q in dataset['questions']]:
        raise ValueError('CSV question IDs differ from JSON')
    corpus = ROOT / 'evaluation/corpus/enterprise_tidb_v2'
    source = json.loads((corpus / 'source_manifest.json').read_text(encoding='utf-8'))
    chunks = {}
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline evidence check network call')):
        for item in source['files']:
            data = (corpus / item['local_path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError(f"Source changed: {item['local_path']}")
            if item['category'] == '出处与许可':
                continue
            doc = load_document(corpus / item['local_path'])
            chunks[item['local_path']] = [normalized(chunk.content) for section in doc.sections
                for chunk in _split_section(text=section.text, file_type=doc.file_type,
                                           section_metadata=section.metadata)[1]]
    checks = []
    for q in dataset['questions']:
        raw = (corpus / q['filename']).read_text(encoding='utf-8')
        anchor_matches = []
        for anchor in q['anchors']:
            if raw[anchor['start_char']:anchor['end_char']] != anchor['quote']:
                raise ValueError(f"Source anchor offset mismatch: {q['id']}")
            anchor_matches.append([i for i, text in enumerate(chunks[q['filename']]) if normalized(anchor['quote']) in text])
        common = sorted(set.intersection(*(set(matches) for matches in anchor_matches)))
        checks.append({'id': q['id'], 'split': q['split'], 'filename': q['filename'],
                       'anchor_chunk_indices': anchor_matches,
                       'single_chunk_contains_all_literal_anchors': common,
                       'needs_review': not common,
                       'note': '文本摘录匹配仅作可用性诊断；不等于语义金标、检索结果或已入库 ID'})
    if dataset_path.read_bytes() != dataset_bytes:
        raise ValueError('Draft unexpectedly changed during read-only check')
    output = {
        'scope': 'offline literal evidence feasibility only; not retrieval or final labels',
        'dataset_sha256': hashlib.sha256(dataset_bytes).hexdigest(),
        'question_count': len(checks), 'csv_rows': len(csv_rows), 'source_integrity_checks': len(source['files']),
        'total_chunks': sum(map(len, chunks.values())),
        'questions_with_single_chunk_all_literal_anchors': sum(bool(q['single_chunk_contains_all_literal_anchors']) for q in checks),
        'needs_review': [q['id'] for q in checks if q['needs_review']],
        'question_checks': checks, 'paid_model_calls': 0,
    }
    (directory / 'evidence_feasibility.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key:value for key,value in output.items() if key != 'question_checks'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
