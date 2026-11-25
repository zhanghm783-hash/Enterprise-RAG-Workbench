"""固定题集读取；FAQ继承父题的入库快照与金标，比较时不混用问法。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PATHS={
    'enterprise':'evaluation/datasets/enterprise_tidb_v2_frozen/questions.json',
    'semantic':'evaluation/datasets/enterprise_semantic_paraphrase_v1/questions.json',
    'faq':'evaluation/datasets/enterprise_quality_faq_v1/questions.json'}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(name):
    path=ROOT/PATHS[name];sha=digest(path)
    if sha!=path.with_suffix('.sha256').read_text(encoding='ascii').strip():raise ValueError('Frozen dataset changed')
    data=json.loads(path.read_text(encoding='utf-8'))
    if name=='faq':
        parent_path=ROOT/PATHS['semantic']
        if digest(parent_path)!=data['parent_sha256']:raise ValueError('FAQ parent changed')
        parent=json.loads(parent_path.read_text(encoding='utf-8'))
        old={q['id']:q for q in parent['questions']}
        if set(old)!={q['id'] for q in data['questions']}:raise ValueError('FAQ tasks were filtered')
        for q in data['questions']:
            for key in ('reference_answer','evidence_mapping','gold_chunk_ids','split'):
                if q[key]!=old[q['id']][key]:raise ValueError('FAQ answer/label changed')
            if q['parent_question']!=old[q['id']]['question']:raise ValueError('FAQ provenance mismatch')
        data={**parent,**data,'dataset_id':'enterprise_quality_faq_v1','status':'frozen'}
    return path,sha,data
