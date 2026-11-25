"""固定依据单元的离线评分；不检索、不调用模型、不访问数据库。"""
from __future__ import annotations

import re


def normalized(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def compile_mapping(rule: dict, chunks: list[dict]) -> dict:
    """将原文选项映射到提供的片段；chunk_id 由调用方提供并注明来源。"""
    ids = [chunk['chunk_id'] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate chunk IDs in mapping input')
    if not rule['required_units']:
        raise ValueError('At least one required evidence unit is necessary')
    texts = [(chunk, normalized(chunk['content'])) for chunk in chunks]

    def matches(unit):
        if not unit['options'] or any(not option['quotes'] for option in unit['options']):
            raise ValueError('Evidence options and quotes cannot be empty')
        return sorted(chunk['chunk_id'] for chunk, text in texts if any(
            chunk['corpus_path'] == option['filename']
            and all(normalized(quote) in text for quote in option['quotes'])
            for option in unit['options']))

    required = {unit['id']: matches(unit) for unit in rule['required_units']}
    if len(required) != len(rule['required_units']):
        raise ValueError('Duplicate required evidence unit IDs')
    optional = {unit['id']: matches(unit) for unit in rule['optional_units']}
    return {'id': rule['id'], 'required_unit_gold_chunk_ids': required,
            'optional_unit_gold_chunk_ids': optional,
            'single_chunk_full_evidence_ids': sorted(set.intersection(*(set(ids) for ids in required.values()))),
            'missing_required_units': [unit_id for unit_id, ids in required.items() if not ids]}


def score_hits(mapping: dict, ordered_chunk_ids: list[str], *, k: int = 5) -> dict:
    """只看原排名前 k 条；重复结果不创造额外依据，不向后补取。"""
    if k < 1:
        raise ValueError('k must be positive')
    required = mapping['required_unit_gold_chunk_ids']
    if not required or mapping['missing_required_units']:
        raise ValueError('Gold mapping is incomplete; cannot score this question')
    top_ids = set(ordered_chunk_ids[:k])
    covered = {unit_id: bool(top_ids.intersection(gold)) for unit_id, gold in required.items()}
    return {'k': k, 'required_unit_coverage': covered,
            'any_evidence_hit_at_k': any(covered.values()),
            'complete_evidence_at_k': all(covered.values()),
            'evidence_unit_coverage_fraction': sum(covered.values()) / len(covered),
            'single_chunk_complete_hit_at_k': bool(top_ids.intersection(mapping['single_chunk_full_evidence_ids'])),
            'optional_unit_coverage': {unit_id: bool(top_ids.intersection(gold))
                                       for unit_id, gold in mapping['optional_unit_gold_chunk_ids'].items()}}
