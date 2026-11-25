"""三路检索消融：共享原始候选，禁用 PG 兜底，保留空结果和执行错误。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from app.config import get_settings  # noqa: E402
from app.core.postgres import get_session_factory  # noqa: E402
from app.core.embeddings import get_embeddings  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.rag.retriever import (  # noqa: E402
    _bm25_retrieve_chunks, _candidate_k, _fuse_hits, _vector_retrieve_chunks, _RRF_K,
)
from pymilvus import MilvusClient  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

DATASET = ROOT / 'evaluation/datasets/tidb_v1_draft/questions_draft.json'
MODES = ('vector', 'bm25', 'hybrid')
TOP_K = 5


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if key != 'source_path'}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def track_embedding_call(original, events):
    """只记录用量白名单；不保存密钥、请求头、完整响应或生成的向量。"""
    def tracked(**kwargs):
        event = {'model': kwargs.get('model'), 'status': 'started'}
        events.append(event)
        try:
            response = original(**kwargs)
            usage = getattr(response, 'usage', None) or {}
            event.update(status='returned', status_code=getattr(response, 'status_code', None),
                         request_id=getattr(response, 'request_id', None),
                         usage={key: usage[key] for key in ('total_tokens', 'input_tokens', 'output_tokens')
                                if usage.get(key) is not None})
            return response
        except Exception as exc:
            event.update(status='exception', error_type=type(exc).__name__)
            raise
    return tracked


def validate_hits(hits, expected):
    ids = [hit.get('chunk_id') for hit in hits]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate candidate IDs')
    for hit in hits:
        original = expected.get(hit.get('chunk_id'))
        if (original is None or hit.get('content') != original['content']
                or hit.get('document_id') != original['document_id']
                or not math.isfinite(float(hit.get('score', float('nan'))))):
            raise ValueError('Candidate differs from benchmark snapshot')


def run_question(question, *, modes, expected, vector_fn, bm25_fn):
    """可注入候选函数，离线测试路径、异常、评分；不向候选函数传标准答案。"""
    if not modes or set(modes) - set(MODES):
        raise ValueError('Unknown or empty mode selection')
    candidate_k = _candidate_k(TOP_K)
    candidates, errors, timings = {}, {}, {}
    for source, enabled, function in (
        ('vector', 'vector' in modes or 'hybrid' in modes, vector_fn),
        ('bm25', 'bm25' in modes or 'hybrid' in modes, bm25_fn),
    ):
        if not enabled:
            continue
        started = perf_counter()
        try:
            hits = function(query=question['question'], candidate_k=candidate_k, raise_on_error=True)
            if len(hits) > candidate_k:
                raise ValueError('Candidate limit exceeded')
            validate_hits(hits, expected)
            candidates[source] = hits
        except Exception as exc:
            candidates[source] = []
            # 异常正文可能含连接凭据，不写入结果或控制台。
            errors[source] = type(exc).__name__
        timings[source] = round((perf_counter() - started) * 1000, 3)

    results = {}
    gold_ids = {item['chunk_id'] for item in question['baseline_full_evidence_chunks']}
    for mode in modes:
        needed = ('vector', 'bm25') if mode == 'hybrid' else (mode,)
        failed = {name: errors[name] for name in needed if name in errors}
        if mode == 'hybrid':
            started = perf_counter()
            hits = _fuse_hits(query=question['question'], vector_hits=copy.deepcopy(candidates['vector']),
                              bm25_hits=copy.deepcopy(candidates['bm25']), top_k=TOP_K)
            fusion_ms = round((perf_counter() - started) * 1000, 3)
        else:
            hits = copy.deepcopy(candidates[mode][:TOP_K])
            fusion_ms = None
        matching = [index for index, hit in enumerate(hits, 1) if hit['chunk_id'] in gold_ids]
        status = 'ok' if not failed else ('degraded' if len(failed) < len(needed) else 'error')
        results[mode] = {
            'status': status, 'errors': failed, 'returned_count': len(hits),
            'hit_at_5': bool(matching), 'first_relevant_rank': matching[0] if matching else None,
            'hits': [dict(sanitize(hit), result_rank=rank) for rank, hit in enumerate(hits, 1)],
            'fusion_ms': fusion_ms,
        }
    return {'question_id': question['id'], 'question': question['question'], 'topic': question['topic'],
            'kind': question['kind'], 'reference_answer': question['reference_answer'],
            'evidence_options': question['evidence_options'], 'gold_chunk_ids': sorted(gold_ids),
            'candidate_k': candidate_k, 'candidates': sanitize(candidates),
            'candidate_component_ms': timings, 'results': results}


def check_postgres(snapshot):
    """只读检查直接运行 BM25 所用数据库与快照一致。"""
    with get_session_factory()() as session:
        actual = session.execute(select(Chunk)).scalars().all()
        expected = {chunk['id']: chunk for chunk in snapshot['chunks']}
        if {chunk.id for chunk in actual} != set(expected):
            raise ValueError('PostgreSQL chunk IDs changed')
        for chunk in actual:
            old = expected[chunk.id]
            if (chunk.content != old['content'] or chunk.enabled != old['enabled']
                    or chunk.document_id != old['document_id'] or chunk.vector_id != old['vector_id']
                    or sanitize(chunk.metadata_json) != old['metadata_json']):
                raise ValueError('PostgreSQL benchmark data changed')


def check_vectors(snapshot, settings):
    client = MilvusClient(uri=settings.resolved_milvus_uri, timeout=15)
    try:
        schema = client.describe_collection(settings.milvus_collection)
        primary = next(field['name'] for field in schema['fields'] if field.get('is_primary'))
        count = client.query(collection_name=settings.milvus_collection, filter='',
                             output_fields=['count(*)'], consistency_level='Strong', timeout=15)[0]['count(*)']
        if count != len(snapshot['chunks']):
            raise ValueError('Milvus benchmark count changed')
        rows = client.query(collection_name=settings.milvus_collection, filter='',
                            output_fields=[primary, 'chunk_id', 'document_id', 'text'],
                            limit=max(1, count), consistency_level='Strong', timeout=15)
        by_id = {str(row[primary]): row for row in rows}
        for chunk in snapshot['chunks']:
            row = by_id.get(str(chunk['vector_id']))
            if (not row or row.get('chunk_id') != chunk['id'] or row.get('text') != chunk['content']
                    or row.get('document_id') != chunk['document_id']):
                raise ValueError('Milvus benchmark data changed')
    finally:
        client.close()


def summarize(records, modes):
    result = {}
    for mode in modes:
        entries = [record['results'][mode] for record in records]
        hits = sum(entry['hit_at_5'] for entry in entries)
        result[mode] = {'questions': len(entries), 'hits': hits,
                        'hit_rate_at_5_including_execution_failures': hits / len(entries) if entries else None,
                        'statuses': dict(Counter(entry['status'] for entry in entries)),
                        'note': 'Provisional automatic evidence labels; execution failures retained, not dropped'}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--allow-embedding', action='store_true', help='明确允许问题向量化的付费 API 调用')
    args = parser.parse_args(argv)
    modes = list(dict.fromkeys(args.modes))
    needs_vector = 'vector' in modes or 'hybrid' in modes
    if needs_vector and not args.allow_embedding:
        parser.error('向量/混合模式需要 --allow-embedding；不带此参数不会调用问题向量化')
    dataset_path = args.dataset.resolve()
    dataset = json.loads(dataset_path.read_text(encoding='utf-8'))
    if args.split == 'test' and dataset.get('status') != 'frozen':
        parser.error('正式题集尚未冻结；当前只允许运行开发题')
    if args.split == 'test':
        frozen_hash = dataset_path.with_suffix('.sha256').read_text(encoding='ascii').strip()
        if digest(dataset_path) != frozen_hash:
            raise ValueError('Frozen question file changed')
    snapshot_path = ROOT / dataset['baseline_snapshot']
    if digest(snapshot_path) != dataset['baseline_snapshot_sha256']:
        raise ValueError('Baseline snapshot changed')
    snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
    if not all(snapshot['checks'].values()):
        raise ValueError('Invalid baseline snapshot')
    for item in dataset['source_files']:
        if digest(ROOT / 'evaluation/corpus/tidb' / item['filename']) != item['sha256']:
            raise ValueError('Corpus changed')
    questions = [item for item in dataset['questions'] if item['split'] == args.split]
    if not questions or len(questions) != dataset['counts'][args.split]:
        raise ValueError('Question count mismatch')
    settings = get_settings()
    connection = make_url(settings.postgres_dsn)
    from urllib.parse import urlsplit
    if (connection.host not in {'127.0.0.1', 'localhost'} or connection.database != 'rag_benchmark'
            or urlsplit(settings.resolved_milvus_uri).hostname not in {'127.0.0.1', 'localhost'}
            or settings.milvus_collection != snapshot['local_settings']['milvus_collection']
            or settings.embedding_model != snapshot['local_settings']['embedding_model']):
        raise ValueError('Benchmark connection or model mismatch')
    check_postgres(snapshot)
    if needs_vector:
        check_vectors(snapshot, settings)
    expected = {chunk['id']: chunk for chunk in snapshot['chunks']}
    now = datetime.now(timezone(timedelta(hours=8)))
    destination = ROOT / 'evaluation/artifacts' / f"{args.split}_{'-'.join(modes)}_{now:%Y%m%d_%H%M%S_%f}"
    destination.mkdir(parents=True, exist_ok=False)
    paths = sorted((ROOT / 'backend/app').rglob('*.py')) + sorted(Path(__file__).parent.glob('*.py'))
    manifest = {
        'started_at': now.isoformat(), 'split': args.split, 'modes': modes,
        'dataset_status': dataset['status'], 'dataset_sha256': digest(dataset_path),
        'baseline_sha256': digest(snapshot_path), 'top_k': TOP_K, 'candidate_k': _candidate_k(TOP_K),
        'rrf_k': _RRF_K, 'question_count': len(questions), 'postgres_fallback': False,
        'embedding_enabled': needs_vector, 'embedding_model': settings.embedding_model,
        'max_logical_embedding_calls': len(questions) if needs_vector else 0,
        'api_call_note': 'Shared vector candidates per question; provider retries can add HTTP attempts',
        'timing_note': 'Components run sequentially and are reused; timings are not live parallel hybrid latency',
        'code_sha256': {path.relative_to(ROOT).as_posix(): digest(path) for path in paths},
        'packages': {name: importlib.metadata.version(name) for name in
                     ('langchain-milvus', 'pymilvus', 'rank-bm25', 'jieba', 'SQLAlchemy', 'dashscope')},
        'post_run_data_check': 'pending',
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    records, embedding_events = [], []
    with ExitStack() as stack:
        if needs_vector:
            client = get_embeddings().client
            stack.enter_context(patch.object(client, 'call', side_effect=track_embedding_call(client.call, embedding_events)))
        output = stack.enter_context((destination / 'results.jsonl').open('w', encoding='utf-8'))
        for question in questions:
            first_event = len(embedding_events)
            record = run_question(question, modes=modes, expected=expected,
                                  vector_fn=_vector_retrieve_chunks, bm25_fn=_bm25_retrieve_chunks)
            record['embedding_sdk_events'] = copy.deepcopy(embedding_events[first_event:])
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps({'id': question['id'], 'results': {
                mode: {'status': record['results'][mode]['status'], 'hit_at_5': record['results'][mode]['hit_at_5']}
                for mode in modes}}, ensure_ascii=False), flush=True)
    try:
        check_postgres(snapshot)
        if needs_vector:
            check_vectors(snapshot, settings)
        manifest['post_run_data_check'] = 'passed'
    except Exception as exc:
        manifest['post_run_data_check'] = f'failed:{type(exc).__name__}'
    summary = summarize(records, modes)
    manifest['embedding_sdk_calls'] = len(embedding_events)
    manifest['embedding_reported_total_tokens'] = sum(event.get('usage', {}).get('total_tokens', 0)
                                                       for event in embedding_events)
    manifest['embedding_usage_note'] = 'SDK-returned usage only; internal HTTP retries and final account bill not inspected'
    manifest['finished_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (destination / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(destination), 'summary': summary,
                      'post_run_data_check': manifest['post_run_data_check']}, ensure_ascii=False, indent=2))
    errors = any(record['results'][mode]['status'] != 'ok' for record in records for mode in modes)
    return 1 if errors or manifest['post_run_data_check'] != 'passed' else 0


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Evaluation stopped: {type(exc).__name__}', file=sys.stderr)
        raise SystemExit(2)
