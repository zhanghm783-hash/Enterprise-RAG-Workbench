"""真实向量化并隔离入库；需 --allow-embedding，拒绝覆盖已有实验。"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from enterprise_runtime import (
    ROOT, TARGET_DB, TARGET_COLLECTION, CONTRACT, CONTRACT_SHA, DRAFT, DRAFT_SHA, CORPUS,
    Base, _normalize_sqlalchemy_dsn, get_engine, get_session_factory, isolated_settings,
    validate_sources, snapshot_pg, fingerprint, write_json, digest,
)
from enterprise_evidence_scoring import compile_mapping
from app.core.embeddings import get_embeddings
from app.rag.ingest import _split_section, ingest_file_document
from app.rag.loader import load_document
from run_retrieval import check_vectors, track_embedding_call
from pymilvus import MilvusClient
from sqlalchemy import create_engine, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-embedding', action='store_true')
    args = parser.parse_args()
    if not args.allow_embedding:
        parser.error('Real document embedding requires --allow-embedding')
    plan, _ = validate_sources()
    original, settings = isolated_settings()
    old_engine = create_engine(_normalize_sqlalchemy_dsn(original.postgres_dsn))
    old_before = snapshot_pg(old_engine)
    if len(old_before['documents']) != 3 or len(old_before['chunks']) != 84:
        raise ValueError('Old benchmark count changed')
    client = MilvusClient(uri=settings.resolved_milvus_uri, timeout=20)
    with old_engine.connect() as connection:
        exists = connection.execute(text('SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname=:name)'), {'name': TARGET_DB}).scalar()
    if exists or client.has_collection(TARGET_COLLECTION):
        raise ValueError('Target already exists; no automatic overwrite or deletion')
    frozen_dir = ROOT / 'evaluation/datasets/enterprise_tidb_v2_frozen'
    if frozen_dir.exists():
        raise ValueError('Frozen enterprise dataset already exists')

    # 在任何付费调用或新建索引前，核对离线计划的实际输入。
    import hashlib
    planned = {(row['corpus_path'], row['chunk_index']): row for row in plan['planned_chunks']}
    actual_texts = {}
    for document in plan['documents']:
        loaded = load_document(CORPUS / document['corpus_path'])
        chunks = [chunk for section in loaded.sections for chunk in _split_section(
            text=section.text, file_type=loaded.file_type, section_metadata=section.metadata)[1]]
        for index, chunk in enumerate(chunks):
            key = (document['corpus_path'], index)
            if key not in planned or hashlib.sha256(chunk.content.encode('utf-8')).hexdigest() != planned[key]['content_sha256']:
                raise ValueError('Offline ingestion plan differs from actual parser output')
            actual_texts[key] = chunk.content
    if set(actual_texts) != set(planned):
        raise ValueError('Planned chunk count changed')
    now = datetime.now(timezone(timedelta(hours=8)))
    output = ROOT / 'evaluation/artifacts' / f'enterprise_ingest_{now:%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'started', 'started_at': now.isoformat(), 'database': TARGET_DB,
              'collection': TARGET_COLLECTION, 'expected_documents': 20, 'expected_chunks': 677,
              'contract_sha256': CONTRACT_SHA, 'draft_sha256': DRAFT_SHA,
              'old_pg_before_sha256': fingerprint(old_before), 'database_created': False,
              'source_code_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in
                                     [ROOT/'backend/app/rag/loader.py', ROOT/'backend/app/rag/ingest.py']},
              'indexed_documents': [], 'long_row_preflight': 'pending', 'usage_note': 'SDK usage; internal HTTP attempts and account bill not independently inspected'}
    write_json(output/'manifest.json', report)
    events, cache = [], {}
    phase = ['long_row_preflight']
    embeddings = get_embeddings()
    original_call = embeddings.client.call
    tracked_call = track_embedding_call(original_call, events)

    def bounded_call(**kwargs):
        if len(events) >= 60 or sum(event.get('usage', {}).get('total_tokens', 0) for event in events) >= 1_000_000:
            raise RuntimeError('Embedding workload stop threshold reached')
        try:
            return tracked_call(**kwargs)
        finally:
            if events:
                events[-1]['phase'] = phase[0]
            write_json(output/'embedding_sdk_events.json', events)

    original_embed = embeddings.embed_documents

    def cached_embed(self, texts):
        if self is not embeddings:
            raise ValueError('Unexpected embedding instance')
        pending = list(dict.fromkeys(value for value in texts if value not in cache))
        if pending:
            vectors = original_embed(pending)
            if len(vectors) != len(pending):
                raise ValueError('Embedding row count mismatch')
            for value, vector in zip(pending, vectors, strict=True):
                if len(vector) != 1536 or not all(math.isfinite(float(x)) for x in vector):
                    raise ValueError('Invalid embedding dimensions or values')
                cache[value] = vector
        return [cache[value] for value in texts]

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(embeddings.client, 'call', side_effect=bounded_call))
            stack.enter_context(patch.object(type(embeddings), 'embed_documents', cached_embed))
            long_rows = [row for row in plan['planned_chunks'] if row['characters'] > 2048]
            cached_embed(embeddings, [actual_texts[(row['corpus_path'], row['chunk_index'])] for row in long_rows])
            report['long_row_preflight'] = {'status': 'provider_accepted_and_returned_1536_dimensions',
                                            'rows': long_rows, 'vectors_reused_during_ingestion': True}
            write_json(output/'manifest.json', report)
            print(json.dumps({'long_row_preflight': 'accepted', 'rows': len(long_rows)}, ensure_ascii=False), flush=True)
            # 名称为本脚本固定字面量；不将外部输入插入 DDL。
            with old_engine.connect().execution_options(isolation_level='AUTOCOMMIT') as connection:
                connection.execute(text('CREATE DATABASE rag_enterprise_v2'))
            report['database_created'] = True
            Base.metadata.create_all(bind=get_engine())
            phase[0] = 'document_ingestion'
            for source in plan['documents']:
                with get_session_factory()() as session:
                    document = ingest_file_document(session, file_path=CORPUS/source['corpus_path'],
                        original_filename=source['upload_filename'], knowledge_base='enterprise_v2',
                        file_size=(CORPUS/source['corpus_path']).stat().st_size)
                    if document.chunk_count != source['expected_chunks'] or document.status != 'indexed':
                        raise ValueError('Indexed document differs from plan')
                    session.commit()
                    report['indexed_documents'].append({'corpus_path':source['corpus_path'], 'document_id':document.id,
                                                       'chunks':document.chunk_count})
                write_json(output/'manifest.json', report)
                print(json.dumps({'indexed':len(report['indexed_documents']), 'file':source['corpus_path'],
                                  'chunks':source['expected_chunks']}, ensure_ascii=False), flush=True)
        filename_to_path = {row['upload_filename']:row['corpus_path'] for row in plan['documents']}
        snapshot = snapshot_pg(get_engine(), filename_to_path)
        if len(snapshot['documents']) != 20 or len(snapshot['chunks']) != 677:
            raise ValueError('Final PostgreSQL count mismatch')
        for row in snapshot['chunks']:
            key = (row['corpus_path'], row['chunk_index'])
            if row['content'] != actual_texts[key] or not row['vector_id'] or not row['enabled']:
                raise ValueError('Final chunk content or status mismatch')
        check_vectors(snapshot, settings)
        snapshot['local_settings'] = {'milvus_collection':TARGET_COLLECTION, 'embedding_model':settings.embedding_model,
                                      'database':TARGET_DB}
        write_json(output/'snapshot.json', snapshot)
        contract = json.loads(CONTRACT.read_text(encoding='utf-8'))
        rules = {row['id']:row for row in contract['questions']}
        mapping_chunks = [{'chunk_id':row['id'], 'corpus_path':row['corpus_path'], 'content':row['content']} for row in snapshot['chunks']]
        mappings = {qid:compile_mapping(rule, mapping_chunks) for qid,rule in rules.items()}
        if any(mapping['missing_required_units'] for mapping in mappings.values()):
            raise ValueError('Real index gold mapping incomplete')
        dataset = copy.deepcopy(json.loads(DRAFT.read_text(encoding='utf-8')))
        dataset.update(status='frozen', frozen_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                       scoring_contract=CONTRACT.relative_to(ROOT).as_posix(), scoring_contract_sha256=CONTRACT_SHA,
                       ingestion_snapshot=(output/'snapshot.json').relative_to(ROOT).as_posix(),
                       ingestion_snapshot_sha256=digest(output/'snapshot.json'), metrics=contract['metrics'])
        for q in dataset['questions']:
            mapping = mappings[q['id']]
            q['evidence_mapping'] = mapping
            q['gold_chunk_ids'] = sorted(set().union(*(set(ids) for ids in mapping['required_unit_gold_chunk_ids'].values())))
            q['gold_mapping_status'] = 'fixed_actual_index_ids_before_ranked_retrieval'
        frozen_dir.mkdir(parents=True, exist_ok=False)
        write_json(frozen_dir/'questions.json', dataset)
        (frozen_dir/'questions.sha256').write_text(digest(frozen_dir/'questions.json')+'\n', encoding='ascii')
        report.update(status='complete', indexed_documents_count=20, indexed_chunks=677,
                      pg_milvus_full_content_check='passed', frozen_dataset_sha256=digest(frozen_dir/'questions.json'),
                      cache_reused_long_rows=len(long_rows))
    except Exception as exc:
        report.update(status='stopped', error_type=type(exc).__name__, note='Partial isolated targets retained; no automatic cleanup or deletion')
        raise
    finally:
        old_after = snapshot_pg(old_engine)
        report['old_pg_after_sha256'] = fingerprint(old_after)
        report['old_pg_unchanged'] = report['old_pg_before_sha256'] == report['old_pg_after_sha256']
        report['embedding_sdk_calls'] = len(events)
        report['reported_total_tokens'] = sum(event.get('usage', {}).get('total_tokens', 0) for event in events)
        report['estimated_cost_rmb'] = report['reported_total_tokens']*0.7/1e6
        report['finished_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        write_json(output/'manifest.json', report)
        print(json.dumps({'output':output.relative_to(ROOT).as_posix(), 'status':report['status'],
                          'embedding_sdk_calls':report['embedding_sdk_calls'], 'tokens':report['reported_total_tokens'],
                          'estimated_cost_rmb':report['estimated_cost_rmb'], 'old_pg_unchanged':report['old_pg_unchanged']},ensure_ascii=False),flush=True)
        client.close()
        old_engine.dispose()
    if not report['old_pg_unchanged']:
        raise ValueError('Old benchmark data changed')


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    try:
        main()
    except Exception as exc:
        print(f'Enterprise ingestion stopped: {type(exc).__name__}', file=sys.stderr)
        raise SystemExit(2)
