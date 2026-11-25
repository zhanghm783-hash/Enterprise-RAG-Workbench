"""企业实验的隔离配置、来源固定和数据核对工具。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from app.config import get_settings  # noqa: E402
from app.core.postgres import _normalize_sqlalchemy_dsn, get_engine, get_session_factory  # noqa: E402
from app.models import Base, Chunk, Document  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

TARGET_DB = 'rag_enterprise_v2'
TARGET_COLLECTION = 'tidb_enterprise_v2'
CONTRACT = ROOT / 'evaluation/datasets/enterprise_tidb_v2_review2/scoring_contract_v2.json'
CONTRACT_SHA = '418f7464d0c0dbd7a054e06e33268ed2f4ee800e1f69953d962422efde764b33'
DRAFT = ROOT / 'evaluation/datasets/enterprise_tidb_v2_draft/questions_draft.json'
DRAFT_SHA = '3f024b87c7ac08c87d6bdc0b56cc0a15b75137a32990ce0a161427fd79299ff9'
PLAN = ROOT / 'evaluation/artifacts/enterprise_ingestion_plan_v2/plan.json'
CORPUS = ROOT / 'evaluation/corpus/enterprise_tidb_v2'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def clean_metadata(value):
    if isinstance(value, dict):
        return {key: clean_metadata(item) for key, item in value.items() if key != 'source_path'}
    if isinstance(value, list):
        return [clean_metadata(item) for item in value]
    return value


def validate_sources():
    if digest(CONTRACT) != CONTRACT_SHA or digest(DRAFT) != DRAFT_SHA:
        raise ValueError('Question or evidence contract changed')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if plan['contract_sha256'] != CONTRACT_SHA:
        raise ValueError('Ingestion plan evidence contract changed')
    manifest_path = CORPUS / 'source_manifest.json'
    if digest(manifest_path) != plan['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for source in manifest['files']:
        if digest(CORPUS / source['local_path']) != source['sha256']:
            raise ValueError('Source file changed')
    old_questions = ROOT / 'evaluation/datasets/tidb_v1_frozen/questions.json'
    if digest(old_questions) != '8168f74ebb963e3494cd1258e12136338369c549249b3fd0eef412f42914cdf3':
        raise ValueError('Old benchmark question file changed')
    return plan, manifest


def isolated_settings():
    original = get_settings()
    url = make_url(original.postgres_dsn)
    from urllib.parse import urlsplit
    if (url.host not in {'127.0.0.1', 'localhost'} or url.database != 'rag_benchmark'
            or original.milvus_collection != 'tidb_benchmark_v1'
            or urlsplit(original.resolved_milvus_uri).hostname not in {'127.0.0.1', 'localhost'}
            or original.embedding_model != 'text-embedding-v1' or original.milvus_dimension != 1536):
        raise ValueError('Original local benchmark configuration changed')
    os.environ['POSTGRES_DSN'] = url.set(database=TARGET_DB).render_as_string(hide_password=False)
    os.environ['MILVUS_COLLECTION'] = TARGET_COLLECTION
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_factory.cache_clear()
    current = get_settings()
    if make_url(current.postgres_dsn).database != TARGET_DB or current.milvus_collection != TARGET_COLLECTION:
        raise ValueError('Isolated environment override failed')
    return original, current


def snapshot_pg(engine, filename_to_path=None):
    with Session(engine) as session:
        docs = session.execute(select(Document).order_by(Document.id)).scalars().all()
        chunks = session.execute(select(Chunk).order_by(Chunk.id)).scalars().all()
        rows = [{'id': chunk.id, 'document_id': chunk.document_id, 'chunk_index': chunk.chunk_index,
                 'content': chunk.content, 'enabled': chunk.enabled, 'vector_id': chunk.vector_id,
                 'metadata_json': clean_metadata(chunk.metadata_json)} for chunk in chunks]
        documents = [{'id': doc.id, 'filename': doc.filename, 'chunk_count': doc.chunk_count,
                      'status': doc.status} for doc in docs]
        if filename_to_path:
            for doc in documents:
                doc['corpus_path'] = filename_to_path[doc['filename']]
            by_id = {doc['id']: doc['corpus_path'] for doc in documents}
            for row in rows:
                row['corpus_path'] = by_id[row['document_id']]
        return {'documents': documents, 'chunks': rows}


def fingerprint(snapshot):
    return hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
