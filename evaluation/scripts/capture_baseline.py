"""只读导出评测前状态；不调用 Embedding、聊天或 OCR 服务。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / 'backend'
sys.path.insert(0, str(BACKEND))

from app.config import get_settings  # noqa: E402
from app.rag.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE  # noqa: E402
from pymilvus import MilvusClient  # noqa: E402

SOURCE_COMMIT = 'd99710dea88eac5f7437ec27acfa63aaa535b436'
EXPECTED = {
    'error-codes.md': '616a9b316d96c21ced5f1a29108e7cfdb9c79d7c21bfd475c5821156ca713b62',
    'configure-memory-usage.md': 'f023111c0a5b708d73484927bdb1dae7733dd474d6e86693429c06f3ccb750de',
    'transaction-isolation-levels.md': 'b883c31a07e5d0906cf2cb0c042ef31eba41eba5a6a7f8e5d48da8dbf1e5c0bc',
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_json(path: str):
    with urlopen(f'http://127.0.0.1:8000{path}', timeout=15) as response:
        return json.load(response)


def sanitize(value):
    """导出数据省略机器绝对存储路径。配置另用白名单导出。"""
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items() if key != 'source_path'}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def main() -> int:
    settings = get_settings()
    database = unquote(urlsplit(settings.postgres_dsn).path.lstrip('/'))
    if database != 'rag_benchmark' or settings.milvus_collection != 'tidb_benchmark_v1':
        raise ValueError('Unexpected benchmark configuration')
    if urlsplit(settings.resolved_milvus_uri).hostname not in {'127.0.0.1', 'localhost'}:
        raise ValueError('Milvus must be local for this capture')

    health = get_json('/health')
    documents = get_json('/api/v1/documents')
    chunks = []
    for document in documents:
        chunks.extend(get_json(f"/api/v1/documents/{document['id']}/chunks"))
    chunks.sort(key=lambda item: (item['document_id'], item['chunk_index'], item['id']))

    client = MilvusClient(uri=settings.resolved_milvus_uri, timeout=15)
    try:
        schema = client.describe_collection(settings.milvus_collection)
        primary = next(field['name'] for field in schema['fields'] if field.get('is_primary'))
        vector_count = client.query(
            collection_name=settings.milvus_collection,
            filter='', output_fields=['count(*)'], consistency_level='Strong', timeout=15,
        )[0]['count(*)']
        if vector_count > 5000:
            raise ValueError('Unexpectedly large benchmark collection')
        vector_rows = client.query(
            collection_name=settings.milvus_collection,
            filter='', output_fields=[primary, 'chunk_id', 'document_id', 'text'],
            limit=max(1, vector_count), consistency_level='Strong', timeout=15,
        )
        indexes = {
            name: client.describe_index(settings.milvus_collection, name)
            for name in client.list_indexes(settings.milvus_collection)
        }
    finally:
        client.close()

    files = []
    for name, expected_hash in EXPECTED.items():
        path = ROOT / 'evaluation' / 'corpus' / 'tidb' / name
        files.append({'filename': name, 'bytes': path.stat().st_size,
                      'sha256': sha256(path), 'matches_frozen_source': sha256(path) == expected_hash})

    vectors_by_id = {str(row[primary]): row for row in vector_rows}
    summaries = []
    for document in sorted(documents, key=lambda item: item['filename']):
        items = [chunk for chunk in chunks if chunk['document_id'] == document['id']]
        indices = sorted(chunk['chunk_index'] for chunk in items)
        original = ROOT / 'evaluation' / 'corpus' / 'tidb' / document['filename']
        uploaded = Path(document.get('source_path') or '')
        if not uploaded.is_absolute():
            uploaded = BACKEND / uploaded
        upload_matches = uploaded.is_file() and original.is_file() and sha256(uploaded) == sha256(original)
        summaries.append({
            'filename': document['filename'], 'document_id': document['id'],
            'status': document['status'], 'declared_chunk_count': document['chunk_count'],
            'actual_chunk_count': len(items), 'count_matches': len(items) == document['chunk_count'],
            'enabled_count': sum(chunk['enabled'] for chunk in items),
            'indices_contiguous': indices == list(range(len(items))),
            'splitters': dict(Counter(chunk['metadata_json'].get('splitter_name') for chunk in items)),
            'parsers': dict(Counter(chunk['metadata_json'].get('parser_name') for chunk in items)),
            'uploaded_bytes_match_corpus': upload_matches,
        })

    vector_checks = []
    for chunk in chunks:
        row = vectors_by_id.get(str(chunk['vector_id']))
        vector_checks.append(bool(row and row.get('chunk_id') == chunk['id']
                                  and row.get('document_id') == chunk['document_id']
                                  and row.get('text') == chunk['content']))
    checks = {
        'services_healthy': bool(health.get('ok')),
        'exact_three_expected_documents': sorted(item['filename'] for item in documents) == sorted(EXPECTED),
        'corpus_hashes_match': all(item['matches_frozen_source'] for item in files),
        'documents_indexed': all(item['status'] == 'indexed' for item in summaries),
        'counts_match': all(item['count_matches'] for item in summaries),
        'indices_contiguous': all(item['indices_contiguous'] for item in summaries),
        'all_chunks_enabled': all(chunk['enabled'] for chunk in chunks),
        'uploaded_bytes_match_corpus': all(item['uploaded_bytes_match_corpus'] for item in summaries),
        'unique_chunk_ids': len({chunk['id'] for chunk in chunks}) == len(chunks),
        'unique_vector_ids': len({chunk['vector_id'] for chunk in chunks}) == len(chunks),
        'vector_count_matches': vector_count == len(chunks) == len(vector_rows),
        'vector_ids_metadata_and_text_match': bool(chunks) and all(vector_checks),
    }
    versions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
    code_files = sorted((BACKEND / 'app').rglob('*.py')) + [
        BACKEND / 'run.py', BACKEND / 'requirements.txt', Path(__file__).resolve(),
        ROOT / 'front/rag_front/package.json', ROOT / 'front/rag_front/package-lock.json',
        ROOT / 'front/rag_front/src/views/ChatView.vue',
    ]
    snapshot = {
        'captured_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'purpose': 'Pre-evaluation baseline; no quality metric measured yet',
        'health': health, 'documents': sanitize(documents), 'chunks': sanitize(chunks),
        'summary': summaries, 'checks': checks,
        'corpus': {'repository': 'pingcap/docs-cn', 'commit': SOURCE_COMMIT, 'files': files},
        'local_settings': {
            'database_name': database, 'milvus_collection': settings.milvus_collection,
            'embedding_model': settings.embedding_model, 'chat_model': settings.model,
            'default_chunk_size_characters': DEFAULT_CHUNK_SIZE,
            'default_chunk_overlap_characters': DEFAULT_CHUNK_OVERLAP,
            'note': 'Settings loaded by capture process; server environment overrides are not exported',
        },
        'milvus': {'row_count': vector_count, 'schema': schema, 'indexes': indexes},
        'python_version': platform.python_version(), 'installed_packages': dict(sorted(versions.items())),
        'code_sha256': {path.relative_to(ROOT).as_posix(): sha256(path) for path in code_files},
        'evaluation_plan': {'top_k': 5, 'candidate_k': 20, 'rrf_k': 60,
                            'query_set': 'not_created', 'scores': 'not_measured'},
    }
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    destination = ROOT / 'evaluation' / 'artifacts' / f'baseline_{timestamp}'
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'snapshot.json'
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps({'snapshot': str(target), 'documents': summaries,
                      'chunks': len(chunks), 'vectors': vector_count, 'checks': checks}, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # 不输出连接异常的完整消息，防止意外带出连接凭据。
        print(f'Baseline capture failed: {type(exc).__name__}', file=sys.stderr)
        raise SystemExit(2)
