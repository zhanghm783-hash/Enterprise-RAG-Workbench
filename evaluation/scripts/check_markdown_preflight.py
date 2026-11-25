"""离线核对企业语料来源与修复后的标题分段；不访问数据库或模型。"""
from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))

from app.rag.ingest import _split_section  # noqa: E402
from app.rag.loader import _split_markdown_sections, load_document  # noqa: E402


def main():
    pack = ROOT / 'evaluation/corpus/enterprise_tidb_v2'
    output_dir = ROOT / 'evaluation/artifacts/markdown_fence_fix_20261123'
    output_dir.mkdir(parents=True, exist_ok=True)
    before_path = pack / 'parser_preflight.json'
    before_bytes = before_path.read_bytes()
    before = json.loads(before_bytes)
    manifest = json.loads((pack / 'source_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        data = (pack / item['local_path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f"Source changed: {item['local_path']}")

    files = []
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline preflight network call')):
        for item in before['files']:
            document = load_document(pack / item['file'])
            chunks = sum(len(_split_section(text=section.text, file_type=document.file_type,
                                           section_metadata=section.metadata)[1])
                         for section in document.sections)
            files.append({'file': item['file'], 'sections': len(document.sections),
                          'chunks_with_current_parser': chunks,
                          'previous_sections': item['sections'],
                          'previous_chunks': item['chunks_with_current_parser']})
        example = _split_markdown_sections(before['fenced_log_example'])
    if len(example) != 1 or '# Query_time:' not in example[0].text or '# Cop_process:' not in example[0].text:
        raise AssertionError('Fenced log example was not preserved in one section')
    if before_path.read_bytes() != before_bytes:
        raise AssertionError('Original preflight evidence changed')
    output = {
        'scope': 'offline parsing and chunk counting; no indexing, database writes or model calls',
        'documents': len(files), 'source_integrity_checks': len(manifest['files']),
        'previous_total_chunks': before['total_chunks_with_current_parser'],
        'total_chunks_with_current_parser': sum(item['chunks_with_current_parser'] for item in files),
        'files': files, 'fenced_log_example': before['fenced_log_example'],
        'previous_fenced_log_split_count': before['fenced_log_split_count'],
        'fenced_log_split_count': len(example),
        'actual_sections': [{'title': section.metadata['section_title'], 'text': section.text} for section in example],
        'parser_source_sha256': hashlib.sha256((ROOT / 'backend/app/rag/loader.py').read_bytes()).hexdigest(),
        'before_preflight_sha256': hashlib.sha256(before_bytes).hexdigest(),
        'paid_model_calls': 0,
        'limitations': ['保护标题分段层的围栏代码内容；后续长度切分仍可能拆开很长的代码块。',
                        '不属于完整 CommonMark 解析器；不新增嵌套列表、引用块等语法的完整支持。',
                        '分块数量变化不是检索指标提升，尚未新建索引或运行检索评测。'],
    }
    (output_dir / 'after_preflight.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: output[key] for key in ('documents', 'source_integrity_checks', 'previous_total_chunks',
                     'total_chunks_with_current_parser', 'previous_fenced_log_split_count', 'fenced_log_split_count',
                     'paid_model_calls')}, ensure_ascii=False))
    print(json.dumps({'changed_files': [item for item in files if item['sections'] != item['previous_sections']
                                      or item['chunks_with_current_parser'] != item['previous_chunks']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
