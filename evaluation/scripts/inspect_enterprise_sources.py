"""核对企业场景资料的原始版本与业务问题依据；不入库，不调用模型。"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / 'evaluation/corpus/enterprise_tidb_v2'


def blob_hash(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def anchor(text, quote, filename):
    position = text.find(quote)
    if position < 0:
        raise ValueError(f'Missing source anchor in {filename}: {quote}')
    return {'file': filename, 'quote': quote, 'line': text.count('\n', 0, position) + 1}


def main():
    planned = json.loads((PACK / 'source_manifest_expected.json').read_text(encoding='utf-8'))
    files, texts, newline_restored = [], {}, []
    for item in planned['files'] + planned['source_files']:
        relative = item['path'] if item in planned['files'] else '_source/' + item['path']
        path = PACK / relative
        data = path.read_bytes()
        if blob_hash(data) != item['expected_git_blob_sha1']:
            # apply_patch 为新文件补了末尾换行；仅在去掉该换行可还原官方 blob 时恢复。
            if data.endswith(b'\n') and blob_hash(data[:-1]) == item['expected_git_blob_sha1']:
                data = data[:-1]
                path.write_bytes(data)
                newline_restored.append(relative)
            else:
                raise ValueError(f'Git blob mismatch: {relative}')
        if len(data) != item['expected_bytes']:
            raise ValueError(f'Source byte count mismatch: {relative}')
        text = data.decode('utf-8')
        texts[item['path']] = text
        title_match = re.search(r'^title:\s*(.+)$', text, re.MULTILINE)
        files.append({**item, 'local_path': relative, 'bytes': len(data), 'characters': len(text),
                      'sha256': hashlib.sha256(data).hexdigest(), 'git_blob_verified': True,
                      'title': title_match.group(1) if title_match else item['path'],
                      'source_url': f"https://github.com/{planned['repository']}/blob/{planned['commit']}/{item['path']}"})

    cases = [
        ('C01', '查询断连与内存故障确认', '业务查询突然断开，报 2013；监控中内存升高后归零。怎样确认是否发生 OOM？',
         'troubleshoot-tidb-oom.md', ['Lost connection to MySQL server during query',
                                     'dmesg -T | grep tidb-server', 'oom-killer'],
         ['2013', '英文报错'], '断连、监控变化与故障确认步骤'),
        ('C02', '长连接与预处理语句资源管理', '应用使用长连接，反复创建 PreparedStatement 且没有释放，内存只涨不降。可以从哪些方面排查和限制？',
         'troubleshoot-tidb-oom.md', ['deallocate prepare stmt', 'session 关闭后才会释放', 'max_prepared_stmt_count'],
         ['PreparedStatement'], '长期连接、资源不释放与内存持续上涨'),
        ('C03', '定位锁等待中的阻塞事务', '业务报 Lock wait timeout exceeded，事务长时间卡住。哪个系统表能看到谁阻塞了谁？',
         'troubleshoot-lock-conflicts.md', ['Lock wait timeout exceeded', 'DATA_LOCK_WAITS',
                                          '阻塞和被阻塞的事务的 `start_ts`'],
         ['Lock wait timeout exceeded'], '业务卡住、寻找阻塞关系'),
        ('C04', '并发写入冲突定位', '并发修改同一批数据，日志出现 [kv:9007]Write conflict，怎样从日志定位冲突的表和索引？',
         'troubleshoot-write-conflicts.md', ['[kv:9007]Write conflict', 'tableID 表示发生冲突的表的 ID',
                                            'indexID 表示是索引数据发生了冲突'],
         ['9007', 'Write conflict'], '并发修改、定位冲突对象'),
        ('C05', '用慢日志定位查询瓶颈', '慢日志 Query_time 约 0.18 秒，Cop_process 约 170ms、Cop_wait 约 1ms，优先定位哪一侧的处理瓶颈？',
         'analyze-slow-queries.md', ['Query_time: 0.18085', 'Cop_process: Avg_time: 170ms', '瓶颈在 TiKV 侧'],
         ['Query_time', 'Cop_process', 'Cop_wait', '数值'], '接口慢、区分数据库不同组件的处理耗时'),
        ('C06', '写入热点与改动生效范围', '没有主键的表，大量 INSERT 总集中在一个 Region，如何把后续写入打散？这会重排已有数据吗？',
         'troubleshoot-hot-spot-issues.md', ['对于非聚簇索引主键或没有主键的表', 'SHARD_ROW_ID_BITS',
                                           '每次修改之后，只对新写入的数据生效'],
         ['INSERT', 'Region'], '局部热点、打散写入与已有数据的边界'),
    ]
    business_cases = []
    for case_id, task, query, filename, quotes, exact_signals, semantic_intent in cases:
        business_cases.append({
            'case_id': case_id, 'business_task': task, 'question_example': query,
            'question_provenance': 'AI 根据官方资料拟定的业务问题示意，不是企业客户记录或冻结评测题',
            'anchors': [anchor(texts[filename], quote, filename) for quote in quotes],
            'exact_signals': exact_signals, 'semantic_intent': semantic_intent,
            'rrf_advantage': 'hypothesis_not_measured',
        })
    syntax_observations = []
    for item in planned['files']:
        fence, hits = False, []
        for line_number, line in enumerate(texts[item['path']].splitlines(), 1):
            if line.lstrip().startswith('```'):
                fence = not fence
            elif fence and re.match(r'^#{1,6}\s+', line):
                hits.append({'line': line_number, 'text': line[:100]})
        if hits:
            syntax_observations.append({'file': item['path'], 'code_lines_resembling_headings': hits})
    output = {
        **planned, 'status': 'source_bytes_verified_not_indexed', 'files': files,
        'business_document_count': len(planned['files']),
        'business_total_bytes': sum(item['bytes'] for item in files if item['category'] != '出处与许可'),
        'business_total_characters': sum(item['characters'] for item in files if item['category'] != '出处与许可'),
        'category_counts': dict(Counter(item['category'] for item in planned['files'])),
        'source_license': 'CC BY-SA 3.0, as stated in the pinned README; upstream LICENSE preserved',
        'source_final_newline_restored': newline_restored,
        'business_cases': business_cases, 'markdown_syntax_observations': syntax_observations,
        'notes': [
            '公开企业产品资料构成模拟运维／技术支持知识库；不声称企业内部部署或真实用户验证。',
            '固定 Git 提交中仍含历史版本说明，标注时保留版本、事务模式和组件条件。',
            '同时保存口语症状和精确日志，作为两路互补的待验证场景；不承诺融合获胜。',
            '代码块中的 # 日志或配置注释不能作为真正的文档标题，入库前检查当前解析器的处理。',
            '业务问题示意尚未冻结，不作为已经建立的开发题／正式题或实际效果。',
        ], 'paid_model_calls': 0,
    }
    (PACK / 'source_manifest.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: output[key] for key in ('status', 'business_document_count', 'business_total_bytes',
                     'business_total_characters', 'category_counts', 'source_final_newline_restored', 'paid_model_calls')},
                     ensure_ascii=False, indent=2))
    print(json.dumps({'source_integrity_checks': len(files), 'business_case_examples': len(business_cases),
                      'files_with_log_or_config_hash_lines': [item['file'] for item in syntax_observations]},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
