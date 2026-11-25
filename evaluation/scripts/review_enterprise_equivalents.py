"""检索前复核等价出处，保存独立版本；不访问数据库或模型。"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import socket
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

from enterprise_evidence_scoring import compile_mapping
from review_enterprise_scoring import ROOT, write_once, _split_section, load_document

PARENT_HASH = '89ce7cbba6002e1dbd3188e0e27e26d7edc848c54ef7b7a807d6774ab1535873'
SEARCHES = {
    'ED01': r'oom-killer|dmesg.*tidb', 'ED02': r'prepared.*statement|max_prepared_stmt_count|session.*生命周期',
    'ED03': r'data_lock_waits|阻塞和被阻塞', 'ED04': r'tableID|indexID|handle=x',
    'ED05': r'Cop_process|瓶颈在 TiKV|170ms', 'ED06': r'SHARD_ROW_ID_BITS|只对新写入',
    'ED07': r'pd-server.*连通|防火墙白名单|nc.*curl', 'ED08': r'server is busy|busy.*原因',
    'EV01': r'HashAgg|STREAM_AGG', 'EV02': r'2 到 3 倍|事务.*内存.*放大|大事务.*拆分',
    'EV03': r'defaultFetchSize|客户端.*缓存|JDBC.*OOM', 'EV04': r'Lock View|归一化|完整语句',
    'EV05': r'fair_locking|pessimistic lock retry|公平.*锁', 'EV06': r'max-txn-ttl|TTL.*小时|大事务.*拆分|拆分.*事务',
    'EV07': r'Wait_TS|等待获取事务 TS', 'EV08': r'Preproc_subqueries|子查询.*提前|优化阶段.*执行',
    'EV09': r'Cop_wait|cop-task.*等待|实例忙', 'EV10': r'gRPC duration|KV duration|网卡带宽',
    'EV11': r'ulimit|Too many open files', 'EV12': r'两个 tikv|文件.*占用',
    'EV13': r'--config|第一次启动.*配置|pd-ctl.*配置', 'EV14': r'oflag=direct|绕开文件系统|绕过文件系统',
    'EV15': r'时间差|时间.*显示|监控.*显示不', 'EV16': r'ORDER BY|结果.*顺序|不保证顺序',
    'EV17': r'DELETE.*空间|compact 时|compaction 时|空间回收', 'EV18': r'在线 DDL|backfill|回填|阻塞 DML',
    'EV19': r'cluster ID|集群 ID|数据不匹配', 'EV20': r'1 TB|mysqldump|Backup & Restore',
    'EV21': r'Data Migration|流量迁移|网络 LB', 'EV22': r'只有一台机器|大部分机器|整体 I/O|I/O.*热点',
    'EV23': r'AUTO_RANDOM|LAST_INSERT_ID|主键.*业务意义', 'EV24': r'BEGIN.*OPTIMISTIC|BEGIN.*PESSIMISTIC|优先级高于.*txn_mode',
}


def additions():
    return [
        ('ED01', 'answer', 'tidb-troubleshooting-map.md',
         ['`dmesg -T | grep tidb-server` 结果中有事故发生附近时间点的 OOM-killer 的日志。'],
         '同样提供故障时间附近的 OS OOM-killer 证据和命令，可回答确认方法；不把重启日志单独当确认。'),
        ('ED04', 'answer', 'troubleshoot-write-conflicts.md',
         ['tableID 表示发生冲突的表的 ID，indexID 表示是索引数据发生了冲突。'],
         '题目要求解释已给出的 tableID/indexID；handle=x 是补充说明，不是问题另问的必要字段。'),
        ('ED04', 'answer', 'troubleshoot-hot-spot-issues.md',
         ['TiDB 对每个表分配一个 TableID，每一个索引都会分配一个 IndexID'],
         '字段定义足以解释题目已给出的表与索引标识；不要求再次召回 9007 报错字符串。'),
        ('EV06', 'ttl_limit', 'pessimistic-transaction.md',
         ['悲观事务的执行时间仍有上限，默认为 1 小时', '`[performance]` 类别下的 `max-txn-ttl` 修改。'],
         '同为悲观事务时限与配置位置；只覆盖限制单元，不替代业务处理建议，也不据此推断 GC 完全相同。'),
        ('EV06', 'business_action', 'troubleshoot-tidb-oom.md',
         ['针对单个大事务，可以通过拆分的方式调小事务大小。'],
         '提供拆分大事务这一通用业务动作，仅覆盖动作单元；仍须另有悲观事务超时依据，不声称保证解决超时。'),
        ('EV07', 'answer', 'analyze-slow-queries.md',
         ['可以对比慢日志中的 `Wait_TS` 和 `Query_time`，因为 TS 有预取操作，通常来说 `Wait_TS` 应该很低。'],
         '题目已提供两个耗时数值，取 TS 阶段的解释足够，不必再次召回相同数值示例。'),
        ('EV07', 'answer', 'identify-slow-queries.md',
         ['`Query_time`：表示执行这个语句花费的时间。', '`Wait_TS`：表示这个语句在等待获取事务 TS 阶段花费的时间。'],
         '字段定义加上题目给出的 25ms/30ms 可推断取 TS 等待占主导；不额外声称该段解释了 TS 预取机制。'),
        ('EV09', 'answer', 'identify-slow-queries.md',
         ['`Cop_wait_avg`：cop-task 的平均等待时间，包括请求排队和获取 snapshot 时间。',
          '`Cop_wait_max`：cop-task 的最大等待时间。', '`Cop_wait_addr`：等待时间最长的 cop-task 所在地址。'],
         '字段定义支持结合题目定位长尾等待所在实例；该段不证明唯一原因是 CPU 忙，也不作为 CPU 故障结论。'),
        ('EV17', 'answer', 'faq/migration-tidb-faq.md',
         ['对于 DELETE 操作 TiDB 的 GC 机制会删除数据，但不会释放空间，而是当后续数据写入 RocksDB 且进行 compact 时对空间重新利用。'],
         '同样解释 DELETE 后 GC 不等于立即回收，后续 compact 才重用空间；不将空间重用说成马上返还操作系统。'),
    ]


REJECTIONS = [
    ('ED01', 'troubleshoot-tidb-cluster.md', 'dmesg 中 TiDB 组件相关的问题', '笼统收集 dmesg 不足以确认是 TiDB OOM；TiKV 被杀也不是该题的服务对象。'),
    ('ED03', 'troubleshoot-lock-conflicts.md', '少数热点 key 造成锁排队', '只按 key 聚合的片段不一定包含阻塞双方事务 ID，不能机械视作完整字段解释。'),
    ('ED06', 'faq/sql-faq.md', '设置该属性后，行 ID 会被打散并写入多个 Region', '仅介绍打散写入，缺少对旧数据是否生效这一必答条件。'),
    ('ED07', 'troubleshoot-tidb-cluster.md', '检查 tikv-server 机器和 pd-server 对应端口之间的连通性', '这是 TiKV 到 PD，问题问 TiDB 入口到 PD，不能混为相同的网络边。'),
    ('EV03', 'faq/sql-faq.md', 'defaultFetchSize=-2147483648', '连接串示例出现同名参数，但没有解释客户端缓存／OOM，不能单凭字符串算完整依据。'),
    ('EV06', 'tidb-troubleshooting-map.md', 'TTL 默认时间为 3 秒', '这是特定小事务乐观锁 TTL，不是本题悲观长事务默认 1 小时限制。'),
    ('EV17', 'faq/sql-faq.md', 'tidb_gc_life_time', '只说明 GC 保留时间的片段未必回答 DELETE 后还需 compaction，不因同一文档就自动纳入。'),
    ('EV18', 'troubleshoot-cpu-issues.md', '创建索引在扫表回填索引的时候会消耗大量资源', '覆盖回填与资源影响，但没有同时说明在线 DDL 不阻塞 DML，不能单独覆盖整题。'),
    ('EV19', 'faq/manage-cluster-faq.md', 'PD 启动报错：etcd cluster ID mismatch', 'PD 的 etcd 成员归属错误与本题旧 TiKV 数据连到新 PD 的错误不同。'),
    ('EV19', 'troubleshoot-tidb-cluster.md', '集群 ID 是由 pd-server 在集群初始化时随机分配', '可解释重建后集群 ID 变化，但未完整说明 TiKV 本地 ID 的启动检查；保留为部分相关，不自动放宽整题标签。'),
    ('EV21', 'tidb-troubleshooting-map.md', 'TiDB Data Migration (DM)', '只解释迁移工具，缺少读写流量分阶段切换及稳定 LB 入口，不能独自回答完整问题。'),
    ('EV23', 'faq/sql-faq.md', 'AUTO_RANDOM', '只介绍缓解热点，缺少主键不再递增和取得新主键的方法。'),
    ('EV24', 'troubleshoot-lock-conflicts.md', 'autocommit 事务仍然会优先尝试使用乐观事务模式', '这是自动提交的策略，不是显式 BEGIN 与系统变量的优先级。'),
]


def main():
    old_dir = ROOT / 'evaluation/datasets/enterprise_tidb_v2_review1'
    parent_path = old_dir / 'scoring_contract_v1.json'
    parent_bytes = parent_path.read_bytes()
    if hashlib.sha256(parent_bytes).hexdigest() != PARENT_HASH:
        raise ValueError('Parent scoring contract changed')
    parent = json.loads(parent_bytes)
    draft_path = ROOT / 'evaluation/datasets/enterprise_tidb_v2_draft/questions_draft.json'
    draft_bytes = draft_path.read_bytes()
    if hashlib.sha256(draft_bytes).hexdigest() != parent['parent_draft_sha256']:
        raise ValueError('Original questions changed')
    draft = json.loads(draft_bytes)
    corpus = ROOT / 'evaluation/corpus/enterprise_tidb_v2'
    manifest_path = corpus / 'source_manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != draft['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    manifest = json.loads(manifest_bytes)
    texts, chunks = {}, []
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline source audit network call')):
        for source in manifest['files']:
            data = (corpus / source['local_path']).read_bytes()
            if hashlib.sha256(data).hexdigest() != source['sha256']:
                raise ValueError(f"Source changed: {source['local_path']}")
            if source['category'] == '出处与许可':
                continue
            texts[source['local_path']] = data.decode('utf-8')
            doc = load_document(corpus / source['local_path'])
            index = 0
            for section in doc.sections:
                for chunk in _split_section(text=section.text, file_type=doc.file_type,
                                            section_metadata=section.metadata)[1]:
                    chunks.append({'chunk_id': f"offline:{source['local_path']}:{index}",
                                   'corpus_path': source['local_path'], 'content': chunk.content})
                    index += 1
    if set(SEARCHES) != {q['id'] for q in draft['questions']}:
        raise ValueError('Source review search does not cover every question')
    search_audit = []
    for qid, pattern in SEARCHES.items():
        matches = []
        for filename, text in texts.items():
            for line_number, line in enumerate(text.splitlines(), 1):
                if re.search(pattern, line, re.I):
                    matches.append({'filename': filename, 'line': line_number, 'text': line})
        search_audit.append({'id': qid, 'pattern': pattern, 'matched_lines': matches})
    child = copy.deepcopy(parent)
    child['contract_id'] = 'enterprise_tidb_v2_evidence_v2'
    child['parent_contract_sha256'] = PARENT_HASH
    child['status'] = 'known_equivalent_sources_reviewed_before_retrieval; actual_index_mapping_pending'
    child['mapping_policy'] = '本版本纳入来源扫描和 AI 复核确认的等价表述；正式检索前映射真实 ID 并冻结。保留已排除的部分相关／异条件案例，不保证搜索已穷尽所有语义等价表述。'
    by_id = {rule['id']: rule for rule in child['questions']}
    accepted = []
    for qid, unit_id, filename, quotes, reason in additions():
        rule = by_id[qid]
        target = next(unit for unit in rule['required_units'] if unit['id'] == unit_id)
        anchors = []
        for quote in quotes:
            start = texts[filename].find(quote)
            if start < 0:
                raise ValueError(f'Missing equivalent source: {qid} {filename}: {quote}')
            anchors.append({'quote': quote, 'start_char': start, 'end_char': start+len(quote),
                            'line': texts[filename].count('\n', 0, start)+1})
        option = {'filename': filename, 'quotes': quotes}
        if option in target['options']:
            raise ValueError(f'Duplicate equivalent option: {qid}')
        target['options'].append(option)
        accepted.append({'id': qid, 'unit': unit_id, 'filename': filename, 'anchors': anchors, 'reason': reason})
    for before, after in zip(parent['questions'], child['questions'], strict=True):
        for role in ('required_units', 'optional_units'):
            if [(unit['id'], unit['description']) for unit in before[role]] != [(unit['id'], unit['description']) for unit in after[role]]:
                raise ValueError('Required/optional semantics changed')
    mappings = [compile_mapping(rule, chunks) for rule in child['questions']]
    if any(mapping['missing_required_units'] for mapping in mappings):
        raise ValueError('Required source candidates missing')
    # 每个新增选项都必须可落到至少一个实际离线分块，不能只在整篇文档中拼字。
    for entry in accepted:
        rule = {'id': entry['id'], 'required_units': [{'id': entry['unit'], 'options': [
            {'filename': entry['filename'], 'quotes': [anchor['quote'] for anchor in entry['anchors']]}]}], 'optional_units': []}
        mapping = compile_mapping(rule, chunks)
        if mapping['missing_required_units']:
            raise ValueError(f"New equivalent option is not supported by one chunk: {entry['id']} {entry['filename']}")
        entry['offline_candidate_ids'] = mapping['required_unit_gold_chunk_ids'][entry['unit']]
    old_mapping = json.loads((old_dir/'offline_mapping_review.json').read_text(encoding='utf-8'))
    old_by_id = {m['id']:m for m in old_mapping['mappings']}
    changed = []
    for mapping in mappings:
        old = old_by_id[mapping['id']]
        if old['required_unit_gold_chunk_ids'] != mapping['required_unit_gold_chunk_ids']:
            changed.append(mapping['id'])
    destination = ROOT / 'evaluation/datasets/enterprise_tidb_v2_review2'
    destination.mkdir(parents=True, exist_ok=True)
    write_once(destination/'scoring_contract_v2.json', child)
    contract_hash = hashlib.sha256((destination/'scoring_contract_v2.json').read_bytes()).hexdigest()
    (destination/'scoring_contract_v2.sha256').write_text(contract_hash+'\n', encoding='ascii')
    report = {
        'scope': 'source-only AI equivalence review; no ranked retrieval, model call or database write',
        'scoring_contract_sha256': contract_hash, 'parent_contract_sha256': PARENT_HASH,
        'parent_draft_sha256': parent['parent_draft_sha256'], 'source_integrity_checks': len(manifest['files']),
        'reviewed_questions': len(search_audit), 'source_documents': len(texts), 'offline_chunks': len(chunks),
        'added_options': len(accepted), 'questions_with_added_options': sorted({entry['id'] for entry in accepted}),
        'questions_with_more_gold_candidates': changed, 'required_units': 34, 'optional_units': 1,
        'single_chunk_full_evidence_questions': sum(bool(m['single_chunk_full_evidence_ids']) for m in mappings),
        'multiple_chunks_required_by_current_mapping': [m['id'] for m in mappings if not m['single_chunk_full_evidence_ids']],
        'questions_answers_and_metric_semantics_changed': False,
        'accepted': accepted,
        'rejected_partial_or_different_context': [dict(zip(('id','filename','signal','reason'), row)) for row in REJECTIONS],
        'review_limit': '全部 32 题都有固定关键词源文扫描；AI 复核找到的候选，未做独立外部标注，不宣称穷尽所有措辞。',
        'actual_index_mapping': 'pending', 'retrieval_calls': 0, 'paid_model_calls': 0,
    }
    write_once(destination/'source_equivalence_review.json', report)
    write_once(destination/'source_search_audit.json', {'searches': search_audit, 'scope': 'unranked source text search, not benchmark retrieval'})
    write_once(destination/'offline_mapping_review.json', {'scope': 'offline candidate IDs, not indexed database IDs',
        'contract_sha256': contract_hash, 'mappings': mappings})
    grouped = defaultdict(list)
    for entry in accepted:
        grouped[entry['id']].append(f"{entry['unit']}: {entry['filename']}（{entry['reason']}）")
    with (destination/'question_equivalence_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id','split','question','reference_answer','extra_accepted_sources','review_status'])
        for q in draft['questions']:
            writer.writerow([q['id'], q['split'], q['question'], q['reference_answer'], '\n'.join(grouped[q['id']]),
                             '有新增选项' if q['id'] in grouped else '本次源文搜索未补充新的完整依据选项'])
    assert parent_path.read_bytes() == parent_bytes and draft_path.read_bytes() == draft_bytes
    print(json.dumps({key:value for key,value in report.items() if key not in ('accepted','rejected_partial_or_different_context')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
