"""从冻结原文建立待复核题集和依据标签；不执行任何检索或模型调用。"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'evaluation/artifacts/baseline_20261123_120506/snapshot.json'
CORPUS = ROOT / 'evaluation/corpus/tidb'
DESTINATION = ROOT / 'evaluation/datasets/tidb_v1_draft'
E = 'error-codes.md'
M = 'configure-memory-usage.md'
T = 'transaction-isolation-levels.md'


def question(qid, split, topic, kind, text, answer, filename, quotes, *, code=None):
    return dict(id=qid, split=split, topic=topic, kind=kind, question=text,
                reference_answer=answer, filename=filename, quotes=quotes, error_code=code)


def error(qid, split, kind, code, text, answer, *quotes):
    return question(qid, split, '错误码', kind, text, answer, E, list(quotes), code=code)


def rows():
    return [
        error('D01', 'dev', '精确标识', 8001, 'TiDB 报错 8001，想增大单条 SQL 的内存上限，应调整哪个变量？',
              '调整 tidb_mem_quota_query。', '请求使用的内存超过 TiDB 内存使用的阈值限制。', 'tidb_mem_quota_query'),
        error('D02', 'dev', '精确标识', 8005, 'TiDB 报错 8005：Write Conflict, txnStartTS is stale，文档建议怎么处理？',
              '发生写入冲突；检查业务逻辑并重试写入操作。', '事务在 TiDB 中遇到了写入冲突。请检查业务逻辑，重试写入操作。'),
        error('D03', 'dev', '数值与边界', 8025, 'TiDB 报错 8025 时，单个键值对默认大小上限是多少，放宽上限用哪个配置？',
              '默认最大 6MB；可适当调整 txn-entry-size-limit。', 'TiDB 默认支持最大 6MB 的单个键值对', 'txn-entry-size-limit'),
        question('D04', 'dev', '内存控制', '语义场景', '希望 SQL 超过内存限额后只记日志、继续运行，tidb_mem_oom_action 应设成什么？',
                 '设为 LOG。', M, ['如果变量值为 `LOG`', '这条 SQL 会继续执行，但 TiDB 会在 log 文件中打印一条 LOG。']),
        question('D05', 'dev', '内存控制', '精确标识', 'tidb_mem_oom_action=CANCEL 时，SQL 超过内存阈值会发生什么？',
                 'TiDB 立即中断该 SQL，并向客户端返回含各物理执行算子内存使用情况的错误。', M,
                 ['如果变量值为 `CANCEL`', 'TiDB 会立即中断这条 SQL 的执行', '各个物理执行算子的内存使用情况']),
        question('D06', 'dev', '内存控制', '数值与边界', '文档中把单条 SQL 的内存阈值设成 8MB 的 SQL 是什么？',
                 'SET tidb_mem_quota_query = 8 << 20;', M, ['SET tidb_mem_quota_query = 8 << 20;']),
        question('D07', 'dev', '事务隔离', '精确标识', '如何用 SQL 将当前会话的事务隔离级别改为 READ-COMMITTED？',
                 "SET SESSION transaction_isolation = 'READ-COMMITTED';", T,
                 ["SET SESSION transaction_isolation = 'READ-COMMITTED';"]),
        question('D08', 'dev', '事务隔离', '数值与边界', '这份文档列出的 SQL-92 标准隔离级别共有几种？',
                 '四种。', T, ['SQL-92 标准定义了 4 种隔离级别']),

        error('E01', 'test', '精确标识', 8048, '第三方框架设置了 TiDB 不支持的隔离级别，出现 8048 且不便改代码，可用哪个变量绕过检查？',
              '可考虑 tidb_skip_isolation_level_check；文档示例将其设为 1。', 'tidb_skip_isolation_level_check', '来绕过这一检查'),
        error('E02', 'test', '精确标识', 8055, 'TiDB 报错 8055，快照过旧可能被 GC，文档建议调大哪个系统变量？',
              '调大 tidb_gc_life_time。', '当前快照过旧，数据可能已经被 GC。', 'tidb_gc_life_time'),
        error('E03', 'test', '数值与边界', 8059, 'AUTO_RANDOM 次数耗尽报错 8059 后，文档是否提供了恢复方法？',
              '没有，文档明确称当前没有恢复这类错误的方法。', '当前没有恢复这类错误的方法。'),
        error('E04', 'test', '精确标识', 8060, '错误 8060 表示自增列偏移量非法，需要检查哪两个变量？',
              'auto_increment_increment 和 auto_increment_offset。', 'auto_increment_increment', 'auto_increment_offset'),
        error('E05', 'test', '精确标识', 8110, 'TiDB 报错 8110 不能进行笛卡尔积运算，文档要求修改哪个配置及其值？',
              '将配置文件中的 cross-join 设为 true。', '不能进行笛卡尔积运算', '`cross-join` 设置为 `true`'),
        error('E06', 'test', '精确标识', 8120, 'TiDB 的 8120 错误获取不到事务 start tso，文档要求检查哪个服务及哪段网络？',
              '检查 PD Server 的状态、监控、日志及 TiDB Server 与 PD Server 之间的网络。',
              '获取不到事务的 start tso，请检查 PD Server 状态/监控/日志以及 TiDB Server 与 PD Server 之间的网络。'),
        error('E07', 'test', '精确标识', 8123, '带聚合函数的 SQL 返回非聚合列并报错 8123，是违反了哪种模式？',
              'only_full_group_by 模式。', '违反了 `only_full_group_by` 模式'),
        error('E08', 'test', '语义场景', 8130, '升级 TiDB 后出现 8130、同一次 COM_QUERY 不能执行多个查询，文档说默认禁止这样做的安全原因是什么？',
              '为了减少 SQL 注入攻击的影响。', '为了减少 SQL 注入攻击的影响', '同一 `COM_QUERY` 调用中执行多个查询'),
        error('E09', 'test', '精确标识', 8154, 'LOAD DATA 报错 8154、不支持从 TiDB 服务器本地导入时，可怎样更换数据来源？',
              '指定 LOCAL 从客户端导入，或将数据上传至 S3/GCS 后导入。', '可以指定 `LOCAL` 从客户端导入，或者将数据上传到 S3/GCS 再进行导入。'),
        error('E10', 'test', '精确标识', 8156, 'TiDB 导入时遇到错误 8156，需要修正什么？',
              '文件路径不能为空，需要设置正确的路径再导入。', '传入的文件路径不能为空。需要设置正确的路径再进行导入。'),
        error('E11', 'test', '语义场景', 8229, '一个事务存活过久导致 TiDB 返回 8229，文档给出的处理方式是什么？',
              '提交或回滚当前事务，然后开启新事务。', '事务超过存活时间，遇到该问题可以提交或者回滚当前事务，开启一个新事务。'),
        error('E12', 'test', '精确标识', 8249, 'TiDB 错误 8249 对应资源组不存在还是资源组已存在？',
              '资源组不存在，修改或绑定不存在的资源组会报该错。', '资源组不存在。在修改或绑定不存在的资源组时返回该错误。'),
        error('E13', 'test', '精确标识', 8250, '使用资源管控时报错 8250，应该打开哪个全局变量？',
              '打开 tidb_enable_resource_control。', 'tidb_enable_resource_control', '启用资源管控'),
        error('E14', 'test', '语义场景', 8252, '资源组配额超限报错 8252，文档建议从事务规模和客户端并发两个方面怎样调整？',
              '调整事务大小或减少客户端并发数。', '需调整事务大小或减少客户端并发数。'),
        error('E15', 'test', '精确标识', 9002, 'TiDB 错误 9002 是请求哪个服务超时，应检查哪两个服务之间的网络？',
              '请求 TiKV 超时，检查 TiDB Server 与 TiKV Server 之间的网络，并检查 TiKV 状态、监控和日志。',
              '请求 TiKV 超时，请检查 TiKV Server 状态/监控/日志以及 TiDB Server 与 TiKV Server 之间的网络。'),
        error('E16', 'test', '精确标识', 9012, 'TiDB 错误 9012 是哪个服务请求超时，应检查哪段网络？',
              '请求 TiFlash 超时，检查 TiDB Server 与 TiFlash Server 之间的网络，并检查 TiFlash 状态、监控和日志。',
              '请求 TiFlash 超时。请检查 TiFlash Server 状态/监控/日志以及 TiDB Server 与 TiFlash Server 之间的网络。'),

        question('M01', 'test', '内存控制', '语义场景', 'tidb-server 实例触及内存总量上限时，会优先终止正在执行的哪类 SQL？',
                 '优先依次终止内存用量最大的 SQL，直到实例内存下降到阈值以下。', M,
                 ['TiDB 会依次终止正在执行的 SQL 操作中内存用量最大的 SQL 操作']),
        question('M02', 'test', '内存控制', '数值与边界', 'tidb_server_memory_limit 的内存限制会不会终止 DDL 操作？',
                 '不会；DDL 在文档列出的不终止操作中。', M,
                 ['当前 `tidb_server_memory_limit` 所设的内存限制**不终止**以下 SQL 操作：', '- DDL 操作']),
        question('M03', 'test', '内存控制', '语义场景', '同一台机器混合部署多个实例时，TiDB 的实例内存限制约束单个实例还是整台机器？',
                 '约束单个 tidb-server 实例，不是整台物理机的总内存。', M,
                 ['为单个 tidb-server 实例的内存使用阈值，而不是整个物理机的总内存阈值。']),
        question('M04', 'test', '内存控制', '数值与边界', '为缓解实例内存压力主动触发的 Golang GC，文档限制其每分钟最多触发几次？',
                 '每分钟最多一次。', M, ['该 GC 方式 1 分钟最多只会触发 1 次。']),
        question('M05', 'test', '内存控制', '语义场景', '想查看当前实例或整个集群的内存使用情况，应查询哪个 INFORMATION_SCHEMA 系统表？',
                 'INFORMATION_SCHEMA.(CLUSTER_)MEMORY_USAGE。', M,
                 ['要查看当前实例或集群的内存使用情况', 'INFORMATION_SCHEMA.(CLUSTER_)MEMORY_USAGE`']),
        question('M06', 'test', '内存控制', '数值与边界', 'MEMORY_USAGE_OPS_HISTORY 对每个实例保留多少条最近的内存操作记录？',
                 '最近 50 条记录。', M,
                 ['INFORMATION_SCHEMA.(CLUSTER_)MEMORY_USAGE_OPS_HISTORY', '对于每个实例，该表保留最近 50 条记录。']),
        question('M07', 'test', '内存控制', '数值与边界', 'tidb-server 内存占用过高报警的默认比例阈值是多少？',
                 '总内存量的 70%；实际报警还要满足文档列出的触发条件之一。', M,
                 ['默认为总内存量的 70%', '且满足以下任一条件时']),
        question('M08', 'test', '内存控制', '数值与边界', 'TiDB 默认保留最近几次内存报警产生的状态文件？',
                 '最近 5 次。', M, ['目前 TiDB 默认只保留最近 5 次报警时所生成的状态文件。']),
        question('M09', 'test', '内存控制', '语义场景', '想从日志中定位 tidb-server 内存过高的风险报警，应搜索什么英文关键词？',
                 'tidb-server has the risk of OOM。', M,
                 ['TiDB 将输出一条包含关键字 `tidb-server has the risk of OOM`']),
        question('M10', 'test', '内存控制', '数值与边界', '内存报警时记录 SQL 状态，内存用量最高和运行时间最长的语句各记录多少条？',
                 '各 10 条。', M,
                 ['当前正在执行的所有 SQL 语句中内存使用最高的 10 条语句和运行时间最长的 10 条语句的相关信息']),
        question('M11', 'test', '内存控制', '语义场景', '为什么向 TiDB 写入一个很大的事务，内存可能成为瓶颈？',
                 '事务模型要求所有待提交的事务写入操作先在内存中缓存。', M,
                 ['所有待提交的事务写入操作需先在内存中进行缓存。']),
        question('M12', 'test', '内存控制', '精确标识', '文档建议在满足限制条件的前提下，把 tidb_dml_type 调成什么值来减少大事务内存使用？',
                 '调成 bulk。', M, ['tidb_dml_type', '为 `"bulk"`', '在满足各项限制条件的前提下']),
        question('M13', 'test', '内存控制', '语义场景', '读数据算子的流量控制被触发时，单条 SQL 内存每超过一次配额，会怎样减少读取并发？',
                 '读数据的算子会停止一个线程。', M,
                 ['当单条 SQL 语句的内存使用每超过', '读数据的算子就会停止一个线程。']),
        question('M14', 'test', '内存控制', '精确标识', '日志出现 memory exceeds quota, destroy one token now，说明触发了哪种内存控制行为？',
                 '触发了流量控制。', M,
                 ['当流控被触发时，会在日志中打印一条包含关键字 `memory exceeds quota, destroy one token now` 的日志。']),
        question('M15', 'test', '内存控制', '数值与边界', 'HashAgg 在使用 distinct 聚合函数且内存过大时，能否通过落盘缓解？',
                 '不能；文档指出 HashAgg 落盘目前不支持 distinct 聚合函数。', M,
                 ['HashAgg 落盘功能目前不支持 distinct 聚合函数。使用 distinct 函数且内存占用过大时，无法进行落盘。']),
        question('M16', 'test', '内存控制', '数值与边界', '并行 HashAgg 落盘从哪个版本开始成为正式功能 GA？',
                 'v8.2.0。', M, ['并行 HashAgg 的落盘在 v8.0.0 版本以实验特性引入，在 v8.2.0 版本成为正式功能 (GA)。']),

        question('T01', 'test', '事务隔离', '精确标识', 'TiDB 为兼容 MySQL 而称为 REPEATABLE READ 的一致性，实际实现的隔离机制是什么？',
                 '快照隔离 Snapshot Isolation，简称 SI。', T,
                 ['TiDB 实现了快照隔离 (Snapshot Isolation, SI) 级别的一致性。', '又称其为“可重复读” (REPEATABLE READ)']),
        question('T02', 'test', '事务隔离', '数值与边界', '在 TiDB 乐观事务模式下设置 Read Committed 会生效吗，实际仍用什么隔离级别？',
                 '不会生效，仍使用可重复读；Read Committed 仅在悲观事务模式下生效。', T,
                 ['在[乐观事务模式](/optimistic-transaction.md)下设置事务隔离级别为 Read Committed 将不会生效，事务将仍旧使用可重复读隔离级别。']),
        question('T03', 'test', '事务隔离', '精确标识', '开启 tidb_rc_read_check_ts 后，SELECT 尝试复用的有效时间戳初始值是什么？',
                 '当前事务的 start_ts。', T,
                 ['tidb_rc_read_check_ts', '初始值为事务的 `start_ts`']),
        question('T04', 'test', '事务隔离', '精确标识', 'tidb_rc_write_check_ts 当前适用哪三种类型的点写语句？',
                 'UPDATE、DELETE、SELECT ... FOR UPDATE。', T,
                 ['tidb_rc_write_check_ts', '目前该变量适用的点写语句包括 `UPDATE`、`DELETE`、`SELECT ...... FOR UPDATE` 三种类型。']),
        question('T05', 'test', '事务隔离', '语义场景', 'MySQL 读已提交中的半一致性读特例，TiDB 是否兼容？',
                 '不兼容该特殊行为。', T,
                 ['半一致性读', 'TiDB 没有兼容这个特殊行为。']),
        question('T06', 'test', '事务隔离', '数值与边界', 'TiDB 的快照隔离 SI 能否避免写偏斜？',
                 '不能，SI 仍会出现写偏斜。', T, ['SI 还会出现写偏斜']),
        question('T07', 'test', '事务隔离', '语义场景', '在可重复读的快照读语义下，别的事务在当前事务启动后才提交的数据，当前事务是否可见？',
                 '不可见；这里只讨论文档该段描述的快照读语义，不包含悲观事务当前读。', T,
                 ['未提交的数据或在事务启动后其他事务提交的数据是不可见的。']),
        question('T08', 'test', '事务隔离', '精确标识', '查看当前会话事务隔离级别的 SQL 是什么？',
                 "SHOW VARIABLES LIKE 'transaction_isolation';", T,
                 ["SHOW VARIABLES LIKE 'transaction_isolation';"]),
    ]


def normalized(text):
    return re.sub(r'\s+', '', text)


def locate(text, quote, start, end):
    position = text.find(quote, start, end)
    if position < 0:
        raise ValueError(f'Missing source quote: {quote!r}')
    return {'text': quote, 'start_character': position, 'end_character': position + len(quote),
            'line_start': text.count('\n', 0, position) + 1,
            'line_end': text.count('\n', 0, position + len(quote)) + 1}


def build():
    baseline = json.loads(BASELINE.read_text(encoding='utf-8'))
    if not all(baseline['checks'].values()):
        raise ValueError('Baseline checks did not pass')
    sources = {name: (CORPUS / name).read_text(encoding='utf-8') for name in (E, M, T)}
    for item in baseline['corpus']['files']:
        actual = hashlib.sha256((CORPUS / item['filename']).read_bytes()).hexdigest()
        if actual != item['sha256']:
            raise ValueError(f"Source changed: {item['filename']}")
    result = rows()
    if len(result) != 48 or len({item['id'] for item in result}) != 48:
        raise ValueError('Expected 48 unique question IDs')
    if len({item['question'] for item in result}) != 48:
        raise ValueError('Duplicate questions')
    for item in result:
        text = sources[item['filename']]
        start, end = 0, len(text)
        if item['error_code'] is not None:
            marker = f"* Error Number: {item['error_code']}\n"
            start = text.index(marker)
            following = text.find('\n* Error Number:', start + len(marker))
            end = following if following >= 0 else len(text)
            item['source_context'] = locate(text, f"Error Number: {item['error_code']}", start, end)
        item['evidence'] = [locate(text, quote, start, end) for quote in item.pop('quotes')]
        for evidence in item['evidence']:
            evidence['filename'] = item['filename']
        item['evidence_options'] = [item['evidence']]
        if item['id'] == 'E13':
            # 报错字符串本身已给出修复命令，不应只认可后面的中文解释。
            alternative = locate(text, "SET GLOBAL tidb_enable_resource_control='on'", start, end)
            alternative['filename'] = E
            item['evidence_options'].append([alternative])
        if item['id'] == 'D01':
            # 同一答案还见于内存文档，不能只认可错误码文档中的原始参考段。
            for quote in (
                '当一条 SQL 的内存使用超过一定阈值（由 session 变量 `tidb_mem_quota_query` 控制）后',
                '使用系统变量 [`tidb_mem_quota_query`](/system-variables.md#tidb_mem_quota_query) 来配置一条 SQL 执行过程中的内存使用阈值，单位为字节。',
            ):
                alternative = locate(sources[M], quote, 0, len(sources[M]))
                alternative['filename'] = M
                item['evidence_options'].append([alternative])
        candidates = [chunk for chunk in baseline['chunks']
                      if chunk['metadata_json'].get('filename') in
                      {e['filename'] for option in item['evidence_options'] for e in option}]
        matches = []
        for chunk in candidates:
            if any(all(e['filename'] == chunk['metadata_json'].get('filename')
                       and normalized(e['text']) in normalized(chunk['content']) for e in option)
                   for option in item['evidence_options']):
                matches.append({'chunk_id': chunk['id'], 'chunk_index': chunk['chunk_index'],
                                'filename': chunk['metadata_json'].get('filename')})
        item['baseline_full_evidence_chunks'] = matches
        item['baseline_partial_evidence_chunks'] = [
            {'chunk_id': chunk['id'], 'chunk_index': chunk['chunk_index']}
            for chunk in candidates
            if any(normalized(e['text']) in normalized(chunk['content']) for e in item['evidence'])
        ]
        if item['error_code'] is not None:
            context = item['source_context']['text']
            item['context_separated_from_primary_evidence'] = not any(
                normalized(context) in normalized(chunk['content'])
                and all(e['filename'] == chunk['metadata_json'].get('filename')
                        and normalized(e['text']) in normalized(chunk['content']) for e in item['evidence'])
                for chunk in candidates)
        item['review_status'] = 'ai_draft_pending_user_review'
    counts = Counter(item['split'] for item in result)
    if counts != {'dev': 8, 'test': 40}:
        raise ValueError('Wrong dev/test allocation')
    summary = {
        'status': 'draft_not_frozen', 'author': 'AI; pending user review',
        'label_revision': 'dev_review_1',
        'label_review_note': 'D01 accepts equivalent memory-control evidence; raw retrieval results preserved and rescored offline',
        'source_commit': baseline['corpus']['commit'],
        'source_files': baseline['corpus']['files'],
        'baseline_snapshot': BASELINE.relative_to(ROOT).as_posix(),
        'baseline_snapshot_sha256': hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        'counts': dict(counts),
        'test_topics': dict(Counter(item['topic'] for item in result if item['split'] == 'test')),
        'test_kinds': dict(Counter(item['kind'] for item in result if item['split'] == 'test')),
        'without_full_evidence_chunk': [item['id'] for item in result if not item['baseline_full_evidence_chunks']],
        'context_separated_from_primary_evidence': [item['id'] for item in result
                                                    if item.get('context_separated_from_primary_evidence')],
        'retrieval_executed': False, 'model_api_calls': 0,
        'scoring_contract': {
            'metric': 'Hit Rate@5', 'answerable_test_questions': 40,
            'hit': 'At least one of the first 5 chunks contains a complete, reviewed evidence option',
            'partial_evidence': 'A code alone, or incomplete answer evidence, is not a full hit; answer-bearing source evidence can be valid without repeating the code',
            'mapping': 'Whitespace-insensitive exact quote matching within source document; draft labels need semantic review',
            'zero_full_evidence': 'Retain in denominator and audit source/chunk boundary; never discard after seeing retrieval results',
            'review': 'Accept equivalent complete evidence even when worded differently; audit before scoring',
            'dev_test_limit': 'Same three documents; claim-level split, not an independent document holdout',
            'unanswerable_questions': 'Not included in Hit Rate@5; add a separate refusal test later',
            'after_freeze': 'Preserve question text and source facts; remap source evidence when chunking changes',
            'exceptions': 'Keep failed and empty retrievals; distinguish execution errors from retrieval misses',
        },
        'questions': result,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / 'questions_draft.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (DESTINATION / 'questions_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['题号', '用途', '主题', '题型', '问题', '参考答案', '来源文件',
                         '原文行号', '原文依据', '当前完整依据块', '待复核项', '本人复核意见'])
        for item in result:
            matches = ', '.join(f"{match['filename']} #{match['chunk_index']}" for match in item['baseline_full_evidence_chunks'])
            writer.writerow([item['id'], '开发' if item['split'] == 'dev' else '正式候选',
                             item['topic'], item['kind'], item['question'], item['reference_answer'],
                             item['filename'], '; '.join(f"{e['line_start']}-{e['line_end']}" for e in item['evidence']),
                             '\n或者\n'.join('\n'.join(f"[{e['filename']}:{e['line_start']}] {e['text']}" for e in option)
                                            for option in item['evidence_options']), matches,
                             '核对跨块或等价依据' if not matches else '核对问题与答案是否相符', ''])
    print(json.dumps({key: summary[key] for key in ('status', 'counts', 'test_topics', 'test_kinds',
                                                  'without_full_evidence_chunk', 'context_separated_from_primary_evidence',
                                                  'retrieval_executed', 'model_api_calls')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
