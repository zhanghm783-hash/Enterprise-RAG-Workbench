"""企业运维场景题与原文依据草稿；不运行检索、入库或模型。"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / 'evaluation/corpus/enterprise_tidb_v2'
DESTINATION = ROOT / 'evaluation/datasets/enterprise_tidb_v2_draft'


def item(qid, split, topic, family, kind, question, answer, filename, quotes, signals=(), condition='固定语料版本中的排查说明'):
    return dict(id=qid, split=split, topic=topic, answer_family=family, kind=kind,
                question=question, reference_answer=answer, filename=filename, quotes=quotes,
                exact_signals=list(signals), applicability=condition)


def rows():
    oom = 'troubleshoot-tidb-oom.md'
    lock = 'troubleshoot-lock-conflicts.md'
    slow = 'analyze-slow-queries.md'
    cluster = 'troubleshoot-tidb-cluster.md'
    deploy = 'faq/deploy-and-maintain-faq.md'
    sql = 'faq/sql-faq.md'
    return [
        item('ED01', 'dev', '内存故障', 'confirm_server_oom', '现象与精确标识',
             '查询报 2013，Lost connection to MySQL server during query；内存监控持续升高后归零。怎样进一步确认是不是 OOM？',
             '在故障时间附近检查操作系统日志，可用 dmesg -T | grep tidb-server；存在 oom-killer 记录可确认 OOM。断连本身不足以确定原因。',
             oom, ['如果故障发生的时间点附近存在 `oom-killer` 的日志，则可以确定是 OOM 问题。',
                   'dmesg -T | grep tidb-server'], ['2013', 'Lost connection to MySQL server during query']),
        item('ED02', 'dev', '内存故障', 'prepared_statement_lifecycle', '现象与精确标识',
             '连接池长期复用连接，应用反复创建 PreparedStatement 没有释放，服务端内存一直上涨。应从什么资源生命周期排查，有什么数量限制可以用？',
             '排查预处理语句未释放与 session 生命周期；文档提示 deallocate prepare stmt、调整 session 生命周期或连接池超时，并可用 max_prepared_stmt_count 限制数量。',
             oom, ['客户端不断创建预处理语句但未执行', '调整 session 的生命周期。', '`max_prepared_stmt_count`'], ['PreparedStatement']),
        item('ED03', 'dev', '事务与锁', 'current_blocking_relation', '现象与精确标识',
             '悲观事务报 Lock wait timeout exceeded，业务想知道谁阻塞了谁。哪个系统表提供阻塞双方的事务标识和等待的 key？',
             'information_schema.DATA_LOCK_WAITS 提供悲观锁等待信息，包括阻塞和被阻塞事务的 start_ts、被阻塞 SQL 的 Digest 及等待 key。',
             lock, ['[`DATA_LOCK_WAITS`]', '包括阻塞和被阻塞的事务的 `start_ts`'], ['Lock wait timeout exceeded'], '悲观事务锁等待；Lock View 自 v5.1 起支持'),
        item('ED04', 'dev', '事务与锁', 'write_conflict_object', '现象与精确标识',
             '并发更新时出现 [kv:9007]Write conflict，怎样从 key={tableID=47, indexID=1, ...} 判断冲突发生在哪类对象上？',
             'tableID 是冲突表的 ID，indexID 表示索引数据冲突；如果打印 handle=x，则表示相应行数据发生冲突。',
             'troubleshoot-write-conflicts.md', ['tableID 表示发生冲突的表的 ID，indexID 表示是索引数据发生了冲突。',
                                                        '会打印 `handle=x` 表示对应哪行数据发生了冲突'], ['9007', 'Write conflict', 'tableID', 'indexID'], '乐观事务写写冲突的日志解释'),
        item('ED05', 'dev', '慢查询', 'cop_process_bottleneck', '现象与精确标识',
             '慢日志 Query_time 约 0.18 秒，Cop_process 约 170ms、Cop_wait 约 1ms。这组现象优先指向哪个组件的处理瓶颈？',
             '优先指向 TiKV 处理侧；文档同类例子中一个 cop-task 耗时约 171ms，接近整个查询的 180.85ms。',
             slow, ['查询整个耗时是 180.85ms 左右', '瓶颈在 TiKV 侧。'], ['Query_time', 'Cop_process', 'Cop_wait']),
        item('ED06', 'dev', '写入热点', 'implicit_rowid_sharding', '口语现象',
             '一张没有主键的表持续插入新行，流量总压在同一个数据分片。有什么办法打散后续写入，改完会重新分布以前的数据吗？',
             '可考虑 SHARD_ROW_ID_BITS 打散隐式 RowID，使写入分散到不同 Region；动态修改只对新写入数据生效，不会重排已有数据。',
             'troubleshoot-hot-spot-issues.md', ['通过设置 [`SHARD_ROW_ID_BITS`]', '每次修改之后，只对新写入的数据生效。'], condition='无主键或 NONCLUSTERED 主键表；不适用于 CLUSTERED 主键'),
        item('ED07', 'dev', '连接与启动', 'tidb_to_pd_startup_connectivity', '口语现象',
             '数据库入口进程启动失败，说连不上调度服务，但调度服务进程还在。接下来怎样区分服务端口和网络连通性问题？',
             '确认 pd-server 日志及端口已打开；从 tidb-server 所在机器检查到 PD 端口的连通性与防火墙白名单，可用 nc 或 curl 检查。',
             cluster, ['首先检查 pd-server 的进程状态和日志，确保 pd-server 成功启动',
                       '检查 tidb-server 机器和 pd-server 对应端口之间的连通性', '可通过 nc 或 curl 工具检查。']),
        item('ED08', 'dev', '容量与流控', 'server_busy_meaning', '现象与精确标识',
             '高峰期客户端反复收到 server is busy，是否仅凭这一句就能认定进程崩了？首先去哪里确认具体压力原因？',
             '不能；这是 TiKV 告知压力过大的流控信号，应查看 Grafana 的 TiKV/errors 监控确认具体 busy 原因，再区分 write stall、scheduler、raftstore、Coprocessor 等情况。',
             'tidb-troubleshooting-map.md', ['**Grafana** -> **TiKV** -> **errors** 确认具体 busy 原因。',
                                            '`server is busy` 是 TiKV 自身的流控机制'], ['server is busy']),

        item('EV01', 'test', '内存故障', 'hashagg_memory_alternative', '现象与精确标识',
             '内存排查发现执行计划中的 HashAgg 占用很高，在聚合执行方式上可以尝试什么替代方案？',
             '可以尝试使用 STREAM_AGG() 替代；文档指出 HashAgg 多线程执行较快，但会消耗较多内存。',
             oom, ['HashAgg 是多线程并发执行', '可以尝试使用 `STREAM_AGG()` 替代。'], ['HashAgg']),
        item('EV02', 'test', '内存故障', 'large_transaction_amplification', '口语现象',
             '批量写入时，进程需要的内存明显大于准备提交的数据大小。容量规划应考虑什么放大现象，单个事务太大时怎样缓解？',
             '文档指出事务执行内存相对提交事务大小有放大，最大可能达到 2 到 3 倍以上；单个大事务可拆分以调小事务大小。',
             oom, ['最大可能达到提交事务大小的 2 到 3 倍以上。', '通过拆分的方式调小事务大小。']),
        item('EV03', 'test', '内存故障', 'client_result_buffering', '口语现象',
             '应用拉取查询结果时，内存爆掉的是客户端，而不是数据库进程。除了看数据流量，驱动的哪类读取设置值得检查？',
             '检查 JDBC 流式读取相关配置，例如 defaultFetchSize 配置错误可能导致数据在客户端大量缓存；结合 Client Data Traffic 观察是否存在网络阻塞。',
             oom, ['流式读取的相关参数 `defaultFetchSize` 配置有误，会造成数据在客户端大量缓存。']),
        item('EV04', 'test', '事务与锁', 'lock_view_sql_visibility', '现象与精确标识',
             '排障同事想用 Lock View 查看所有事务模式的锁冲突，并还原带原始参数的完整 SQL。这两个预期都成立吗？',
             '不成立。文档说明 Lock View 目前只提供悲观锁冲突和等待信息；SQL 由 Digest 获得，是去掉格式和参数的归一化语句，不能恢复含原始格式和参数的完整 SQL。',
             lock, ['Lock View 功能目前仅提供悲观锁的冲突和等待信息。', '因而无法获取包括格式和参数在内的完整语句。'], ['Lock View'], '该固定版本文档描述的 Lock View 范围'),
        item('EV05', 'test', '事务与锁', 'fair_locking_tradeoff', '现象与精确标识',
             '业务大量并发争抢同一行，频繁出现 pessimistic lock retry limit reached。可以考虑哪个公平加锁选项，代价是什么？',
             '先从业务逻辑调整；同一 key 高并发冲突时可考虑 tidb_pessimistic_txn_fair_locking，但可能降低冲突事务吞吐、增加平均延迟。',
             lock, ['tidb_pessimistic_txn_fair_locking', '吞吐下降（平均延迟上升）的代价。'], ['pessimistic lock retry limit reached'], '悲观事务同一 key 高并发；该变量自 v7.0.0 引入'),
        item('EV06', 'test', '事务与锁', 'pessimistic_ttl_timeout', '现象与精确标识',
             '后台长事务运行很久后提示 TTL manager has timed out, pessimistic locks may expire。应检查什么时间限制，业务上优先考虑怎样调整？',
             '检查悲观锁 TTL 上限及 performance.max-txn-ttl；文档默认上限为 1 小时，超时可能提交失败。可优化业务、拆分大事务，并评估相关参数。',
             lock, ['悲观锁的 TTL 有上限，默认为 1 小时', 'performance.max-txn-ttl', '如将大事务拆分为小事务。'], ['TTL manager has timed out'], '悲观事务；实际 TTL 取决于配置并受 GC 时间等条件约束'),
        item('EV07', 'test', '慢查询', 'timestamp_wait_bottleneck', '现象与精确标识',
             '一条查询 Query_time 为 0.030 秒，其中 Wait_TS 为 0.025 秒。这一阶段的等待是否值得重点检查，为什么？',
             '值得重点检查取时间戳阶段；比较 Wait_TS 与 Query_time，文档指出时间戳通常有预取，Wait_TS 应较低，而这里占了大部分查询时间。',
             slow, ['因为 TS 有预取操作，通常来说 `Wait_TS` 应该很低。', '# Query_time: 0.0300000', '# Wait_TS: 0.02500000'], ['Query_time', 'Wait_TS']),
        item('EV08', 'test', '慢查询', 'preexecuted_subquery_latency', '现象与精确标识',
             'SQL 总耗时约 7.77 秒，但 explain analyze 里的执行算子只有几毫秒；慢日志里 Rewrite_time 和 Preproc_subqueries_time 接近 7.76 秒。时间可能花在哪里？',
             '可能是非关联子查询在优化阶段提前执行，执行算子耗时无法直接反映这段时间；文档同类例子有一个提前执行的子查询耗时约 7.76 秒。',
             slow, ['子查询部分可能被提前执行', '这种查询用 `explain analyze` 看不到对应的耗时',
                    '# Rewrite_time: 7.765673663 Preproc_subqueries: 1 Preproc_subqueries_time: 7.765231874'], ['Rewrite_time', 'Preproc_subqueries_time']),
        item('EV09', 'test', '慢查询', 'cop_wait_outlier_instance', '现象与精确标识',
             '慢日志 Cop_wait 平均只有 1ms、P90 为 2ms，但 Max_time 达到 110ms，并给出一个 Max_Addr。这说明应优先排查什么？',
             '可能是该地址对应 TiKV 实例当时繁忙，一个 cop-task 排队等待约 110ms；应结合当时该实例 CPU 监控判断，而不能只看平均等待。',
             slow, ['# Cop_wait: Avg_time: 1ms P90_time: 2ms Max_time: 110ms', '此时可以打开当时的 CPU 监控辅助判断。'], ['Cop_wait', 'Max_time', 'Max_Addr']),
        item('EV10', 'test', '连接与启动', 'kv_grpc_network_gap', '现象与精确标识',
             '监控显示 TiDB 的 KV Cmd Duration 很长，可 TiKV 的 gRPC duration 很短。除了服务端计算慢，还应怀疑哪一段？',
             'TiDB 与 TiKV 之间可能网络延迟较高，或网卡带宽被占满；可通过客户端观察的 KV duration 与服务端 gRPC duration 差异定位。',
             'troubleshoot-cpu-issues.md', ['gRPC duration 很短但是 TiDB 的 KV duration 显示很长', 'TiDB 和 TiKV 之间网络延迟可能很高，或者 TiDB 和 TiKV 之间的网卡带宽被占满。'], ['KV Cmd Duration', 'gRPC duration']),
        item('EV11', 'test', '连接与启动', 'file_descriptor_limit', '准确报错',
             '组件启动时报 Too many open files，启动前应检查哪项操作系统限制？',
             '检查 ulimit -n 的文件描述符数量限制；文档建议足够大，给出的建议值是 unlimited 或大于 1000000。',
             cluster, ['`ulimit -n` 的结果足够大，推荐设为 unlimited 或者是大于 1000000。'], ['Too many open files']),
        item('EV12', 'test', '连接与启动', 'shared_tikv_database_directory', '口语现象',
             '新增存储进程启动失败，提示数据库文件被占用。能否让两个存储实例直接打开同一个数据库文件目录？',
             '不能；文档明确不要在同一个数据库文件目录上打开两个 tikv，应核对实例的数据目录是否冲突。',
             cluster, ['不要在一个数据库文件目录上打开两个 tikv。']),
        item('EV13', 'test', '部署与配置', 'binary_config_loading', '现象与精确标识',
             '二进制部署的 TiKV/PD 改了 toml 却不生效，应先检查哪个启动参数？两个组件后续读取配置的方式有什么区别？',
             '先检查是否用 --config 指定文件；TiKV 修改配置后可重启服务，PD 只在首次启动读取文件，后续可通过 pd-ctl 修改配置。',
             deploy, ['没有使用 `--config` 参数来指定配置文件', '对于 PD 组件，只会在第一次启动时读取配置文件'], ['toml', 'TiKV', 'PD'], '文档说明的 binary 部署场景；不套用于 TiUP 配置管理'),
        item('EV14', 'test', '部署与配置', 'direct_io_cache_bypass', '现象与精确标识',
             '磁盘压测结果怀疑测到的是操作系统缓存。dd 的 oflag=direct 为什么有助于测实际磁盘读写能力？',
             'Direct 模式将写入请求直接发给磁盘，绕开文件系统缓存，减少缓存对磁盘 I/O 能力测量的影响。',
             deploy, ['这样是为了绕开文件系统的缓存，可以直接测试磁盘的真实的 I/O 读写能力。'], ['dd', 'oflag=direct']),
        item('EV15', 'test', '部署与配置', 'monitoring_clock_skew', '口语现象',
             '服务在运行，但监控页面只有部分数据能显示。除了查采集服务，有什么与机器时间有关的检查值得先做？',
             '检查访问监控的机器与集群机器的时间差；文档指出时间差较大时，更正时间可以恢复显示。',
             deploy, ['查看访问监控的机器时间跟集群内机器的时间差，如果比较大，更正时间后即可显示正常。']),
        item('EV16', 'test', 'SQL 与业务迁移', 'unordered_result_contract', '口语现象',
             '迁移后同一条查询没写排序条件，返回行的先后顺序和原数据库不同。能否据此认定结果错误，应用该怎样保证顺序？',
             '不能。没有 ORDER BY 时行顺序不保证一致；应用依赖固定顺序时应显式使用 ORDER BY。',
             sql, ['返回结果的顺序视不同情况而定，不保证顺序统一。', '都推荐使用 `ORDER BY` 条件。']),
        item('EV17', 'test', 'SQL 与业务迁移', 'delete_space_compaction', '口语现象',
             '业务删除大量行后，为什么不能期待磁盘空间马上下降？对普通行删除，清理旧数据之后还要等待什么过程？',
             'DELETE 不会立即释放空间；GC 删除数据后，还要等待后续 compaction 才释放空间，不能把数据删除等同于磁盘立即回收。',
             sql, ['对于 DELETE 操作，TiDB 的 GC 机制会删除数据，但不会立即释放空间，而是等到后续进行 compaction 时释放空间。']),
        item('EV18', 'test', 'SQL 与业务迁移', 'online_index_backfill', '口语现象',
             '大表加索引耗时很长，团队担心线上读写完全被阻塞。文档怎样解释在线建索引的耗时与资源影响？',
             'TiDB 支持在线 DDL，文档说明不阻塞 DML；添加索引需要 backfill，耗时取决于表大小并消耗额外资源，对业务影响可调节，不能理解成零开销。',
             sql, ['这些 DDL 操作不会阻塞 DML 操作。', 'TiDB 需要进行回填 (backfill) 操作', '额外的资源消耗。对在线业务的影响可调节。']),
        item('EV19', 'test', '连接与启动', 'cluster_id_mismatch', '现象与精确标识',
             '重建 PD 后仍用原来的 TiKV 数据目录启动，报 cluster ID mismatch。这个错误体现了什么元信息不一致？',
             'TiKV 本地保存的 cluster ID 与新 PD 的 cluster ID 不一致。新 PD 会生成新的 ID，原 TiKV 初始化时存下的是旧 ID，启动检查不一致会退出。',
             'faq/manage-cluster-faq.md', ['TiKV 本地存储的 cluster ID 和指定的 PD 的 cluster ID 不一致。', 'PD 会随机生成一个 cluster ID'], ['cluster ID mismatch'], '只解释原因；不建议或执行删除生产数据'),
        item('EV20', 'test', '备份与迁移', 'large_backup_tool_choice', '口语现象',
             '团队要给超过 1 TB 的库做备份，现有资料优先推荐哪个工具？为什么不优先选通用的逻辑导出工具？',
             '文档对大于 1 TB 的场景推荐 Backup & Restore（BR）；并指出 mysqldump 的性能低于 Dumpling，备份恢复大量数据耗时更长。该建议来自固定资料，不代替生产方案评审。',
             'faq/manage-cluster-faq.md', ['大于 1 TB）推荐使用 [Backup & Restore (BR)]', '`mysqldump` 备份和恢复大量数据的耗费更长。']),
        item('EV21', 'test', '备份与迁移', 'staged_mysql_migration', '口语现象',
             'MySQL 业务要迁入 TiDB，希望数据迁移和读写流量切换分开推进。资料建议怎样组织迁移工具和网络入口？',
             '用 TiDB Data Migration 进行 MySQL 到 TiDB 数据迁移；业务读写可按需分阶段通过网络配置切换，在数据库上层放稳定的 LB 作为入口。',
             'faq/migration-tidb-faq.md', ['进行 MySQL -> TiDB 的业务数据的迁移', '业务读写可以按照需求分阶段通过修改网络配置进行流量迁移', '建议 DB 上层部署一个稳定的网络 LB']),
        item('EV22', 'test', '容量与流控', 'io_hotspot_vs_global_load', '口语现象',
             '监控里一台机器的磁盘占用率很高，与大部分机器都很高，这两种分布分别提示什么排查方向？',
             '只有一台机器 I/O 高时可辅助判断读写热点；大部分机器都高时提示集群整体 I/O 负载高，需进一步用磁盘延迟、负载等监控验证。',
             'troubleshoot-high-disk-io.md', ['只有一台机器的 I/O 高，那么可以辅助判断当前有读写热点。', '大部分机器的 I/O 都高，那么集群现在有高 I/O 负载存在。']),
        item('EV23', 'test', '写入热点', 'auto_random_business_id_boundary', '现象与精确标识',
             '团队打算用 AUTO_RANDOM 缓解自增主键热点，但业务还依赖主键按生成顺序递增。这个改法满足要求吗，怎样取得刚分配的主键？',
             '不满足递增要求。AUTO_RANDOM 主键不再自增，适合只要求唯一、没有业务含义的主键；可用 LAST_INSERT_ID() 取得上次分配值。',
             'troubleshoot-hot-spot-issues.md', ['TiDB 生成的主键不再是自增的主键', '`LAST_INSERT_ID()` 获取上次分配的主键值。', '适用于主键只需要保证唯一，不包含业务意义的场景。'], ['AUTO_RANDOM']),
        item('EV24', 'test', '事务与锁', 'explicit_transaction_mode_override', '现象与精确标识',
             '会话的 tidb_txn_mode 已有设置，但某次业务要显式启动另一种事务模式。BEGIN PESSIMISTIC 或 BEGIN OPTIMISTIC 与变量设置谁优先？',
             '这两个显式 BEGIN 语句优先级高于 tidb_txn_mode，开启的事务忽略系统变量，因此支持乐观与悲观模式混用。',
             'pessimistic-transaction.md', ['`BEGIN PESSIMISTIC;` 和 `BEGIN OPTIMISTIC;` 等语句的优先级高于 `tidb_txn_mode` 系统变量。'], ['tidb_txn_mode', 'BEGIN PESSIMISTIC', 'BEGIN OPTIMISTIC']),
    ]


def main():
    manifest_path = CORPUS / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    sources = {item['local_path']: item for item in manifest['files'] if item['category'] != '出处与许可'}
    texts = {}
    for name, source in sources.items():
        data = (CORPUS / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise ValueError(f'Source changed: {name}')
        texts[name] = data.decode('utf-8')
    questions = rows()
    if len({q['id'] for q in questions}) != len(questions) or len({q['question'] for q in questions}) != len(questions):
        raise ValueError('Duplicate question or ID')
    families = {split: {q['answer_family'] for q in questions if q['split'] == split} for split in ('dev', 'test')}
    if families['dev'] & families['test']:
        raise ValueError('Answer family overlaps between development and validation')
    for q in questions:
        text = texts[q['filename']]
        q['source_sha256'] = sources[q['filename']]['sha256']
        q['source_url'] = sources[q['filename']]['source_url']
        q['anchors'] = []
        for quote in q.pop('quotes'):
            start = text.find(quote)
            if start < 0:
                raise ValueError(f"Missing anchor {q['id']}: {quote}")
            q['anchors'].append({'filename': q['filename'], 'quote': quote, 'start_char': start,
                                 'end_char': start + len(quote), 'line_start': text.count('\n', 0, start) + 1,
                                 'line_end': text.count('\n', 0, start + len(quote)) + 1})
        q['review_status'] = 'AI source-anchored draft; user semantic review pending'
        q['gold_chunk_ids'] = None
        q['gold_mapping_status'] = 'pending_new_index_and_equivalent_evidence_review'
    counts = dict(Counter(q['split'] for q in questions))
    if counts != {'dev': 8, 'test': 24}:
        raise ValueError(f'Unexpected split sizes: {counts}')
    output = {
        'schema_version': 1, 'dataset_id': 'enterprise_tidb_v2', 'status': 'draft_not_frozen',
        'purpose': '企业运维与技术支持场景的定向检索对照；RRF 优势是待验证假设',
        'provenance': 'AI 根据固定官方资料拟定的模拟业务问题；非客户工单或独立人工标注',
        'repository': manifest['repository'], 'commit': manifest['commit'],
        'source_manifest': str(manifest_path.relative_to(ROOT)).replace('\\', '/'),
        'source_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'counts': counts, 'question_count': len(questions), 'source_documents_in_index_plan': len(sources),
        'source_documents_with_questions': len({q['filename'] for q in questions}),
        'topic_counts': dict(Counter(q['topic'] for q in questions)),
        'kind_counts': dict(Counter(q['kind'] for q in questions)),
        'split_policy': {
            'unit': 'answer_family', 'answer_family_overlap': [],
            'qualification': '同一领域语料中的问题留出；共享部分源文档，不是独立文档或真实生产流量留出',
            'dev': '用于连通性、评分标签、检索策略及参数开发；保留全部开发结果',
            'test': '24 道候选题待复核、建立等价依据和冻结；冻结前不运行检索',
        },
        'scoring_plan': {
            'primary': 'Hit Rate@5：前五条中是否有包含足够回答依据的片段；等价原文必须纳入标签',
            'incomplete_evidence': '先记录需跨块组合的题；不把只含报错名的片段当完整依据，也不按结果事后删题',
            'mapping': '入库后按原文锚点与语义复核映射真实 chunk ID；本草稿不伪造 ID 或沿用旧库 ID',
            'comparison': '相同语料、解析与切分、题目、候选数和 Top K；保留向量、BM25、RRF 全部结果',
        },
        'paid_model_calls': 0, 'retrieval_executed': False, 'questions': questions,
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    destination = DESTINATION / 'questions_draft.json'
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    fields = ['id', 'split', 'topic', 'kind', 'question', 'reference_answer', 'applicability',
              'filename', 'source_lines', 'source_quotes', 'review_status']
    with (DESTINATION / 'questions_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for q in questions:
            writer.writerow({**{key: q[key] for key in fields if key in q},
                             'source_lines': '; '.join(f"{a['line_start']}-{a['line_end']}" for a in q['anchors']),
                             'source_quotes': '\n'.join(a['quote'] for a in q['anchors'])})
    report = {key: output[key] for key in ('dataset_id', 'status', 'counts', 'question_count',
              'source_documents_in_index_plan', 'source_documents_with_questions', 'topic_counts', 'kind_counts',
              'paid_model_calls', 'retrieval_executed')}
    report.update(source_hash_checks=len(sources), anchor_checks=sum(len(q['anchors']) for q in questions),
                  dataset_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (DESTINATION / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
