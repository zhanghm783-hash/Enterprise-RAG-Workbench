"""预先固定24条企业技术FAQ；继承原文答案与真实依据，不调用模型。"""
from __future__ import annotations

from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
QUESTIONS=[
    'HashAgg内存消耗大，有什么替代聚合算子？',
    'TiDB执行大事务时，内存需要预留多少余量？事务太大如何处理？',
    'JDBC读取大结果集导致客户端OOM，应检查哪个流式读取参数？',
    'Lock View支持哪些锁冲突信息？能还原带原始参数和格式的SQL吗？',
    'TiDB热点key的悲观锁争用如何启用公平锁？有什么性能代价？',
    '悲观锁TTL超时应检查哪个配置？长事务可以怎样调整？',
    '慢查询中Wait_TS占了大部分Query_time，应优先检查哪个阶段？',
    '非关联子查询在优化阶段提前执行，为什么可能没体现在执行算子耗时中？',
    '单个TiKV的cop-task排队时间远高于其他实例，应结合什么监控排查？',
    '客户端KV duration很高、服务端gRPC duration很低，应怀疑什么瓶颈？',
    'TiDB组件启动报Too many open files，应检查哪个系统限制？',
    '两个TiKV实例能共用一个数据库文件目录吗？',
    'TiKV和PD修改配置文件后不生效，应检查什么？PD重启会重新读配置文件吗？',
    '用dd测试磁盘性能，Direct模式为什么能减少缓存干扰？',
    '监控采集正常但Grafana缺曲线，访问机器和集群的时间差会有影响吗？',
    'TiDB查询不加ORDER BY，返回顺序有保证吗？怎样固定排序？',
    'TiDB执行DELETE后磁盘空间没立即释放，还需要等待什么？',
    'TiDB在线添加索引为什么耗时并占用资源？是否会阻塞DML？',
    '重建PD后旧TiKV报cluster ID不匹配，是什么原因？',
    '备份超过1TB的TiDB数据推荐什么工具？mysqldump有什么效率问题？',
    'MySQL迁移到TiDB可用什么工具？分阶段切流如何配合稳定的LB入口？',
    '单台机器I/O高和大部分机器I/O都高，分别提示什么排查方向？',
    'AUTO_RANDOM主键还能保证自增顺序吗？怎样取回上次分配的ID？',
    '显式BEGIN PESSIMISTIC或OPTIMISTIC与tidb_txn_mode，谁的优先级更高？',
]


def main():
    parent=ROOT/'evaluation/datasets/enterprise_semantic_paraphrase_v1/questions.json'
    parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
    assert parent_sha=='4f4549d7002d07371f31e474e5869f807183cdde52a41fbea4ebb8f3c54aa10f'
    original=json.loads(parent.read_text(encoding='utf-8'))
    rows=sorted(original['questions'],key=lambda q:q['id']);assert len(rows)==len(QUESTIONS)==24
    revised=[]
    for old,question in zip(rows,QUESTIONS):
        row=dict(old);row['parent_question']=old['question'];row['question']=question;revised.append(row)
        assert row['reference_answer']==old['reference_answer']
    dest=ROOT/'evaluation/datasets/enterprise_quality_faq_v1'
    if dest.exists():raise ValueError('Frozen FAQ version already exists; do not overwrite')
    dest.mkdir(parents=True)
    result={'dataset':'enterprise_quality_faq_v1','frozen_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
            'parent_sha256':parent_sha,'scope':'AI-authored concise enterprise technical FAQ development cohort; not independent new tasks or production traffic',
            'selection':'All 24 parent tasks retained; wording shortened and technical terms made explicit before new runs',
            'before_after_rule':'Both baseline and improved systems must use these same frozen questions; do not splice parent-query baseline scores',
            'references_and_gold_changed':False,'questions':revised}
    path=dest/'questions.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    (dest/'questions.sha256').write_text(sha+'\n',encoding='utf-8')
    print(json.dumps({'status':'frozen','questions':24,'sha256':sha,'model_calls':0,'retrievals_run':0},ensure_ascii=False))


if __name__=='__main__':main()
